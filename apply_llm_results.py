"""
Apply AI Geologist Results to Cross-Sections and Generate HTML Report
AI 분석 결과를 단면도에 적용 및 HTML 보고서 생성
"""

import os
import sys
import io
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, io.UnsupportedOperation):
    pass

import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import LineString, Point
from pathlib import Path
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import base64
from io import BytesIO
from datetime import datetime

# 범용 지질학 원리 모듈 (한반도 제외 형태 검증용)
try:
    from geological_principles import KOREA_SPECIFIC_EXCLUSIONS
    PRINCIPLES_AVAILABLE = True
except ImportError:
    KOREA_SPECIFIC_EXCLUSIONS = {}
    PRINCIPLES_AVAILABLE = False

# Load configuration (supports CLI arguments: --config, --data-dir, --output-dir, --dem-file)
from config_loader import init_config

CONFIG = init_config()
PATHS = CONFIG['paths']
SHAPEFILES = CONFIG['shapefiles']

# Paths from config
BASE_DIR = PATHS['data_dir']
GEOLOGY_DIR = PATHS['geology_dir']
DEM_FILE = PATHS['dem_file']
OUTPUT_DIR = PATHS['output_dir']

TARGET_CRS = CONFIG['crs']

# 지역명 (reconfigure()로 업데이트됨)
REGION_NAME_KR = "서울"
REGION_NAME_EN = "Seoul"

# Korean font
viz_params = CONFIG['visualization']
plt.rcParams['font.family'] = viz_params.get('font_family', 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# Colors from config (with override for Qa)
LITHO_COLORS = CONFIG['litho_colors'].copy()
LITHO_COLORS['Qa'] = '#FFFF00'  # 밝은 노란색으로 변경 (충적층 강조)

# Names from config (with override for PCEls)
LITHO_NAMES = CONFIG['litho_names_kr'].copy()
LITHO_NAMES['PCEls'] = '결정질석회암'  # 상세 명칭

CONTACT_COLORS = {
    'intrusive': '#FF4444',
    'unconformable': '#4444FF',
    'conformable': '#44AA44',
    'fault': '#000000'
}

# 암석 시대 순서 (숫자가 작을수록 오래됨, 먼저 그림)
# 오래된 것이 먼저 그려지고, 젊은 것이 위에 덮임
LITHO_AGE_ORDER = {
    # 선캠브리아기 기저 변성암 (가장 오래됨) - 가장 먼저 그림 (맨 뒤)
    'PCEbngn': 0,  # 호상흑운모편마암 - 최고령 기반암
    # 선캠브리아기 변성암 (PCEbngn 위에 퇴적/관입)
    'PCEls': 1,    # 결정질석회암
    'PCEqz': 1,    # 규암
    'PCEam': 1,    # 각섬암
    # 선캠브리아기 관입 기원 변성암
    'PCEggn': 2,   # 화강편마암 (원암: 화강암 관입체)
    # 쥬라기 관입암
    'Jbgr': 3,     # 흑운모화강암
    # 백악기 관입암
    'Pgr': 4,      # 반상화강암
    'Kqp': 4,      # 석영반암
    'Kqv': 4,      # 석영맥
    'Kfl': 4,      # 규장암
    # 제4기 (가장 젊음) - 마지막에 그림 (맨 앞)
    'Qa': 5,       # 충적층
}


def load_data():
    """Load all required data"""
    global LITHO_COLORS, LITHO_NAMES, LITHO_AGE_ORDER
    print("Loading data...")

    shp_paths = CONFIG['shapefile_paths']
    data = {}

    # read_shapefile_safe: SHAPE_ENCODING='' 로 인코딩 자동감지
    from config_loader import read_shapefile_safe, auto_populate_litho_info, auto_populate_age_order
    data['litho'] = read_shapefile_safe(shp_paths['litho'], target_crs=TARGET_CRS)
    data['boundary'] = gpd.read_file(shp_paths['boundary']).to_crs(TARGET_CRS)
    data['fault'] = gpd.read_file(shp_paths['fault']).to_crs(TARGET_CRS)

    # Shapefile에서 미등록 암상 코드의 색상·이름·시대순서 자동 보충
    auto_populate_litho_info(data['litho'], LITHO_COLORS, LITHO_NAMES)
    auto_populate_age_order(data['litho'], LITHO_AGE_ORDER)

    # Load LLM results
    with open(OUTPUT_DIR / "llm_geologist_results.json", 'r', encoding='utf-8') as f:
        data['llm_results'] = json.load(f)

    # Load section definitions
    with open(OUTPUT_DIR / "crosssection_analysis.json", 'r', encoding='utf-8') as f:
        section_data = json.load(f)
        data['sections'] = section_data['existing_crosssections'] + section_data['auto_crosssections']

    print(f"  Lithology: {len(data['litho'])}")
    print(f"  LLM analyzed boundaries: {len(data['llm_results']['boundaries'])}")
    print(f"  Sections: {len(data['sections'])}")

    return data


def _load_kb_module():
    """Find and return the active geological knowledge base module from sys.modules"""
    for name, mod in sys.modules.items():
        if 'geological_knowledge' in name and hasattr(mod, 'ROCK_UNITS'):
            return mod
    return None


# body_geometry 캐시 (반복 호출 최적화)
_BODY_GEOMETRY_CACHE = {}


def _validate_korea_geometry(geo):
    """한반도 제외 형태 검증. 부적용 형태이면 안전한 기본값으로 대체."""
    if geo in KOREA_SPECIFIC_EXCLUSIONS:
        fallback = KOREA_SPECIFIC_EXCLUSIONS[geo].get('safe_fallback', 'stock')
        return fallback
    return geo


def _get_body_geometry(litho_code, intrusion_shape=None, unit_width=None):
    """암체의 산출 형태를 판정.

    판정 우선순위:
      1. LLM 분석 결과 intrusion_shape (가장 높은 우선순위)
      2. KB body_geometry (새 필드)
      3. KB expected_contact_type == 'intrusive' → 크기 기반 휴리스틱
      4. 기본값: 'conformable'

    모든 결과는 한반도 부적용 형태(lopolith, phacolith) 검증을 거침.

    Returns: 'batholith'|'stock'|'dome'|'plug'|'dike'|'sill'|'flow'|'conformable'
    """
    # 1) LLM 분석 결과 (가장 높은 우선순위)
    if intrusion_shape and intrusion_shape != 'null' and intrusion_shape != 'unknown':
        return _validate_korea_geometry(intrusion_shape)

    # 캐시 확인 (LLM 결과 없는 경우에만)
    if litho_code in _BODY_GEOMETRY_CACHE:
        return _BODY_GEOMETRY_CACHE[litho_code]

    # 2) KB body_geometry
    kb = _load_kb_module()
    if kb:
        mapping = getattr(kb, 'LITHOIDX_TO_KB', {})
        kb_code = mapping.get(litho_code, litho_code)
        rock_units = getattr(kb, 'ROCK_UNITS', {})
        if kb_code in rock_units:
            geo = rock_units[kb_code].get('body_geometry')
            if geo:
                geo = _validate_korea_geometry(geo)
                _BODY_GEOMETRY_CACHE[litho_code] = geo
                return geo
            # body_geometry가 없으면 expected_contact_type으로 추정
            if rock_units[kb_code].get('expected_contact_type') == 'intrusive':
                # 3) 크기 기반 휴리스틱
                if unit_width is not None:
                    if unit_width < 200:
                        geo = 'dike'
                    elif unit_width > 2000:
                        geo = 'batholith'
                    else:
                        geo = 'stock'
                else:
                    geo = 'stock'  # 관입암 기본
                _BODY_GEOMETRY_CACHE[litho_code] = geo
                return geo

    # 4) 기본값
    _BODY_GEOMETRY_CACHE[litho_code] = 'conformable'
    return 'conformable'


def _parse_dip_range(dip_range_str):
    """Parse expected_dip_range string to a numeric dip angle"""
    import re
    if not dip_range_str:
        return None
    s = str(dip_range_str)
    # "20-25°NE" or "20-40°" → average
    m = re.search(r'(\d+)\s*[-~]\s*(\d+)', s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    # "관입암체" or "급경사" → steep contact
    if '관입' in s or '급경사' in s:
        return 75
    # "맥상" → dyke-like steep
    if '맥상' in s or '맥' in s:
        return 80
    # "수평" → nearly flat
    if '수평' in s:
        return 5
    # "변화무쌍" → variable, moderate
    if '변화' in s or '무쌍' in s:
        return 40
    # "다양" → various, moderate
    if '다양' in s:
        return 55
    # Single number like "0-5°"
    m = re.search(r'(\d+)', s)
    if m:
        return int(m.group(1))
    return None


def _classify_pair_from_kb(litho1, litho2):
    """
    Classify contact type and estimate dip angle from Knowledge Base.
    Returns (contact_type, dip_angle) or (None, None) if KB not available.
    """
    kb_mod = _load_kb_module()
    if not kb_mod:
        return None, None

    rock_units = getattr(kb_mod, 'ROCK_UNITS', {})
    idx_to_kb = getattr(kb_mod, 'LITHOIDX_TO_KB', {})
    contact_rules = getattr(kb_mod, 'CONTACT_RULES', {})

    # Map shapefile LITHOIDX to KB code (direct match first, then via mapping)
    def to_kb_code(litho):
        if litho in rock_units:
            return litho
        return idx_to_kb.get(litho)

    kb1 = to_kb_code(litho1)
    kb2 = to_kb_code(litho2)

    # 1) Try CONTACT_RULES for pair-specific classification
    if kb1 and kb2 and contact_rules:
        for rule_type in ['unconformable_pairs', 'intrusive_pairs', 'conformable_pairs']:
            pairs_list = contact_rules.get(rule_type, [])
            for p in pairs_list:
                if (p[0] == kb1 and p[1] == kb2) or (p[0] == kb2 and p[1] == kb1):
                    contact_type = rule_type.replace('_pairs', '')
                    # Get pair-specific dip from KB helper
                    get_dip_fn = getattr(kb_mod, 'get_expected_dip_range', None)
                    dip_str = get_dip_fn(kb1, kb2) if get_dip_fn else None
                    dip_angle = _parse_dip_range(dip_str)
                    return contact_type, dip_angle

    # 2) Fallback: individual rock's expected_contact_type
    # unconformable > intrusive > conformable
    contact_type = None
    dip_angle = None

    for litho_code in [kb1, kb2]:
        if litho_code and litho_code in rock_units:
            unit = rock_units[litho_code]
            ct = unit.get('expected_contact_type')
            if ct == 'unconformable':
                contact_type = 'unconformable'
                dip_angle = _parse_dip_range(unit.get('expected_dip_range'))
                break

    if contact_type is None:
        for litho_code in [kb1, kb2]:
            if litho_code and litho_code in rock_units:
                unit = rock_units[litho_code]
                ct = unit.get('expected_contact_type')
                if ct == 'intrusive':
                    contact_type = 'intrusive'
                    dip_angle = _parse_dip_range(unit.get('expected_dip_range'))
                    break

    if contact_type is None:
        for litho_code in [kb1, kb2]:
            if litho_code and litho_code in rock_units:
                unit = rock_units[litho_code]
                ct = unit.get('expected_contact_type')
                if ct == 'conformable':
                    contact_type = 'conformable'
                    # For conformable, average both units' dip ranges if available
                    dips = []
                    for kc in [kb1, kb2]:
                        if kc and kc in rock_units:
                            d = _parse_dip_range(rock_units[kc].get('expected_dip_range'))
                            if d is not None:
                                dips.append(d)
                    dip_angle = sum(dips) / len(dips) if dips else None
                    break

    return contact_type, dip_angle


def get_dip_for_litho_pair(litho1, litho2, llm_results):
    """Get dip estimate for a lithology pair from LLM results and Knowledge Base.

    Priority chain:
      1. LLM analysis results (exact boundary match)
      2. Knowledge Base ROCK_UNITS (dynamic, via LITHOIDX_TO_KB mapping)
      3. Hardcoded base (accumulated across regions, expandable)
      4. Generic fallback
    """

    def extract_full_info(boundary):
        """Extract all geometry info from boundary"""
        depth_behavior = boundary.get('depth_behavior', {})
        fold_influence = boundary.get('fold_influence', {})
        return {
            'dip_angle': boundary['dip_angle'],
            'dip_direction': boundary['dip_direction'],
            'contact_type': boundary.get('contact_type', 'unknown'),
            'confidence': boundary['confidence'],
            'reasoning': boundary.get('reasoning', ''),
            # New fields
            'contact_geometry': boundary.get('contact_geometry', 'planar'),
            'depth_behavior': depth_behavior.get('dip_change', 'constant'),
            'depth_to_flatten': depth_behavior.get('estimated_depth_to_flatten'),
            'intrusion_shape': boundary.get('intrusion_shape'),
            'fold_present': fold_influence.get('present', False),
            'fold_type': fold_influence.get('fold_type'),
            'fold_axis': fold_influence.get('axis_trend'),
        }

    # Priority 2: Knowledge Base classification (dynamic) - 먼저 계산하여 교차검증용으로 사용
    expected_type, kb_dip = _classify_pair_from_kb(litho1, litho2)

    # Priority 1: Both lithologies match exactly in LLM results
    # 단, KB 분류와 교차검증: LLM이 3종 이상 암상을 묶어 잘못 분류하는 것 방지
    for boundary in llm_results['boundaries']:
        lithos = boundary.get('adjacent_lithologies', [])
        if litho1 in lithos and litho2 in lithos:
            llm_type = boundary.get('contact_type', 'unknown')
            # KB가 분류한 접촉유형과 LLM이 일치하면 LLM 결과 사용
            if expected_type is None or llm_type == expected_type:
                return extract_full_info(boundary)
            # 불일치 시 LLM 결과 무시하고 KB 기반으로 폴스루
            break

    # Priority 3: Hardcoded base (accumulated across regions)
    if expected_type is None:
        # Seoul region rocks
        intrusive_rocks = {'Pgr', 'Jbgr', 'Kqp', 'Kqv', 'Kfl'}
        quaternary = {'Qa', 'Qal'}
        metamorphic = {'PCEbngn', 'PCEggn', 'PCEls', 'PCEqz', 'PCEam'}
        # Busan region rocks - 심성암/맥암만 (화산암은 conformable_volcanic로)
        intrusive_rocks |= {'Kbgr', 'Khgdi', 'Kga', 'Kgp', 'Kad'}
        conformable_volcanic = {'Kan', 'Kanb', 'Kts', 'Kdban', 'Kdlw', 'Kdtb', 'Kdup',
                                'Krb', 'Krwt', 'Krt', 'Krh'}

        if litho1 in quaternary or litho2 in quaternary:
            expected_type = 'unconformable'
        elif litho1 in intrusive_rocks or litho2 in intrusive_rocks:
            expected_type = 'intrusive'
        elif litho1 in metamorphic and litho2 in metamorphic:
            expected_type = 'conformable'
        elif litho1 in conformable_volcanic or litho2 in conformable_volcanic:
            expected_type = 'conformable'

    # KB/하드코딩 분류 결과를 직접 사용 (LLM 부분매칭 제거)
    # LLM은 소수 경계만 분석하므로 부분매칭 시 부정확한 75° 경사를
    # KB의 정밀한 경사각(20-25° 등)에 덮어쓰는 문제가 있었음

    # Priority 4: Use typical values based on contact type
    default_info = {
        'confidence': 'low',
        'reasoning': '지식베이스 기반 분류' if kb_dip else '전형값 사용',
        'contact_geometry': 'planar',
        'depth_behavior': 'constant',
        'depth_to_flatten': None,
        'intrusion_shape': None,
        'fold_present': False,
        'fold_type': None,
        'fold_axis': None,
    }

    if expected_type == 'unconformable':
        angle = kb_dip or 8
        return {**default_info, 'dip_angle': angle, 'dip_direction': 180, 'contact_type': 'unconformable'}
    elif expected_type == 'intrusive':
        angle = kb_dip or 75
        return {**default_info, 'dip_angle': angle, 'dip_direction': 135, 'contact_type': 'intrusive',
                'contact_geometry': 'curved', 'depth_behavior': 'flattening'}
    elif expected_type == 'conformable':
        angle = kb_dip or 30
        return {**default_info, 'dip_angle': angle, 'dip_direction': 135, 'contact_type': 'conformable'}

    return {**default_info, 'dip_angle': 55, 'dip_direction': 135, 'contact_type': 'unknown',
            'confidence': 'very_low'}


def extract_profile(section, sample_interval=50):
    """Extract terrain profile"""
    start = (section['start']['x'], section['start']['y'])
    end = (section['end']['x'], section['end']['y'])

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)
    num_samples = int(length / sample_interval) + 1

    distances = np.linspace(0, length, num_samples)
    xs = np.linspace(start[0], end[0], num_samples)
    ys = np.linspace(start[1], end[1], num_samples)

    elevations = []
    with rasterio.open(DEM_FILE) as dem:
        dem_data = dem.read(1)
        transform = dem.transform
        nodata = dem.nodata

        # DEM CRS와 TARGET_CRS가 다르면 좌표 변환 준비
        dem_crs = CRS(dem.crs) if dem.crs else None
        target_crs = CRS(TARGET_CRS)
        need_transform = dem_crs is not None and dem_crs != target_crs
        if need_transform:
            coord_transformer = Transformer.from_crs(target_crs, dem_crs, always_xy=True)

        for x, y in zip(xs, ys):
            try:
                # TARGET_CRS → DEM CRS 변환 (필요 시)
                dx, dy = coord_transformer.transform(x, y) if need_transform else (x, y)
                row, col = rowcol(transform, dx, dy)
                if 0 <= row < dem.height and 0 <= col < dem.width:
                    elev = dem_data[row, col]
                    if nodata and elev == nodata:
                        elevations.append(np.nan)
                    else:
                        elevations.append(float(elev))
                else:
                    elevations.append(np.nan)
            except:
                elevations.append(np.nan)

    elevations = np.array(elevations)
    valid = ~np.isnan(elevations)
    if np.sum(valid) > 0 and np.sum(~valid) > 0:
        elevations[~valid] = np.interp(distances[~valid], distances[valid], elevations[valid])

    return distances, elevations, list(zip(xs, ys)), length


def project_geology_with_llm_dip(section, data, profile_data):
    """Project geology with LLM-derived dip angles"""

    distances, elevations, coords, length = profile_data
    section_az = section['azimuth']

    start = (section['start']['x'], section['start']['y'])
    end = (section['end']['x'], section['end']['y'])
    section_line = LineString([start, end])

    projections = []

    for idx, row in data['litho'].iterrows():
        geom = row.geometry
        if geom is None or not geom.is_valid:
            continue

        intersection = section_line.intersection(geom)
        if intersection.is_empty:
            continue

        litho_idx = row.get('LITHOIDX', 'Unknown')

        segments = []
        if intersection.geom_type == 'LineString':
            segments = [intersection]
        elif intersection.geom_type == 'MultiLineString':
            segments = list(intersection.geoms)

        for seg in segments:
            if seg.length < 10:
                continue

            seg_coords = list(seg.coords)
            start_dist = section_line.project(Point(seg_coords[0]))
            end_dist = section_line.project(Point(seg_coords[-1]))

            if start_dist > end_dist:
                start_dist, end_dist = end_dist, start_dist

            # Get elevation at boundaries
            start_idx = min(np.searchsorted(distances, start_dist), len(elevations)-1)
            end_idx = min(np.searchsorted(distances, end_dist), len(elevations)-1)

            start_elev = elevations[start_idx]
            end_elev = elevations[end_idx]

            # Find adjacent lithology for dip lookup
            buffer = seg.buffer(100)
            adjacent = data['litho'][data['litho'].intersects(buffer)]
            adjacent_lithos = [r.get('LITHOIDX') for _, r in adjacent.iterrows() if r.get('LITHOIDX') != litho_idx]
            adjacent_litho = adjacent_lithos[0] if adjacent_lithos else None

            # Get dip from LLM results
            dip_info = get_dip_for_litho_pair(litho_idx, adjacent_litho, data['llm_results'])

            # Debug: print which boundaries get which depth behavior
            if dip_info.get('depth_behavior') != 'constant':
                print(f"    [{litho_idx}-{adjacent_litho}] depth_behavior={dip_info.get('depth_behavior')}, geometry={dip_info.get('contact_geometry')}")

            # Calculate apparent dip
            true_dip = dip_info['dip_angle']
            dip_az = dip_info['dip_direction']
            angle_diff = np.radians(dip_az - section_az)
            apparent_dip = np.degrees(np.arctan(np.tan(np.radians(true_dip)) * np.cos(angle_diff)))

            projections.append({
                'litho_idx': litho_idx,
                'start_dist': float(start_dist),
                'end_dist': float(end_dist),
                'start_elev': float(start_elev),
                'end_elev': float(end_elev),
                'true_dip': true_dip,
                'apparent_dip': apparent_dip,
                'contact_type': dip_info['contact_type'],
                'confidence': dip_info['confidence'],
                'width': float(end_dist - start_dist),
                # New geometry fields for curved boundaries
                'contact_geometry': dip_info['contact_geometry'],
                'depth_behavior': dip_info['depth_behavior'],
                'depth_to_flatten': dip_info['depth_to_flatten'],
                'intrusion_shape': dip_info['intrusion_shape'],
                'fold_present': dip_info['fold_present'],
                'fold_type': dip_info['fold_type'],
                'fold_axis': dip_info['fold_axis'],
            })

    projections.sort(key=lambda x: x['start_dist'])
    return projections


def add_natural_irregularity(coords, amplitude=20, frequency=3):
    """Add natural irregularity to boundary coordinates"""
    if len(coords) < 3:
        return coords

    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    # Add subtle waviness using sine waves with different frequencies
    for i in range(1, len(x_coords) - 1):  # Don't modify endpoints
        progress = i / len(x_coords)
        # Multiple harmonics for natural look
        noise = (np.sin(progress * frequency * np.pi) * amplitude * 0.6 +
                 np.sin(progress * frequency * 2.3 * np.pi) * amplitude * 0.3 +
                 np.sin(progress * frequency * 5.7 * np.pi) * amplitude * 0.1)
        x_coords[i] += noise

    return list(zip(x_coords, y_coords))


def generate_intrusion_boundary(start_d, start_e, min_elev, base_dip, is_left_boundary, length, unit_width=None, expansion_factor=1.5):
    """
    Generate realistic intrusion boundary (batholith/stock shape).

    관입체 저반(Batholith) 형태 - 수직과장(vertical exaggeration) 고려:
    - 표면에서 40-45° 외향 경사 (수직과장으로 60-70°처럼 보임)
    - 심부로 갈수록 급격히 확장 (플레어 형태)
    - 불규칙한 경계 (자연스러운 굴곡)

    Args:
        is_left_boundary: True면 왼쪽 경계, False면 오른쪽 경계
        unit_width: 암체의 지표 폭 (심부 확장 계산에 사용)
        expansion_factor: 심부 확장 배율 (batholith=5, stock=3)
    """
    total_depth = start_e - min_elev
    num_points = 60

    # 관입체 표면 접촉각: base_dip 기반 (KB에서 전달된 경사각 사용)
    # 범위 제한: 25-50° (너무 완만하거나 수직벽이 되지 않도록)
    surface_dip = max(25, min(50, abs(base_dip))) if base_dip else 35

    depths = np.linspace(0, total_depth, num_points)
    x_coords = [start_d]
    y_coords = [start_e]

    # 심부 확장량 (expansion_factor에 의해 조절)
    effective_width = unit_width or 1000
    max_expansion = effective_width * expansion_factor

    # 경사 변화 마일스톤 (surface_dip 기반으로 동적 계산)
    mid_dip = surface_dip * 0.5       # 중부: 표면 경사의 50%
    bottom_dip = max(5, surface_dip * 0.15)  # 하부: 표면의 15%, 최소 5°

    for i, depth in enumerate(depths[1:], 1):
        progress = depth / total_depth  # 0 to 1

        # 비선형 경사 변화: 심부로 갈수록 급격히 완만해짐
        # 상부 20%: surface_dip에서 약간 감소
        # 중부 30%: mid_dip까지 점진 감소
        # 하부 50%: bottom_dip까지 (거의 수평 플레어)
        if progress < 0.2:
            local_dip = surface_dip - progress * (surface_dip - mid_dip) / 0.2 * 0.3
        elif progress < 0.5:
            upper_end = surface_dip * 0.85
            mid_progress = (progress - 0.2) / 0.3
            local_dip = upper_end - mid_progress * (upper_end - mid_dip)
        else:
            bottom_progress = (progress - 0.5) / 0.5
            local_dip = mid_dip - bottom_progress * (mid_dip - bottom_dip)

        # 경사에 따른 수평 이동
        delta_depth = depths[i] - depths[i-1]
        if local_dip > 2:
            delta_h = delta_depth / np.tan(np.radians(local_dip))
        else:
            delta_h = delta_depth * 12  # 거의 수평일 때 크게 확장

        # 방향: 외향 (왼쪽 경계는 왼쪽으로, 오른쪽은 오른쪽으로)
        if is_left_boundary:
            new_x = x_coords[-1] - delta_h
        else:
            new_x = x_coords[-1] + delta_h

        # 범위 제한
        new_x = max(0, min(length, new_x))
        new_y = start_e - depth

        x_coords.append(new_x)
        y_coords.append(new_y)

    coords = list(zip(x_coords, y_coords))

    # 강한 불규칙성 추가 (자연스러운 관입체 경계)
    return add_natural_irregularity(coords, amplitude=100, frequency=3)


def generate_laccolith_boundary(start_d, start_e, min_elev, base_dip, is_left_boundary, length, unit_width=None):
    """
    Generate laccolith (병반) boundary.

    Laccolith 단면:
    - 상부: 외향 확장 (돔이 상부 지층을 밀어올린 형태)
    - 하부: 평탄 바닥 (원래 층리면, 기울어진 지층을 따라감)

    경계 이동 = 돔 확장(expansion) + 구조 기울기(tilt)
    - expansion: is_left_boundary에 의해 외향 (돔 형태)
    - tilt: base_dip 부호에 의해 전체 암체 기울기 (주변 지층 경사 반영)

    |base_dip| → 표면 돔 시작 각도 (35°~75°)
    sign(base_dip) → 구조 기울기 방향 (양수=좌, 음수=우)
    """
    total_depth = start_e - min_elev
    num_points = 50

    effective_width = unit_width or 800

    # 표면 돔 시작 각도: |base_dip| 기반 (35°~75° 범위)
    surface_dip = max(35, min(75, abs(base_dip))) if base_dip else 55

    # 구조 기울기 (주변 지층 경사를 따라 전체 암체가 기울어짐)
    tilt_factor = 0.3
    if base_dip and abs(base_dip) > 5:
        tilt_per_depth = tilt_factor / np.tan(np.radians(abs(base_dip)))
        tilt_sign = -1 if base_dip >= 0 else 1  # 양수 dip → 좌로 기울기
    else:
        tilt_per_depth = 0
        tilt_sign = 0

    depths = np.linspace(0, total_depth, num_points)
    x_coords = [start_d]
    y_coords = [start_e]

    for i, depth in enumerate(depths[1:], 1):
        progress = depth / total_depth

        if progress < 0.4:
            # 상부 40%: 외향 확장 (돔 부분) — surface_dip에서 점점 완만해짐
            local_dip = surface_dip - progress * (surface_dip - 35)  # surface_dip → 35°
            local_dip = max(35, local_dip)
        elif progress < 0.6:
            # 중부 20%: 거의 수직 (최대 폭 유지 → 바닥으로 전환)
            local_dip = 85
        else:
            # 하부 40%: 거의 수직 (평탄 바닥 효과 — 수평 이동 최소화)
            local_dip = 88

        delta_depth = depths[i] - depths[i-1]

        # 1. 돔 확장 (외향)
        expansion_h = delta_depth / np.tan(np.radians(local_dip))
        if is_left_boundary:
            expansion_x = -expansion_h
        else:
            expansion_x = +expansion_h

        # 2. 구조 기울기 (양쪽 경계 동일 방향)
        tilt_x = tilt_sign * delta_depth * tilt_per_depth

        # 3. 합산
        new_x = x_coords[-1] + expansion_x + tilt_x

        new_x = max(0, min(length, new_x))
        new_y = start_e - depth
        x_coords.append(new_x)
        y_coords.append(new_y)

    coords = list(zip(x_coords, y_coords))
    return add_natural_irregularity(coords, amplitude=min(40, effective_width * 0.05), frequency=3)


def generate_volcanic_neck_boundary(start_d, start_e, min_elev, base_dip, is_left_boundary, length, unit_width=None):
    """
    Generate volcanic neck (화산암경) boundary — 거의 수직 원통형.
    plug과 동일한 형태.
    """
    return generate_plug_boundary(start_d, start_e, min_elev, base_dip, is_left_boundary, length, unit_width)


def generate_plug_boundary(start_d, start_e, min_elev, base_dip, is_left_boundary, length, unit_width=None):
    """
    Generate volcanic plug boundary (nearly vertical, slight expansion).

    화산암경(Plug) 형태:
    - 거의 수직 (75-85°)
    - 심부에서 미세하게 확장
    - 불규칙한 경계

    base_dip 부호로 구조 기울기 반영 (tilt_factor=0.15, plug은 수직성이 강해 tilt 영향 적음)
    """
    total_depth = start_e - min_elev
    num_points = 40

    effective_width = unit_width or 500

    # 구조 기울기 (plug은 수직성이 강해 tilt 영향 적게)
    tilt_factor = 0.15
    if base_dip and abs(base_dip) > 5:
        tilt_per_depth = tilt_factor / np.tan(np.radians(abs(base_dip)))
        tilt_sign = -1 if base_dip >= 0 else 1
    else:
        tilt_per_depth = 0
        tilt_sign = 0

    depths = np.linspace(0, total_depth, num_points)
    x_coords = [start_d]
    y_coords = [start_e]

    for i, depth in enumerate(depths[1:], 1):
        progress = depth / total_depth

        # 거의 수직, 심부에서 미세 확장
        local_dip = 80 - progress * 8  # 80° → 72°

        delta_depth = depths[i] - depths[i-1]

        # 1. 외향 확장
        expansion_h = delta_depth / np.tan(np.radians(local_dip))
        if is_left_boundary:
            expansion_x = -expansion_h
        else:
            expansion_x = +expansion_h

        # 2. 구조 기울기
        tilt_x = tilt_sign * delta_depth * tilt_per_depth

        # 3. 합산
        new_x = x_coords[-1] + expansion_x + tilt_x

        new_x = max(0, min(length, new_x))
        new_y = start_e - depth
        x_coords.append(new_x)
        y_coords.append(new_y)

    coords = list(zip(x_coords, y_coords))
    return add_natural_irregularity(coords, amplitude=30, frequency=4)


def generate_dike_boundary(start_d, start_e, min_elev, base_dip, is_left_boundary, length, unit_width=None):
    """
    Generate dike boundary (vertical, constant width).

    암맥(Dike) 형태:
    - 완전 수직 또는 거의 수직 (85-90°)
    - 폭 유지 (확장 없음)
    - 약간의 자연 굴곡만

    base_dip 부호로 구조 기울기 반영 (tilt_factor=0.2)
    """
    total_depth = start_e - min_elev
    num_points = 30

    # 구조 기울기
    tilt_factor = 0.2
    if base_dip and abs(base_dip) > 5:
        tilt_per_depth = tilt_factor / np.tan(np.radians(abs(base_dip)))
        tilt_sign = -1 if base_dip >= 0 else 1
    else:
        tilt_per_depth = 0
        tilt_sign = 0

    depths = np.linspace(0, total_depth, num_points)
    x_coords = [start_d]
    y_coords = [start_e]

    for i, depth in enumerate(depths[1:], 1):
        # 거의 수직 (87°) - 약간의 경사만
        local_dip = 87

        delta_depth = depths[i] - depths[i-1]

        # 1. 미세 외향 확장
        expansion_h = delta_depth / np.tan(np.radians(local_dip))
        if is_left_boundary:
            expansion_x = -expansion_h
        else:
            expansion_x = +expansion_h

        # 2. 구조 기울기
        tilt_x = tilt_sign * delta_depth * tilt_per_depth

        # 3. 합산
        new_x = x_coords[-1] + expansion_x + tilt_x

        new_x = max(0, min(length, new_x))
        new_y = start_e - depth
        x_coords.append(new_x)
        y_coords.append(new_y)

    coords = list(zip(x_coords, y_coords))
    return add_natural_irregularity(coords, amplitude=15, frequency=5)


def generate_metamorphic_boundary(start_d, start_e, min_elev, app_dip, length):
    """
    Generate metamorphic rock boundary (conformable, follows foliation).

    변성암은 (수직과장 고려):
    - 엽리 방향 SE 40-50° (수직과장으로 더 가파르게 보임)
    - 실제 경사 30-40°로 설정 (시각적으로 45-55° 보임)
    - 습곡 영향으로 완만한 굴곡
    """
    total_depth = start_e - min_elev
    num_points = 40

    # 변성암 경사: 25-35° (수직과장 고려, 시각적으로 40-50°)
    # 입력값이 너무 가파르면 30°로 제한
    base_angle = min(35, max(25, abs(app_dip) * 0.5))

    depths = np.linspace(0, total_depth, num_points)
    x_coords = [start_d]
    y_coords = [start_e]

    for i, depth in enumerate(depths[1:], 1):
        progress = depth / total_depth

        # 습곡 영향으로 경사 변화 (±10°) - 더 뚜렷한 굴곡
        fold_effect = np.sin(progress * 2 * np.pi) * 10
        local_dip = base_angle + fold_effect

        # 최대 경사 제한 (50° 이상으로 가파르지 않게)
        local_dip = min(50, max(25, local_dip))

        delta_depth = depths[i] - depths[i-1]
        if local_dip > 5:
            delta_h = delta_depth / np.tan(np.radians(local_dip))
        else:
            delta_h = delta_depth * 3

        # 방향 유지
        if app_dip >= 0:
            new_x = x_coords[-1] - delta_h
        else:
            new_x = x_coords[-1] + delta_h

        new_x = max(0, min(length, new_x))
        new_y = start_e - depth

        x_coords.append(new_x)
        y_coords.append(new_y)

    coords = list(zip(x_coords, y_coords))
    return add_natural_irregularity(coords, amplitude=40, frequency=3)


def generate_curved_boundary(start_d, start_e, min_elev, app_dip, depth_behavior, depth_to_flatten, contact_geometry, intrusion_shape, length, contact_type=None, is_left=True, unit_width=None, body_geometry='conformable'):
    """
    Generate curved boundary path based on body geometry and contact relationship.

    Args:
        contact_type: 'intrusive', 'conformable', 'unconformable', etc.
        is_left: True if this is the left boundary of a rock unit
        unit_width: Width of the rock unit at surface (for intrusion expansion calc)
        body_geometry: 'batholith'|'stock'|'dome'|'plug'|'dike'|'sill'|'flow'|'conformable'
    """
    # body_geometry 기반 라우팅
    if body_geometry == 'batholith':
        return generate_intrusion_boundary(start_d, start_e, min_elev, app_dip, is_left, length, unit_width=unit_width, expansion_factor=5)
    elif body_geometry == 'stock':
        return generate_intrusion_boundary(start_d, start_e, min_elev, app_dip, is_left, length, unit_width=unit_width, expansion_factor=3)
    elif body_geometry == 'laccolith':
        return generate_laccolith_boundary(start_d, start_e, min_elev, app_dip, is_left, length, unit_width=unit_width)
    elif body_geometry in ('plug', 'volcanic_neck'):
        return generate_plug_boundary(start_d, start_e, min_elev, app_dip, is_left, length, unit_width=unit_width)
    elif body_geometry == 'dike':
        return generate_dike_boundary(start_d, start_e, min_elev, app_dip, is_left, length, unit_width=unit_width)
    elif body_geometry in ('flow', 'conformable', 'sill'):
        return generate_metamorphic_boundary(start_d, start_e, min_elev, app_dip, length)

    # 기본 처리 (기존 로직 개선)
    total_depth = start_e - min_elev
    num_points = 30

    # 기본 직선 경계 (단순 케이스)
    if contact_geometry == 'planar' and depth_behavior == 'constant':
        if abs(app_dip) > 1:
            h_offset = total_depth / np.tan(np.radians(abs(app_dip)))
        else:
            h_offset = 0

        if app_dip >= 0:
            end_d = max(0, start_d - h_offset)
        else:
            end_d = min(length, start_d + h_offset)

        coords = [(start_d, start_e), (end_d, min_elev)]
        return add_natural_irregularity(coords, amplitude=10, frequency=2)

    # 곡선 경계 생성
    depths = np.linspace(0, total_depth, num_points)
    x_coords = [start_d]
    y_coords = [start_e]

    for i, depth in enumerate(depths[1:], 1):
        progress = depth / total_depth

        if depth_behavior == 'flattening':
            flatten_factor = progress ** 1.2
            min_dip = 25  # 더 현실적인 최소 경사
            local_dip = abs(app_dip) - (abs(app_dip) - min_dip) * flatten_factor
        elif depth_behavior == 'steepening':
            steep_factor = progress ** 0.7
            target_dip = 75  # 최대 75°
            local_dip = abs(app_dip) + (target_dip - abs(app_dip)) * steep_factor
        else:
            local_dip = abs(app_dip)

        delta_depth = depths[i] - depths[i-1]
        if local_dip > 1:
            delta_h = delta_depth / np.tan(np.radians(local_dip))
        else:
            delta_h = 0

        if app_dip >= 0:
            new_x = x_coords[-1] - delta_h
        else:
            new_x = x_coords[-1] + delta_h

        new_x = max(0, min(length, new_x))
        new_y = start_e - depth

        x_coords.append(new_x)
        y_coords.append(new_y)

    coords = list(zip(x_coords, y_coords))
    return add_natural_irregularity(coords, amplitude=20, frequency=4)


def create_section_figure(section, profile_data, projections, output_name):
    """Create cross-section figure with LLM-derived dip and curved boundaries"""

    distances, elevations, coords, length = profile_data

    # 수직-수평 비율 조정 (이미지 크기 제한: 2000px 이하로)
    # figsize=(50, 10) * dpi=100 = 5000x1000 픽셀
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(50, 10),
                                    gridspec_kw={'height_ratios': [6, 1]})

    # 깊이를 1km로 설정 (관입체 형태를 보여주기 위해)
    valid_elevations = elevations[np.isfinite(elevations)] if hasattr(elevations, '__len__') else elevations
    if len(valid_elevations) == 0:
        # 유효한 고도 데이터가 없으면 기본값 사용
        min_elev = -1000
        max_elev = 100
    else:
        min_elev = np.nanmin(valid_elevations) - 1000
        max_elev = np.nanmax(valid_elevations) + 50
    depth = 1050

    # 수직 과장 계산 (X/Y 스케일 비율)
    v_exag = (length / (max_elev - min_elev)) * 0.6

    # 지층 선후관계에 따라 정렬 (오래된 것 먼저 그림 → 젊은 것이 위에 덮음)
    # Qa(충적층)는 별도 처리하므로 제외
    non_qa_projections = [p for p in projections if p['litho_idx'] != 'Qa']
    qa_projections = [p for p in projections if p['litho_idx'] == 'Qa']

    sorted_non_qa = sorted(non_qa_projections,
                           key=lambda p: (LITHO_AGE_ORDER.get(p['litho_idx'], 0),
                                         p['start_dist']))

    # ========== 배경 채우기: 빈 공간이 없도록 ==========
    # 전체 단면을 가장 오래된 암석으로 먼저 채움
    # 회색 "미분화 기반암"은 검토자가 비과학적이라고 비판하므로
    # 기저 암석으로 채워서 빈 공간이 발생하지 않도록 함
    oldest_code = min(LITHO_AGE_ORDER.keys(), key=lambda k: LITHO_AGE_ORDER[k]) if LITHO_AGE_ORDER else None
    base_rock_color = LITHO_COLORS.get(oldest_code, '#FFB6C1') if oldest_code else '#FFB6C1'
    ax1.fill([0, length, length, 0], [max_elev, max_elev, min_elev, min_elev],
             color=base_rock_color, alpha=0.35, zorder=-1)

    # ========== 1차 패스: Qa 이외의 암석 먼저 그리기 ==========
    plotted = set()
    for draw_idx, proj in enumerate(sorted_non_qa):
        color = LITHO_COLORS.get(proj['litho_idx'], '#CCCCCC')
        litho_code = proj['litho_idx']
        litho_name = LITHO_NAMES.get(litho_code, litho_code)
        age_order = LITHO_AGE_ORDER.get(litho_code, 0)

        label = f"{litho_code} ({litho_name})" if litho_code not in plotted else None
        poly_zorder = age_order * 100 + draw_idx
        alpha = 1.0

        start_d = proj['start_dist']
        end_d = proj['end_dist']
        start_e = proj['start_elev']
        end_e = proj['end_elev']
        app_dip = proj['apparent_dip']

        contact_geometry = proj.get('contact_geometry', 'planar')
        depth_behavior = proj.get('depth_behavior', 'constant')
        depth_to_flatten = proj.get('depth_to_flatten')
        intrusion_shape = proj.get('intrusion_shape')
        contact_type = proj.get('contact_type', 'unknown')

        # 암체의 지표 폭 (관입체 확장 계산에 사용)
        unit_width = end_d - start_d

        # 암체 형태 판정 (LLM → KB → 휴리스틱 → conformable)
        body_geo = _get_body_geometry(litho_code, intrusion_shape, unit_width)
        is_non_conformable = body_geo not in ('flow', 'conformable', 'sill')

        # 곡선 경계 생성 - body_geometry에 따라 형태별 분기
        left_curve = generate_curved_boundary(
            start_d, start_e, min_elev, app_dip,
            depth_behavior, depth_to_flatten, contact_geometry, intrusion_shape, length,
            contact_type=contact_type,
            is_left=True,
            unit_width=unit_width,
            body_geometry=body_geo
        )
        right_curve = generate_curved_boundary(
            end_d, end_e, min_elev, app_dip,
            depth_behavior, depth_to_flatten, contact_geometry, intrusion_shape, length,
            contact_type=contact_type,
            is_left=False,
            unit_width=unit_width,
            body_geometry=body_geo
        )

        poly_x = [p[0] for p in left_curve] + [p[0] for p in reversed(right_curve)]
        poly_y = [p[1] for p in left_curve] + [p[1] for p in reversed(right_curve)]

        # 암체 폴리곤 그리기 - body_geometry별 테두리 색상/스타일
        if body_geo in ('batholith', 'stock'):
            edge_color = CONTACT_COLORS.get('intrusive', '#FF4444')
            edge_width = 2.5
        elif body_geo in ('laccolith', 'plug', 'volcanic_neck'):
            edge_color = '#FF8C00'  # 주황색 (subvolcanic)
            edge_width = 2.0
        elif body_geo == 'dike':
            edge_color = CONTACT_COLORS.get('intrusive', '#FF4444')
            edge_width = 1.5
        else:
            edge_color = 'black'
            edge_width = 0.8

        # dike는 점선 테두리
        if body_geo == 'dike':
            ax1.fill(poly_x, poly_y, color=color, alpha=alpha,
                    edgecolor=edge_color, linewidth=edge_width, linestyle='--',
                    label=label, zorder=poly_zorder)
        else:
            ax1.fill(poly_x, poly_y, color=color, alpha=alpha,
                    edgecolor=edge_color, linewidth=edge_width,
                    label=label, zorder=poly_zorder)

        # 암체 내부에 암상 코드 라벨 추가 (폭이 충분한 경우)
        body_width = end_d - start_d
        if body_width > length * 0.02:
            label_x = (start_d + end_d) / 2
            label_y = (start_e + end_e) / 2 - depth * 0.15
            ax1.text(label_x, label_y, litho_code,
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white', zorder=poly_zorder + 50,
                    bbox=dict(facecolor=color, edgecolor='black', alpha=0.8,
                             boxstyle='round,pad=0.3'))

        # 경계선 그리기 (정합암만) - 상부 500m만 표시
        # 비정합 암체는 폴리곤 테두리로 이미 표시됨
        contact_color = CONTACT_COLORS.get(proj['contact_type'], 'gray')
        max_boundary_depth = 500
        if not is_non_conformable and proj['start_dist'] > 0:
            curve_x = [p[0] for p in left_curve]
            curve_y = [p[1] for p in left_curve]
            # 상부 500m까지만 경계선 표시 (최소 표고 기준)
            boundary_min_elev = start_e - max_boundary_depth
            clipped_x = [x for x, y in zip(curve_x, curve_y) if y >= boundary_min_elev]
            clipped_y = [y for y in curve_y if y >= boundary_min_elev]
            if len(clipped_x) > 1:
                if contact_geometry == 'curved' or depth_behavior != 'constant':
                    ax1.plot(clipped_x, clipped_y, color=contact_color, linewidth=2.5,
                            linestyle='-', alpha=0.9, zorder=1000)
                else:
                    ax1.plot(clipped_x, clipped_y, color=contact_color, linewidth=2,
                            linestyle='--', alpha=0.8, zorder=1000)

        plotted.add(proj['litho_idx'])

    # ========== 2차 패스: Qa(충적층) 맨 마지막에 그리기 ==========
    # Qa는 저지대/계곡에만 퇴적되는 표층 퇴적물
    # 두께 10-25m (과장되지 않게), 경계에서 0으로 수렴 (렌즈형)
    for proj in qa_projections:
        color = LITHO_COLORS.get('Qa', '#FFFACD')
        litho_name = LITHO_NAMES.get('Qa', '충적층')
        label = f"Qa ({litho_name})" if 'Qa' not in plotted else None

        start_d = proj['start_dist']
        end_d = proj['end_dist']
        qa_width = end_d - start_d

        # 충적층: 지표면 따라가되, 경계에서 두께가 0으로 수렴
        num_surface_pts = 50  # 더 세밀하게
        surface_x = np.linspace(start_d, end_d, num_surface_pts)
        surface_indices = [np.searchsorted(distances, x) for x in surface_x]
        surface_indices = [min(i, len(elevations)-1) for i in surface_indices]
        surface_y = [elevations[i] for i in surface_indices]

        # 하부 경계: 매우 얇은 렌즈형 - 최대 두께 20m로 제한 (현실적)
        max_thickness = 20  # 최대 두께 20m (현실적 범위)
        bottom_y = []
        for i, (sx, sy) in enumerate(zip(surface_x, surface_y)):
            # 경계에서의 상대 위치 (0~1~0: 양 끝이 0, 중앙이 1)
            relative_pos = (sx - start_d) / qa_width  # 0 to 1
            # 포물선 형태: 중앙에서 최대, 양끝에서 0
            thickness_factor = 4 * relative_pos * (1 - relative_pos)  # 0→1→0
            # 약간의 비대칭 추가 (자연스러움)
            thickness_factor *= (0.85 + 0.3 * np.sin(relative_pos * np.pi))
            local_thickness = max_thickness * thickness_factor
            bottom_y.append(sy - local_thickness)

        # 폴리곤 생성 - 상부는 지표면, 하부는 렌즈형
        poly_x = list(surface_x) + list(reversed(surface_x))
        poly_y = list(surface_y) + list(reversed(bottom_y))

        # Qa는 zorder=9999로 가장 위에 (지형선 제외)
        # 두꺼운 테두리로 명확히 구분
        ax1.fill(poly_x, poly_y, color=color, alpha=1.0,
                edgecolor='#B8860B', linewidth=2.5, label=label, zorder=9999)

        # 추가: 하부 경계에 점선 추가 (부정합면 표시)
        ax1.plot(surface_x, bottom_y, color='#B8860B', linewidth=1.5,
                linestyle='--', alpha=0.8, zorder=9999)

        plotted.add('Qa')

    # Terrain (zorder=2000으로 가장 위에)
    ax1.fill_between(distances, elevations, max_elev + 50, color='lightblue', alpha=0.3, zorder=1500)
    ax1.plot(distances, elevations, 'k-', linewidth=3, zorder=2000)

    # Dip indicators with geometry info
    interval = length / 6
    for i in range(1, 6):
        d = i * interval
        idx = np.searchsorted(distances, d)
        if idx >= len(elevations):
            continue
        e = elevations[idx]

        # Find projection at this distance
        for proj in projections:
            if proj['start_dist'] <= d <= proj['end_dist']:
                app_dip = proj['apparent_dip']
                conf = proj['confidence']
                depth_behavior = proj.get('depth_behavior', 'constant')
                contact_geometry = proj.get('contact_geometry', 'planar')

                # Draw dip symbol - 경계선과 같은 방향으로
                sym_len = 400
                ax1.plot([d - sym_len/2, d + sym_len/2], [e + 30, e + 30], 'b-', lw=2)

                if abs(app_dip) > 3:
                    tick_len = 300
                    # 경계선 방향과 일치시킴
                    if app_dip >= 0:
                        tick_dx = -tick_len
                    else:
                        tick_dx = tick_len
                    tick_dy = -abs(tick_len * np.tan(np.radians(app_dip)))
                    ax1.plot([d, d + tick_dx], [e + 30, e + 30 + tick_dy], 'b-', lw=2)

                # Confidence color
                conf_color = {'high': 'green', 'medium': 'blue', 'low': 'orange'}.get(conf, 'red')

                # Add geometry indicator symbol
                geo_symbol = ''
                if depth_behavior == 'flattening':
                    geo_symbol = ' ↘'  # Flattening with depth
                elif depth_behavior == 'steepening':
                    geo_symbol = ' ↗'  # Steepening with depth
                if contact_geometry == 'curved':
                    geo_symbol += '⌒'  # Curved contact

                ax1.annotate(f'{abs(app_dip):.0f}°{geo_symbol}', (d, e + 80),
                           ha='center', fontsize=18, color=conf_color, fontweight='bold')
                break

    ax1.set_xlim(0, length)
    ax1.set_ylim(min_elev, max_elev)
    ax1.set_xlabel('Distance (m)', fontsize=20)
    ax1.set_ylabel('Elevation (m)', fontsize=20)
    ax1.set_title(f"Cross-Section: {section['name']} (AI Geoscientist)\n"
                  f"단면: {section['name']} (AI Geoscientist 경사 적용)",
                  fontsize=24, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=16)

    # Endpoints
    name_parts = section['name'].replace("'", "").split('-')
    ax1.text(0, max_elev - 20, name_parts[0] if name_parts else 'A',
            ha='center', fontsize=28, fontweight='bold')
    ax1.text(length, max_elev - 20, name_parts[-1] if len(name_parts) > 1 else "A'",
            ha='center', fontsize=28, fontweight='bold')

    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    # Add contact type legend
    for ct, color in CONTACT_COLORS.items():
        if ct != 'fault':
            handles.append(mpatches.Patch(facecolor='white', edgecolor=color,
                                         linewidth=2, linestyle='--'))
            labels.append(f'{ct} contact')

    # Add curved boundary indicator
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='-'))
    labels.append('curved boundary')
    handles.append(Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--'))
    labels.append('planar boundary')

    ax1.legend(handles[:14], labels[:14], loc='upper right', fontsize=14, ncol=2)

    # Bottom: Geology strip with confidence
    ax2.set_xlim(0, length)
    ax2.set_ylim(0, 1)

    for proj in projections:
        if proj['width'] < 10:
            continue

        color = LITHO_COLORS.get(proj['litho_idx'], '#CCCCCC')

        # Confidence-based alpha
        alpha = {'high': 0.9, 'medium': 0.7, 'low': 0.5, 'very_low': 0.3}.get(proj['confidence'], 0.5)

        rect = plt.Rectangle((proj['start_dist'], 0), proj['width'], 0.7,
                             facecolor=color, edgecolor='black', alpha=alpha)
        ax2.add_patch(rect)

        # Confidence bar
        conf_color = {'high': 'green', 'medium': 'blue', 'low': 'orange', 'very_low': 'red'}.get(proj['confidence'], 'gray')
        conf_rect = plt.Rectangle((proj['start_dist'], 0.75), proj['width'], 0.2,
                                  facecolor=conf_color, alpha=0.7)
        ax2.add_patch(conf_rect)

        # Label
        if proj['width'] > length * 0.03:
            mid = (proj['start_dist'] + proj['end_dist']) / 2
            ax2.text(mid, 0.35, f"{proj['litho_idx']}\n{proj['apparent_dip']:.0f}°",
                    ha='center', va='center', fontsize=14, rotation=0)

    ax2.set_xlabel('Distance (m)', fontsize=18)
    ax2.set_yticks([0.35, 0.85])
    ax2.set_yticklabels(['Lithology', 'Confidence'], fontsize=14)
    ax2.set_title('Surface Geology with AI Confidence', fontsize=18)
    ax2.tick_params(axis='x', labelsize=14)

    # Confidence legend
    for i, (conf, color) in enumerate([('high', 'green'), ('medium', 'blue'),
                                        ('low', 'orange'), ('very_low', 'red')]):
        ax2.add_patch(plt.Rectangle((length * 0.85 + i * length * 0.035, 0.85),
                                    length * 0.03, 0.1, facecolor=color, alpha=0.7))
        ax2.text(length * 0.85 + i * length * 0.035 + length * 0.015, 0.78,
                conf[0].upper(), ha='center', fontsize=12)

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{output_name}_llm_section.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')

    plt.close()
    return str(output_path)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def create_summary_plots(data):
    """Create summary visualization plots for HTML report"""

    llm_results = data['llm_results']['boundaries']

    plots = {}

    # 1. Dip angle distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    dips = [b['dip_angle'] for b in llm_results]
    bar_colors = [CONTACT_COLORS.get(b.get('contact_type', 'unknown'), 'gray') for b in llm_results]

    ax.bar(range(len(dips)), dips, color=bar_colors, edgecolor='black', alpha=0.7)
    ax.axhline(y=np.mean(dips), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Boundary Index', fontsize=11)
    ax.set_ylabel('Dip Angle (°)', fontsize=11)
    ax.set_title('AI 추정 경사각 분포 (Estimated Dip Angles)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Legend for contact types - 명시적으로 색상 지정
    legend_handles = [
        mpatches.Patch(facecolor='#FF4444', edgecolor='black', label='intrusive (관입)', alpha=0.7),
        mpatches.Patch(facecolor='#4444FF', edgecolor='black', label='unconformable (부정합)', alpha=0.7),
        mpatches.Patch(facecolor='#44AA44', edgecolor='black', label='conformable (정합)', alpha=0.7),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dips):.1f}°')
    ]
    ax.legend(handles=legend_handles, loc='upper right')

    plots['dip_distribution'] = fig_to_base64(fig)

    # 2. Contact type pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    contact_types = [b.get('contact_type', 'unknown') for b in llm_results]
    ct_counts = {}
    for ct in contact_types:
        ct_counts[ct] = ct_counts.get(ct, 0) + 1

    colors = [CONTACT_COLORS.get(ct, 'gray') for ct in ct_counts.keys()]
    ax.pie(ct_counts.values(), labels=ct_counts.keys(), colors=colors,
           autopct='%1.0f%%', startangle=90)
    ax.set_title('접촉 유형 분포 (Contact Type Distribution)', fontsize=12)

    plots['contact_types'] = fig_to_base64(fig)

    # 3. Confidence distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    confidences = [b['confidence'] for b in llm_results]
    conf_counts = {}
    conf_order = ['high', 'medium', 'low', 'very_low']
    for c in conf_order:
        conf_counts[c] = confidences.count(c)

    colors = ['green', 'blue', 'orange', 'red']
    bars = ax.bar(conf_counts.keys(), conf_counts.values(), color=colors, edgecolor='black')
    ax.set_xlabel('Confidence Level', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('신뢰도 분포 (Confidence Distribution)', fontsize=12)

    for bar, count in zip(bars, conf_counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               str(count), ha='center', fontsize=11, fontweight='bold')

    plots['confidence'] = fig_to_base64(fig)

    # 4. Map view with analyzed boundaries (larger size)
    fig, ax = plt.subplots(figsize=(20, 18))

    # Plot lithology
    for idx, row in data['litho'].iterrows():
        litho_idx = row.get('LITHOIDX', 'Unknown')
        color = LITHO_COLORS.get(litho_idx, '#CCCCCC')
        geom = row.geometry
        if geom.geom_type == 'Polygon':
            ax.fill(*geom.exterior.xy, color=color, alpha=0.5, edgecolor='gray', linewidth=0.2)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                ax.fill(*poly.exterior.xy, color=color, alpha=0.5, edgecolor='gray', linewidth=0.2)

    # Highlight analyzed boundaries with labels
    for i, b in enumerate(llm_results):
        idx = b['boundary_idx']
        if idx < len(data['boundary']):
            geom = data['boundary'].iloc[idx].geometry
            if geom is not None:
                contact_type = b.get('contact_type', 'unknown')
                color = CONTACT_COLORS.get(contact_type, 'gray')
                lithos = b.get('adjacent_lithologies', [])
                litho_str = '-'.join(lithos[:2]) if lithos else ''

                if geom.geom_type == 'LineString':
                    ax.plot(*geom.xy, color=color, linewidth=3, alpha=0.8)
                    centroid = geom.centroid
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        ax.plot(*line.xy, color=color, linewidth=3, alpha=0.8)
                    centroid = geom.geoms[0].centroid
                else:
                    continue

                # Add number label
                ax.annotate(f"{i+1}", (centroid.x, centroid.y),
                           fontsize=8, fontweight='bold', color='white',
                           bbox=dict(boxstyle='circle', facecolor=color, edgecolor='black', alpha=0.9),
                           ha='center', va='center')
                # Add litho labels at boundary
                if litho_str:
                    ax.annotate(litho_str, (centroid.x, centroid.y),
                               fontsize=7, ha='center', va='top',
                               xytext=(0, -15), textcoords='offset points',
                               color='black', fontweight='bold',
                               bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray',
                                        boxstyle='round,pad=0.2'))

    # Plot sections - 두꺼운 흰색 배경선 + 검정 전경선으로 가시성 확보
    for section in data['sections']:
        sx, sy = section['start']['x'], section['start']['y']
        ex, ey = section['end']['x'], section['end']['y']
        # 흰색 배경선 (두꺼운)
        ax.plot([sx, ex], [sy, ey], color='white', linewidth=5, alpha=0.9, zorder=10)
        # 검정 전경선
        ax.plot([sx, ex], [sy, ey], color='black', linewidth=2.5, alpha=0.9, zorder=11)
        # 시종점 마커
        ax.plot(sx, sy, 'o', color='white', markersize=8, markeredgecolor='black',
                markeredgewidth=2, zorder=12)
        ax.plot(ex, ey, 's', color='white', markersize=8, markeredgecolor='black',
                markeredgewidth=2, zorder=12)
        # 단면 이름 라벨
        mid_x = (sx + ex) / 2
        mid_y = (sy + ey) / 2
        ax.annotate(section['name'], (mid_x, mid_y), fontsize=10, fontweight='bold',
                   color='black', zorder=13,
                   bbox=dict(facecolor='yellow', alpha=0.9, edgecolor='black',
                            boxstyle='round,pad=0.3'))

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('AI 분석 경계 위치 (Analyzed Boundary Locations)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Contact type legend (upper left)
    contact_handles = []
    for ct, ct_color in CONTACT_COLORS.items():
        if ct != 'fault':
            contact_handles.append(mpatches.Patch(facecolor=ct_color, alpha=0.8, label=ct))
    contact_legend = ax.legend(handles=contact_handles, loc='upper left', fontsize=8,
                               title='접촉 유형 (Contact Type)', title_fontsize=9)
    ax.add_artist(contact_legend)

    # Litho legend (lower right) - all unique rock types with colors
    unique_lithos = sorted(data['litho']['LITHOIDX'].dropna().unique())
    litho_patches = []
    for code in unique_lithos:
        lc = LITHO_COLORS.get(code, '#CCCCCC')
        ln = LITHO_NAMES.get(code, code)
        litho_patches.append(mpatches.Patch(facecolor=lc, edgecolor='gray',
                                            alpha=0.7, label=f"{code} ({ln})"))
    ncol = max(1, (len(litho_patches) + 5) // 6)
    ax.legend(handles=litho_patches, loc='lower right', fontsize=7, ncol=ncol,
              title='암상 (Lithology)', title_fontsize=9)

    # Save map as separate file
    fig.savefig(OUTPUT_DIR / 'boundary_location_map.png', dpi=150, bbox_inches='tight', facecolor='white')

    plots['map_view'] = fig_to_base64(fig)

    return plots


def _generate_geological_overview_html():
    """지식베이스의 지역 지질 개요를 HTML로 변환"""
    import sys
    import html as html_mod

    # geologist_agent_llm에서 GEOLOGICAL_CONTEXT 가져오기
    geo_mod = sys.modules.get('geologist_agent_llm')
    context = getattr(geo_mod, 'GEOLOGICAL_CONTEXT', '') if geo_mod else ''

    if not context or context.strip().startswith('# 지질 개요'):
        # 기본 컨텍스트(짧은 것)만 있으면 생략
        return '<p style="color: #999;">지역 지식베이스가 로드되지 않았습니다.</p>'

    # Markdown → 간단 HTML 변환
    lines = context.strip().split('\n')
    html_parts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('# '):
            html_parts.append(f'<h3>{html_mod.escape(line[2:])}</h3>')
        elif line.startswith('## '):
            html_parts.append(f'<h4 style="margin-top:15px; color:#2c3e50;">{html_mod.escape(line[3:])}</h4>')
        elif line.startswith('### '):
            html_parts.append(f'<h5 style="color:#34495e;">{html_mod.escape(line[4:])}</h5>')
        elif line.startswith('- **'):
            # Bold list item: - **KEY** text
            parts = line[2:].split('**')
            if len(parts) >= 3:
                bold = html_mod.escape(parts[1])
                rest = html_mod.escape('**'.join(parts[2:]))
                html_parts.append(f'<li><strong>{bold}</strong>{rest}</li>')
            else:
                html_parts.append(f'<li>{html_mod.escape(line[2:])}</li>')
        elif line.startswith('- '):
            html_parts.append(f'<li>{html_mod.escape(line[2:])}</li>')
        else:
            html_parts.append(f'<p>{html_mod.escape(line)}</p>')

    return '\n'.join(html_parts)


def _generate_litho_legend_html(data):
    """현재 데이터의 LITHOIDX 코드에 해당하는 범례 HTML 동적 생성"""
    import html as html_mod
    legend_items = []
    # 현재 데이터에 실제 존재하는 암상 코드만 표시
    if 'litho' in data and 'LITHOIDX' in data['litho'].columns:
        unique_codes = data['litho']['LITHOIDX'].dropna().unique()
    else:
        unique_codes = list(LITHO_NAMES.keys())

    for code in sorted(unique_codes, key=lambda c: LITHO_AGE_ORDER.get(c, 99)):
        color = LITHO_COLORS.get(code, '#CCCCCC')
        name = html_mod.escape(LITHO_NAMES.get(code, code))
        legend_items.append(
            f'<div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">'
            f'<span style="display: inline-block; width: 30px; height: 20px; background: {color}; border: 1px solid #999; margin-right: 10px;"></span>'
            f'<strong>{html_mod.escape(code)}</strong>&nbsp;- {name}'
            f'</div>'
        )
    return '\n                '.join(legend_items)


def generate_html_report(data, section_images, plots):
    """Generate comprehensive HTML report"""

    llm_results = data['llm_results']
    boundaries = llm_results['boundaries']

    # Statistics
    dips = [b['dip_angle'] for b in boundaries]
    confidences = [b['confidence'] for b in boundaries]
    contact_types = [b.get('contact_type', 'unknown') for b in boundaries]

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Geologist Agent Report - {REGION_NAME_KR} 지질 분석</title>
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #3498db;
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
            --light: #ecf0f1;
            --dark: #2c3e50;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Malgun Gothic', 'Segoe UI', sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background: var(--light);
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 10px;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .card h3 {{
            color: var(--secondary);
            font-size: 2.5em;
            margin-bottom: 5px;
        }}

        .card p {{
            color: #666;
            font-size: 0.95em;
        }}

        .section {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .section h2 {{
            color: var(--primary);
            border-bottom: 3px solid var(--secondary);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}

        .chart-container {{
            text-align: center;
        }}

        .chart-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .section-image {{
            margin: 20px 0;
            text-align: center;
        }}

        .section-image img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        th {{
            background: var(--primary);
            color: white;
        }}

        tr:hover {{
            background: #f5f5f5;
        }}

        .confidence-high {{ color: var(--success); font-weight: bold; }}
        .confidence-medium {{ color: var(--secondary); font-weight: bold; }}
        .confidence-low {{ color: var(--warning); font-weight: bold; }}
        .confidence-very_low {{ color: var(--danger); font-weight: bold; }}

        .contact-intrusive {{ color: #e74c3c; }}
        .contact-unconformable {{ color: #3498db; }}
        .contact-conformable {{ color: #27ae60; }}

        .reasoning {{
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid var(--secondary);
            margin: 10px 0;
            font-size: 0.95em;
            border-radius: 0 8px 8px 0;
        }}

        .boundary-detail {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }}

        .boundary-detail h4 {{
            color: var(--primary);
            margin-bottom: 10px;
        }}

        .tag {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            margin: 2px;
        }}

        .tag-intrusive {{ background: #ffebee; color: #c62828; }}
        .tag-unconformable {{ background: #e3f2fd; color: #1565c0; }}
        .tag-conformable {{ background: #e8f5e9; color: #2e7d32; }}

        footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            header h1 {{ font-size: 1.8em; }}
            .chart-grid {{ grid-template-columns: 1fr; }}
        }}

        /* 이미지 팝업 모달 스타일 */
        .modal {{
            display: none;
            position: fixed;
            z-index: 9999;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            overflow: auto;
        }}

        .modal-content {{
            display: block;
            margin: 20px auto;
            max-width: 95%;
            max-height: 90vh;
            object-fit: contain;
        }}

        .modal-close {{
            position: fixed;
            top: 20px;
            right: 40px;
            color: #fff;
            font-size: 50px;
            font-weight: bold;
            cursor: pointer;
            z-index: 10000;
        }}

        .modal-close:hover {{
            color: #ccc;
        }}

        .modal-caption {{
            text-align: center;
            color: #fff;
            padding: 10px;
            font-size: 1.2em;
        }}

        .clickable-img {{
            cursor: zoom-in;
            transition: transform 0.2s;
        }}

        .clickable-img:hover {{
            transform: scale(1.02);
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }}

        .zoom-hint {{
            font-size: 0.85em;
            color: #888;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <!-- 이미지 팝업 모달 -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImg">
        <div class="modal-caption" id="modalCaption"></div>
    </div>

    <script>
        function openModal(img) {{
            var modal = document.getElementById('imageModal');
            var modalImg = document.getElementById('modalImg');
            var modalCaption = document.getElementById('modalCaption');
            modal.style.display = 'block';
            modalImg.src = img.src;
            modalCaption.textContent = img.alt || '';
            document.body.style.overflow = 'hidden';
        }}

        function closeModal() {{
            document.getElementById('imageModal').style.display = 'none';
            document.body.style.overflow = 'auto';
        }}

        // ESC 키로 모달 닫기
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                closeModal();
            }}
        }});
    </script>
    <div class="container">
        <header>
            <h1>🌍 AI Geologist Agent Report</h1>
            <p>{REGION_NAME_KR} 지역 지질 경계 지하구조 추정 보고서</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </header>

        <div class="summary-cards">
            <div class="card">
                <h3>{len(boundaries)}</h3>
                <p>분석된 경계<br>Analyzed Boundaries</p>
            </div>
            <div class="card">
                <h3>{np.mean(dips):.1f}°</h3>
                <p>평균 경사각<br>Mean Dip Angle</p>
            </div>
            <div class="card">
                <h3>{confidences.count('high') + confidences.count('medium')}</h3>
                <p>High/Medium 신뢰도<br>Reliable Estimates</p>
            </div>
            <div class="card">
                <h3>{contact_types.count('intrusive')}</h3>
                <p>관입 접촉<br>Intrusive Contacts</p>
            </div>
        </div>

        <div class="section">
            <h2>🏔️ {REGION_NAME_KR} 지역 지질 개요 (Regional Geological Context)</h2>
            {_generate_geological_overview_html()}
        </div>

        <div class="section">
            <h2>📊 분석 요약 (Analysis Summary)</h2>
            <div class="chart-grid">
                <div class="chart-container">
                    <h4>경사각 분포</h4>
                    <img src="data:image/png;base64,{plots['dip_distribution']}" alt="Dip Distribution">
                </div>
                <div class="chart-container">
                    <h4>접촉 유형</h4>
                    <img src="data:image/png;base64,{plots['contact_types']}" alt="Contact Types">
                </div>
            </div>
            <div class="chart-grid" style="margin-top: 20px;">
                <div class="chart-container">
                    <h4>신뢰도 분포</h4>
                    <img src="data:image/png;base64,{plots['confidence']}" alt="Confidence">
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🪨 암석 단위 범례 (Lithology Legend)</h2>
            <p style="margin-bottom: 15px; color: #666;">단면도에 표시된 암석 코드와 한글 이름입니다.</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px;">
                {_generate_litho_legend_html(data)}
            </div>
        </div>

        <div class="section">
            <h2>🗺️ 단면도 (Cross-Sections with AI Geoscientist)</h2>
            <p class="zoom-hint">💡 이미지를 클릭하면 확대됩니다 (Click image to zoom)</p>
"""

    # Add section images with click-to-zoom
    for name, img_path in section_images.items():
        with open(img_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        img_src = f"data:image/png;base64,{img_data}"

        html += f"""
            <div class="section-image">
                <h4>{name}</h4>
                <img src="{img_src}" alt="{name}" class="clickable-img"
                     onclick="openModal(this)"
                     title="클릭하여 확대">
            </div>
"""

    html += f"""
        </div>

        <div class="section">
            <h2>🗺️ 분석 경계 위치도 (Boundary Location Map)</h2>
            <p style="margin-bottom: 15px; color: #666;">
                아래 지도에서 번호는 상세 분석 테이블의 "지도 #" 열과 일치합니다.
                색상은 접촉 유형을 나타냅니다:
                <span style="color: #FF4444; font-weight: bold;">■ 관입(intrusive)</span>,
                <span style="color: #4444FF; font-weight: bold;">■ 부정합(unconformable)</span>,
                <span style="color: #44AA44; font-weight: bold;">■ 정합(conformable)</span>
                <br><span class="zoom-hint">💡 이미지를 클릭하면 확대됩니다</span>
            </p>
            <div style="text-align: center;">
                <img src="data:image/png;base64,{plots['map_view']}" alt="분석 경계 위치도"
                     class="clickable-img"
                     style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px;"
                     onclick="openModal(this)"
                     title="클릭하여 확대">
            </div>
        </div>

        <div class="section">
            <h2>📋 상세 분석 결과 (Detailed Analysis Results)</h2>
            <table>
                <thead>
                    <tr>
                        <th>지도 #</th>
                        <th>암석 유형</th>
                        <th>주향</th>
                        <th>경사</th>
                        <th>접촉 유형</th>
                        <th>신뢰도</th>
                    </tr>
                </thead>
                <tbody>
"""

    for i, b in enumerate(boundaries):
        lithos = ' - '.join(b.get('adjacent_lithologies', []))
        contact_class = f"contact-{b.get('contact_type', 'unknown')}"
        conf_class = f"confidence-{b['confidence']}"
        contact_color = CONTACT_COLORS.get(b.get('contact_type', 'unknown'), 'gray')

        html += f"""
                    <tr>
                        <td><span style="display:inline-block; width:24px; height:24px; border-radius:50%; background:{contact_color}; color:white; text-align:center; line-height:24px; font-weight:bold;">{i+1}</span></td>
                        <td>{lithos}</td>
                        <td>{b['strike']:.0f}°</td>
                        <td>{b['dip_angle']:.0f}° → {b['dip_direction']:.0f}°</td>
                        <td class="{contact_class}">{b.get('contact_type', 'unknown')}</td>
                        <td class="{conf_class}">{b['confidence']}</td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>💡 AI Geoscientist 상세 (Detailed AI Reasoning)</h2>
"""

    for i, b in enumerate(boundaries[:10]):  # Show first 10
        lithos = ' - '.join(b.get('adjacent_lithologies', []))
        tag_class = f"tag-{b.get('contact_type', 'unknown')}"
        conf_class = f"confidence-{b['confidence']}"
        contact_color = CONTACT_COLORS.get(b.get('contact_type', 'unknown'), 'gray')

        html += f"""
            <div class="boundary-detail">
                <h4><span style="display:inline-block; width:28px; height:28px; border-radius:50%; background:{contact_color}; color:white; text-align:center; line-height:28px; font-weight:bold; margin-right:8px;">{i+1}</span> {lithos}</h4>
                <p>
                    <span class="tag {tag_class}">{b.get('contact_type', 'unknown')}</span>
                    <span class="tag" style="background: #f5f5f5;">길이: {b.get('boundary_length', 0):.0f}m</span>
                    <span class="tag" style="background: #f5f5f5;">표고차: {b.get('elevation_range', 0):.0f}m</span>
                </p>
                <p style="margin-top: 10px;">
                    <strong>추정:</strong> 주향 {b['strike']:.0f}°, 경사 {b['dip_angle']:.0f}° → {b['dip_direction']:.0f}°
                    <span class="{conf_class}">({b['confidence']})</span>
                </p>
                <div class="reasoning">
                    <strong>AI Geoscientist:</strong><br>
                    {b.get('reasoning', 'No reasoning provided')}
                </div>
            </div>
"""

    if len(boundaries) > 10:
        html += f"""
            <p style="text-align: center; color: #666; margin-top: 20px;">
                ... 외 {len(boundaries) - 10}개 경계 (전체 결과는 JSON 파일 참조)
            </p>
"""

    html += f"""
        </div>

        <div class="section">
            <h2>📝 방법론 (Methodology)</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                <div>
                    <h4>🔬 데이터 수집</h4>
                    <ul>
                        <li>지질경계선 좌표 및 길이</li>
                        <li>DEM에서 경계를 따른 표고 프로파일</li>
                        <li>인접 암석 유형 확인</li>
                        <li>주변 엽리 측정값 수집</li>
                    </ul>
                </div>
                <div>
                    <h4>🤖 AI Geoscientist</h4>
                    <ul>
                        <li>3점법 (Three-Point Method)</li>
                        <li>V-자 규칙 (V-Rule)</li>
                        <li>암석 유형별 전형적 구조</li>
                        <li>지역 구조 트렌드 (NE-SW)</li>
                    </ul>
                </div>
                <div>
                    <h4>✅ 신뢰도 평가</h4>
                    <ul>
                        <li><span class="confidence-high">High</span>: 명확한 증거, 일관된 패턴</li>
                        <li><span class="confidence-medium">Medium</span>: 부분적 증거</li>
                        <li><span class="confidence-low">Low</span>: 제한적 데이터</li>
                        <li><span class="confidence-very_low">Very Low</span>: 추정치</li>
                    </ul>
                </div>
            </div>
        </div>

        <footer>
            <p>Generated by AI Geologist Agent | Claude API (claude-sonnet-4-20250514)</p>
            <p>{REGION_NAME_KR} 지역 1:50,000 수치지질도 기반</p>
        </footer>
    </div>
</body>
</html>
"""

    return html


def main():
    print("\n" + "=" * 60)
    print("  Apply LLM Results & Generate HTML Report")
    print("  AI 결과 적용 및 HTML 보고서 생성")
    print("=" * 60)

    # Load data
    data = load_data()

    # 이전 실행의 단면도 PNG 삭제 (필터링된 단면이 남아있는 것 방지)
    import glob as glob_mod
    for old_png in glob_mod.glob(str(OUTPUT_DIR / '*_llm_section.png')):
        os.remove(old_png)
        print(f"  Cleaned up: {Path(old_png).name}")

    # Process sections
    print("\nProcessing cross-sections...")
    section_images = {}

    for section in data['sections']:
        name = section['name']
        print(f"  {name}...", end=" ")

        profile_data = extract_profile(section)
        projections = project_geology_with_llm_dip(section, data, profile_data)

        safe_name = name.replace("'", "").replace("-", "_").replace(" ", "_")
        img_path = create_section_figure(section, profile_data, projections, safe_name)
        section_images[name] = img_path

        print("Done")

    # Create summary plots
    print("\nCreating summary plots...")
    plots = create_summary_plots(data)

    # Generate HTML report
    print("\nGenerating HTML report...")
    html_content = generate_html_report(data, section_images, plots)

    html_path = OUTPUT_DIR / "llm_geologist_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n  HTML Report: {html_path}")

    print("\n" + "=" * 60)
    print("  COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - HTML Report: {html_path}")
    for name, path in section_images.items():
        print(f"  - Section: {path}")

    return html_path


if __name__ == "__main__":
    main()
