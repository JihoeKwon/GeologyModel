"""
Apply AI Geologist Results to Cross-Sections and Generate HTML Report
AI 분석 결과를 단면도에 적용 및 HTML 보고서 생성
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import LineString, Point
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import base64
from io import BytesIO
from datetime import datetime

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
    print("Loading data...")

    shp_paths = CONFIG['shapefile_paths']
    data = {}
    data['litho'] = gpd.read_file(shp_paths['litho']).to_crs(TARGET_CRS)
    data['boundary'] = gpd.read_file(shp_paths['boundary']).to_crs(TARGET_CRS)
    data['fault'] = gpd.read_file(shp_paths['fault']).to_crs(TARGET_CRS)

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


def get_dip_for_litho_pair(litho1, litho2, llm_results):
    """Get dip estimate for a lithology pair from LLM results"""

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

    # Priority 1: Both lithologies match exactly
    for boundary in llm_results['boundaries']:
        lithos = boundary.get('adjacent_lithologies', [])
        if litho1 in lithos and litho2 in lithos:
            return extract_full_info(boundary)

    # Priority 2: Match by contact type based on rock characteristics
    intrusive_rocks = {'Pgr', 'Jbgr', 'Kqp', 'Kqv', 'Kfl'}
    quaternary = {'Qa'}
    metamorphic = {'PCEbngn', 'PCEggn', 'PCEls', 'PCEqz', 'PCEam'}

    if litho1 in quaternary or litho2 in quaternary:
        expected_type = 'unconformable'
    elif (litho1 in intrusive_rocks or litho2 in intrusive_rocks):
        expected_type = 'intrusive'
    elif litho1 in metamorphic and litho2 in metamorphic:
        expected_type = 'conformable'
    else:
        expected_type = None

    # Find boundary with matching contact type
    if expected_type:
        for boundary in llm_results['boundaries']:
            if boundary.get('contact_type') == expected_type:
                lithos = boundary.get('adjacent_lithologies', [])
                if litho1 in lithos or litho2 in lithos:
                    info = extract_full_info(boundary)
                    info['confidence'] = 'low'
                    return info

    # Priority 3: Use typical values based on contact type
    default_info = {
        'confidence': 'low',
        'reasoning': '전형값 사용',
        'contact_geometry': 'planar',
        'depth_behavior': 'constant',
        'depth_to_flatten': None,
        'intrusion_shape': None,
        'fold_present': False,
        'fold_type': None,
        'fold_axis': None,
    }

    if expected_type == 'unconformable':
        return {**default_info, 'dip_angle': 8, 'dip_direction': 180, 'contact_type': 'unconformable'}
    elif expected_type == 'intrusive':
        return {**default_info, 'dip_angle': 75, 'dip_direction': 135, 'contact_type': 'intrusive',
                'contact_geometry': 'curved', 'depth_behavior': 'flattening', 'intrusion_shape': 'stock'}
    elif expected_type == 'conformable':
        return {**default_info, 'dip_angle': 50, 'dip_direction': 135, 'contact_type': 'conformable'}

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

        for x, y in zip(xs, ys):
            try:
                row, col = rowcol(transform, x, y)
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


def generate_intrusion_boundary(start_d, start_e, min_elev, base_dip, is_left_boundary, length, unit_width=None):
    """
    Generate realistic intrusion boundary (batholith/stock shape).

    관입체 저반(Batholith) 형태 - 수직과장(vertical exaggeration) 고려:
    - 표면에서 40-45° 외향 경사 (수직과장으로 60-70°처럼 보임)
    - 심부로 갈수록 급격히 확장 (플레어 형태)
    - 불규칙한 경계 (자연스러운 굴곡)

    Args:
        is_left_boundary: True면 왼쪽 경계, False면 오른쪽 경계
        unit_width: 암체의 지표 폭 (심부 확장 계산에 사용)
    """
    total_depth = start_e - min_elev
    num_points = 60

    # 관입체 표면 접촉각: 35° (수직과장 2-3x 고려, 시각적으로 55-65° 보임)
    surface_dip = 35

    depths = np.linspace(0, total_depth, num_points)
    x_coords = [start_d]
    y_coords = [start_e]

    # 심부 확장량 (지표 폭의 100-200%)
    effective_width = unit_width or 1000
    max_expansion = effective_width * 1.5

    for i, depth in enumerate(depths[1:], 1):
        progress = depth / total_depth  # 0 to 1

        # 비선형 경사 변화: 심부로 갈수록 급격히 완만해짐
        # 상부 15%: 42° → 38° (급한 상부)
        # 중부 35%: 38° → 18° (빠르게 완만해짐)
        # 하부 50%: 18° → 5° (거의 수평 플레어)
        if progress < 0.15:
            local_dip = surface_dip - progress * 27  # 42° → 38°
        elif progress < 0.5:
            mid_progress = (progress - 0.15) / 0.35  # 0 to 1
            local_dip = 38 - mid_progress * 20  # 38° → 18°
        else:
            bottom_progress = (progress - 0.5) / 0.5  # 0 to 1
            local_dip = 18 - bottom_progress * 13  # 18° → 5°

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


def generate_metamorphic_boundary(start_d, start_e, min_elev, app_dip, length):
    """
    Generate metamorphic rock boundary (conformable, follows foliation).

    변성암은 (수직과장 고려):
    - 서울 지역 엽리 방향 SE 40-50° (수직과장으로 더 가파르게 보임)
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


def generate_curved_boundary(start_d, start_e, min_elev, app_dip, depth_behavior, depth_to_flatten, contact_geometry, intrusion_shape, length, contact_type=None, is_left=True, unit_width=None):
    """
    Generate curved boundary path based on rock type and contact relationship.

    Args:
        contact_type: 'intrusive', 'conformable', 'unconformable', etc.
        is_left: True if this is the left boundary of a rock unit
        unit_width: Width of the rock unit at surface (for intrusion expansion calc)
    """
    # 관입 접촉인 경우 특수 처리
    if contact_type == 'intrusive' or intrusion_shape in ['batholith', 'stock', 'dome']:
        return generate_intrusion_boundary(start_d, start_e, min_elev, app_dip, is_left, length, unit_width=unit_width)

    # 정합 접촉 (변성암) 처리
    if contact_type == 'conformable':
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
    min_elev = np.nanmin(elevations) - 1000
    max_elev = np.nanmax(elevations) + 50
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
    # 전체 단면을 가장 오래된 암석(PCEbngn)으로 먼저 채움
    # 회색 "미분화 기반암"은 검토자가 비과학적이라고 비판하므로
    # 기저 변성암으로 채워서 빈 공간이 발생하지 않도록 함
    base_rock_color = LITHO_COLORS.get('PCEbngn', '#FFB6C1')
    ax1.fill([0, length, length, 0], [max_elev, max_elev, min_elev, min_elev],
             color=base_rock_color, alpha=0.6, zorder=-1)

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

        # 관입암 판별
        intrusive_rocks = {'Pgr', 'Jbgr', 'Kqp', 'Kqv', 'Kfl'}
        is_intrusive = litho_code in intrusive_rocks or contact_type == 'intrusive'

        # 암체의 지표 폭 (관입체 확장 계산에 사용)
        unit_width = end_d - start_d

        # 관입암은 항상 intrusive 접촉 타입 사용 (LLM 결과와 무관)
        effective_contact_type = 'intrusive' if is_intrusive else 'conformable'

        # 곡선 경계 생성 - 관입체는 좌우가 외향으로 경사
        left_curve = generate_curved_boundary(
            start_d, start_e, min_elev, app_dip,
            depth_behavior, depth_to_flatten, contact_geometry, intrusion_shape, length,
            contact_type=effective_contact_type,
            is_left=True,
            unit_width=unit_width
        )
        right_curve = generate_curved_boundary(
            end_d, end_e, min_elev, app_dip,
            depth_behavior, depth_to_flatten, contact_geometry, intrusion_shape, length,
            contact_type=effective_contact_type,
            is_left=False,  # 오른쪽 경계는 오른쪽으로 경사 (외향)
            unit_width=unit_width
        )

        poly_x = [p[0] for p in left_curve] + [p[0] for p in reversed(right_curve)]
        poly_y = [p[1] for p in left_curve] + [p[1] for p in reversed(right_curve)]

        ax1.fill(poly_x, poly_y, color=color, alpha=alpha,
                edgecolor='none', linewidth=0, label=label, zorder=poly_zorder)

        # 경계선 그리기 - 상부 500m만 표시 (너무 깊게 표시하면 혼란 유발)
        contact_color = CONTACT_COLORS.get(proj['contact_type'], 'gray')
        if proj['start_dist'] > 0:
            curve_x = [p[0] for p in left_curve]
            curve_y = [p[1] for p in left_curve]
            # 상부 500m까지만 경계선 표시 (최소 표고 기준)
            max_boundary_depth = 500
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
                    # Add label at centroid
                    centroid = geom.centroid
                    ax.annotate(f"{i+1}", (centroid.x, centroid.y),
                               fontsize=8, fontweight='bold', color='white',
                               bbox=dict(boxstyle='circle', facecolor=color, edgecolor='black', alpha=0.9),
                               ha='center', va='center')
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        ax.plot(*line.xy, color=color, linewidth=3, alpha=0.8)
                    # Label on first segment
                    centroid = geom.geoms[0].centroid
                    ax.annotate(f"{i+1}", (centroid.x, centroid.y),
                               fontsize=8, fontweight='bold', color='white',
                               bbox=dict(boxstyle='circle', facecolor=color, edgecolor='black', alpha=0.9),
                               ha='center', va='center')

    # Plot sections
    for section in data['sections']:
        ax.plot([section['start']['x'], section['end']['x']],
               [section['start']['y'], section['end']['y']],
               'k-', linewidth=2, alpha=0.7)
        mid_x = (section['start']['x'] + section['end']['x']) / 2
        mid_y = (section['start']['y'] + section['end']['y']) / 2
        ax.annotate(section['name'], (mid_x, mid_y), fontsize=9, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('AI 분석 경계 위치 (Analyzed Boundary Locations)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Legend
    for ct, color in CONTACT_COLORS.items():
        if ct != 'fault':
            ax.plot([], [], color=color, linewidth=3, label=ct)
    ax.legend(loc='upper left')

    # Save map as separate file
    fig.savefig(OUTPUT_DIR / 'boundary_location_map.png', dpi=150, bbox_inches='tight', facecolor='white')

    plots['map_view'] = fig_to_base64(fig)

    return plots


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
    <title>AI Geologist Agent Report - 서울 지질 분석</title>
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
            <p>서울 지역 지질 경계 지하구조 추정 보고서</p>
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
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">
                <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                    <span style="display: inline-block; width: 30px; height: 20px; background: #FFB6C1; border: 1px solid #999; margin-right: 10px;"></span>
                    <strong>PCEbngn</strong>&nbsp;- 호상흑운모편마암
                </div>
                <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                    <span style="display: inline-block; width: 30px; height: 20px; background: #DDA0DD; border: 1px solid #999; margin-right: 10px;"></span>
                    <strong>PCEggn</strong>&nbsp;- 화강편마암
                </div>
                <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                    <span style="display: inline-block; width: 30px; height: 20px; background: #87CEEB; border: 1px solid #999; margin-right: 10px;"></span>
                    <strong>PCEls</strong>&nbsp;- 결정질석회암
                </div>
                <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                    <span style="display: inline-block; width: 30px; height: 20px; background: #F0E68C; border: 1px solid #999; margin-right: 10px;"></span>
                    <strong>PCEqz</strong>&nbsp;- 규암
                </div>
                <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                    <span style="display: inline-block; width: 30px; height: 20px; background: #90EE90; border: 1px solid #999; margin-right: 10px;"></span>
                    <strong>PCEam</strong>&nbsp;- 각섬암
                </div>
                <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                    <span style="display: inline-block; width: 30px; height: 20px; background: #FFA07A; border: 1px solid #999; margin-right: 10px;"></span>
                    <strong>Pgr</strong>&nbsp;- 반상화강암
                </div>
                <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                    <span style="display: inline-block; width: 30px; height: 20px; background: #FF6347; border: 1px solid #999; margin-right: 10px;"></span>
                    <strong>Jbgr</strong>&nbsp;- 흑운모화강암
                </div>
                <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                    <span style="display: inline-block; width: 30px; height: 20px; background: #DEB887; border: 1px solid #999; margin-right: 10px;"></span>
                    <strong>Kqp</strong>&nbsp;- 석영반암
                </div>
                <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                    <span style="display: inline-block; width: 30px; height: 20px; background: #D2691E; border: 1px solid #999; margin-right: 10px;"></span>
                    <strong>Kqv</strong>&nbsp;- 석영맥
                </div>
                <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                    <span style="display: inline-block; width: 30px; height: 20px; background: #BC8F8F; border: 1px solid #999; margin-right: 10px;"></span>
                    <strong>Kfl</strong>&nbsp;- 규장암
                </div>
                <div style="display: flex; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                    <span style="display: inline-block; width: 30px; height: 20px; background: #FFFACD; border: 1px solid #999; margin-right: 10px;"></span>
                    <strong>Qa</strong>&nbsp;- 충적층
                </div>
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

    html += """
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
            <p>서울 지역 1:50,000 수치지질도 기반</p>
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
