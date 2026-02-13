"""
Geologist Agent with LLM API Integration
LLM 기반 지하구조 추정 에이전트
"""

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
from typing import List, Dict, Tuple, Optional
from pyproj import CRS, Transformer
import anthropic
import yaml
import os
import sys

# 범용 지질학 원리 모듈
try:
    from geological_principles import generate_principles_context
    PRINCIPLES_LOADED = True
except ImportError:
    generate_principles_context = None
    PRINCIPLES_LOADED = False

# Load configuration (supports CLI arguments: --config, --data-dir, --output-dir, --dem-file)
from config_loader import init_config

CONFIG = init_config()
PATHS = CONFIG['paths']
SHAPEFILES = CONFIG['shapefiles']

# Paths from config
CONFIG_FILE = PATHS['base_dir'] / "config.yaml"
BASE_DIR = PATHS['data_dir']
GEOLOGY_DIR = PATHS['geology_dir']
DEM_FILE = PATHS['dem_file']
OUTPUT_DIR = PATHS['output_dir']

TARGET_CRS = CONFIG['crs']

# 지역명 (reconfigure()로 업데이트됨)
REGION_NAME_KR = "서울"
REGION_NAME_EN = "Seoul"

# =============================================================================
# 지식베이스 동적 로딩
# =============================================================================

KNOWLEDGE_BASE_LOADED = False
generate_context_for_boundary = None
generate_regional_context = None

DEFAULT_GEOLOGICAL_CONTEXT = """
# 지질 개요
- 정합적 접촉: 40-70° 경사
- 관입 접촉: 70-90° 급경사
- 부정합 접촉: 0-15° 수평
"""

GEOLOGICAL_CONTEXT = DEFAULT_GEOLOGICAL_CONTEXT


def _dict_context_to_str(ctx_dict):
    """지식베이스 dict를 LLM 프롬프트용 텍스트로 변환"""
    lines = []
    for key, value in ctx_dict.items():
        title = key.replace('_', ' ').title()
        lines.append(f"\n## {title}")
        if isinstance(value, dict):
            for k, v in value.items():
                lines.append(f"- **{k}**: {v}")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for k, v in item.items():
                        lines.append(f"- {k}: {v}")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append(str(value))
    return '\n'.join(lines)


def load_knowledge_base(data_dir=None):
    """지역별 지식베이스 동적 로딩"""
    global KNOWLEDGE_BASE_LOADED, generate_context_for_boundary, generate_regional_context, GEOLOGICAL_CONTEXT

    search_dir = data_dir or str(PATHS['data_dir'])
    if str(search_dir) not in sys.path:
        sys.path.insert(0, str(search_dir))

    # data_dir 내 *_geological_knowledge*.py 파일 찾기
    import glob
    kb_files = glob.glob(str(Path(search_dir) / "*geological_knowledge*.py"))
    # auto 생성본 제외, 원본 우선
    kb_files = [f for f in kb_files if '_auto' not in f and '_vision' not in f]
    if not kb_files:
        # auto 버전이라도 사용
        kb_files = glob.glob(str(Path(search_dir) / "*geological_knowledge*.py"))

    if kb_files:
        kb_path = Path(kb_files[0])
        module_name = kb_path.stem
        try:
            import importlib
            if module_name in sys.modules:
                # 이미 로드된 모듈이면 리로드
                mod = importlib.reload(sys.modules[module_name])
            else:
                spec = importlib.util.spec_from_file_location(module_name, str(kb_path))
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)

            generate_context_for_boundary = getattr(mod, 'generate_context_for_boundary', None)
            generate_regional_context = getattr(mod, 'generate_regional_context', None)
            KNOWLEDGE_BASE_LOADED = True

            if generate_regional_context:
                ctx = generate_regional_context()
                # dict 반환 시 LLM 프롬프트용 텍스트로 변환
                if isinstance(ctx, dict):
                    GEOLOGICAL_CONTEXT = _dict_context_to_str(ctx)
                elif isinstance(ctx, str):
                    GEOLOGICAL_CONTEXT = ctx
                else:
                    GEOLOGICAL_CONTEXT = str(ctx)

            print(f"  Knowledge base loaded: {kb_path.name}")
            return True
        except Exception as e:
            print(f"  Warning: Failed to load knowledge base {kb_path.name}: {e}")

    KNOWLEDGE_BASE_LOADED = False
    GEOLOGICAL_CONTEXT = DEFAULT_GEOLOGICAL_CONTEXT
    print(f"  Warning: No knowledge base found in {search_dir}. Using default context.")
    return False


# 초기 로딩 (서버 시작 시 기본 data_dir에서)
load_knowledge_base()


# =============================================================================
# 데이터 수집 함수
# =============================================================================

def load_data():
    """Load geological data"""
    print("Loading geological data...")
    shp_paths = CONFIG['shapefile_paths']
    data = {}
    from config_loader import read_shapefile_safe
    data['litho'] = read_shapefile_safe(shp_paths['litho'], target_crs=TARGET_CRS)
    data['boundary'] = gpd.read_file(shp_paths['boundary']).to_crs(TARGET_CRS)
    data['fault'] = gpd.read_file(shp_paths['fault']).to_crs(TARGET_CRS)
    data['foliation'] = gpd.read_file(shp_paths['foliation']).to_crs(TARGET_CRS)
    return data


def get_boundary_data(boundary_geom, litho_gdf, dem_path, sample_interval=100):
    """Extract comprehensive data for a boundary"""

    if boundary_geom is None or boundary_geom.is_empty:
        return None

    # Get line geometry
    if boundary_geom.geom_type == 'LineString':
        line = boundary_geom
    elif boundary_geom.geom_type == 'MultiLineString':
        line = max(boundary_geom.geoms, key=lambda x: x.length)
    else:
        return None

    length = line.length
    if length < 50:  # Skip very short boundaries
        return None

    # Sample points along boundary
    num_samples = max(5, min(20, int(length / sample_interval)))

    points_data = []
    with rasterio.open(dem_path) as dem:
        dem_data = dem.read(1)
        transform = dem.transform
        nodata = dem.nodata

        # DEM CRS와 TARGET_CRS가 다르면 좌표 변환 준비
        dem_crs = CRS(dem.crs) if dem.crs else None
        target_crs = CRS(TARGET_CRS)
        need_transform = dem_crs is not None and dem_crs != target_crs
        if need_transform:
            coord_transformer = Transformer.from_crs(target_crs, dem_crs, always_xy=True)

        for i in range(num_samples + 1):
            fraction = i / num_samples
            point = line.interpolate(fraction, normalized=True)
            x, y = point.x, point.y

            try:
                # TARGET_CRS → DEM CRS 변환 (필요 시)
                dx, dy = coord_transformer.transform(x, y) if need_transform else (x, y)
                row, col = rowcol(transform, dx, dy)
                if 0 <= row < dem.height and 0 <= col < dem.width:
                    elev = dem_data[row, col]
                    if nodata and elev == nodata:
                        elev = None
                    else:
                        elev = float(elev)
                else:
                    elev = None
            except:
                elev = None

            points_data.append({
                'x': round(x, 1),
                'y': round(y, 1),
                'elevation': round(elev, 1) if elev else None,
                'distance_along': round(fraction * length, 1)
            })

    # Find adjacent lithologies
    buffer = line.buffer(50)
    intersecting = litho_gdf[litho_gdf.intersects(buffer)]

    adjacent_lithos = []
    for idx, row in intersecting.iterrows():
        litho_idx = row.get('LITHOIDX', 'Unknown')
        if litho_idx not in adjacent_lithos:
            adjacent_lithos.append(litho_idx)

    # Calculate boundary orientation
    coords = list(line.coords)
    if len(coords) >= 2:
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        azimuth = np.degrees(np.arctan2(dx, dy)) % 360
    else:
        azimuth = None

    # Elevation statistics
    elevations = [p['elevation'] for p in points_data if p['elevation'] is not None]

    return {
        'length': round(length, 1),
        'azimuth': round(azimuth, 1) if azimuth else None,
        'adjacent_lithologies': adjacent_lithos[:3],  # Limit to 3
        'sample_points': points_data,
        'elevation_range': {
            'min': round(min(elevations), 1) if elevations else None,
            'max': round(max(elevations), 1) if elevations else None,
            'range': round(max(elevations) - min(elevations), 1) if len(elevations) > 1 else 0
        }
    }


def get_nearby_foliation(boundary_geom, foliation_gdf, search_radius=2000):
    """Get nearby foliation measurements"""

    if boundary_geom is None:
        return []

    centroid = boundary_geom.centroid
    buffer = centroid.buffer(search_radius)

    nearby = foliation_gdf[foliation_gdf.intersects(buffer)]

    measurements = []
    for idx, row in nearby.iterrows():
        strike = row.get('STRIKE')
        dip_dir = row.get('DIP')
        dip_angle = row.get('DIPANGLE')

        if dip_angle and dip_angle > 0:
            measurements.append({
                'strike': strike,
                'dip_direction': dip_dir,
                'dip_angle': int(dip_angle)
            })

    return measurements[:5]  # Limit to 5 nearest


# =============================================================================
# LLM Geologist Agent
# =============================================================================

class LLMGeologistAgent:
    """LLM-based geological inference agent"""

    def __init__(self, config_path: Path = None):
        """Initialize with Anthropic API from config file"""
        config_path = config_path or CONFIG_FILE

        # Load config from YAML
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            self.api_key = config.get('anthropic', {}).get('api_key')
            self.model = config.get('model', {}).get('name', 'claude-sonnet-4-20250514')
            self.max_tokens = config.get('model', {}).get('max_tokens', 1024)
            self.max_boundaries = config.get('analysis', {}).get('max_boundaries', 20)
            self.system_prompt = config.get('persona', {}).get('system_prompt', '')
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Check API key
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            raise ValueError(f"API 키를 config.yaml 파일에 입력해주세요: {config_path}")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.data = None
        self.results = []

        print(f"  Model: {self.model}")
        print(f"  Max boundaries to analyze: {self.max_boundaries}")
        print(f"  System persona: {'Loaded' if self.system_prompt else 'Not configured'}")
        print(f"  Knowledge base: {'Loaded' if KNOWLEDGE_BASE_LOADED else 'Default context only'}")
        print(f"  Geological principles: {'Loaded' if PRINCIPLES_LOADED else 'Not available'}")

    def load_data(self):
        """Load geological data"""
        self.data = load_data()
        print(f"  Loaded {len(self.data['litho'])} lithology units")
        print(f"  Loaded {len(self.data['boundary'])} boundaries")
        print(f"  Loaded {len(self.data['foliation'])} foliation measurements")

    def _create_inference_prompt(self, boundary_data: Dict, foliation_data: List) -> str:
        """Create prompt for LLM inference"""

        prompt = f"""## Regional Geological Context
{GEOLOGICAL_CONTEXT}
"""
        # Add universal geological principles
        if PRINCIPLES_LOADED and generate_principles_context:
            lithologies = boundary_data.get('adjacent_lithologies', [])
            principles_text = generate_principles_context(
                rock_codes=lithologies, region_name=REGION_NAME_KR
            )
            prompt += f"""
## 범용 지질학 원리 (Universal Geological Principles)
{principles_text}
"""

        # Add dynamic knowledge base context if available
        if KNOWLEDGE_BASE_LOADED and generate_context_for_boundary:
            lithologies = boundary_data.get('adjacent_lithologies', [])
            dynamic_context = generate_context_for_boundary(lithologies)
            if dynamic_context:
                # dict 반환 시 텍스트로 변환
                if isinstance(dynamic_context, dict):
                    dynamic_context = _dict_context_to_str(dynamic_context)
                prompt += f"""
## Specific Knowledge for This Boundary
{dynamic_context}
"""

        prompt += f"""
## Boundary Data to Analyze

**Adjacent Rock Units**: {', '.join(boundary_data['adjacent_lithologies'])}

**Boundary Geometry**:
- Length: {boundary_data['length']} m
- Surface azimuth: {boundary_data['azimuth']}°

**Elevation Profile Along Boundary**:
"""
        # Add elevation data
        for pt in boundary_data['sample_points']:
            if pt['elevation']:
                prompt += f"- Distance {pt['distance_along']}m: Elevation {pt['elevation']}m (X:{pt['x']}, Y:{pt['y']})\n"

        prompt += f"""
**Elevation Range**: {boundary_data['elevation_range']['min']} - {boundary_data['elevation_range']['max']} m (range: {boundary_data['elevation_range']['range']} m)

**Nearby Foliation Measurements** (within 2km):
"""
        if foliation_data:
            for f in foliation_data:
                prompt += f"- Strike: {f['strike']}, Dip: {f['dip_angle']}° toward {f['dip_direction']}\n"
        else:
            prompt += "- No nearby measurements available\n"

        prompt += """
## Your Task

Based on the geological context and data above, estimate the **complete subsurface geometry** of this boundary.
Think like a structural geologist constructing a 3D conceptual model.

Provide your analysis in the following JSON format:
```json
{
    "dip_angle": <estimated dip angle in degrees, 0-90>,
    "dip_direction": <azimuth of dip direction in degrees, 0-360>,
    "strike": <strike azimuth in degrees, 0-360>,
    "confidence": "<high/medium/low/very_low>",
    "contact_type": "<conformable/intrusive/unconformable/fault>",

    "contact_geometry": "<planar/curved/irregular>",
    "geometry_description": "<접촉면의 3D 형태 설명 - 예: 평면, 돔형, 포물선형, 불규칙 등>",

    "depth_behavior": {
        "dip_change": "<steepening/constant/flattening>",
        "estimated_depth_to_flatten": <깊이(m) 또는 null if constant>,
        "description": "<깊이에 따른 경사 변화 설명>"
    },

    "intrusion_shape": "<dome/stock/dike/sill/batholith/null if not intrusive>",

    "fold_influence": {
        "present": <true/false>,
        "fold_type": "<anticline/syncline/monocline/null>",
        "axis_trend": <축 방향 방위각 또는 null>,
        "description": "<습곡 영향 설명>"
    },

    "reasoning": "<(적용 원리) + (지역 데이터 근거) → (결론) 형식으로 상세 추론, 한국어>"
}
```

Consider:
1. **Contact type & 3D geometry**: Is this a planar contact or does it curve at depth?
2. **Intrusion mechanics**: If intrusive, what shape? Granite batholiths often have outward-dipping contacts (~70-80°) that may steepen or flatten with depth
3. **Depth behavior**: Do contacts typically steepen (listric) or flatten (sole out) at depth?
4. **Fold influence**: Is there evidence of folding affecting this contact?
5. **V-rule & elevation profile**: What does the topographic expression tell us about geometry?
6. **Regional structure**: Consider the dominant structural trends of this area

Respond ONLY with the JSON object, no additional text."""

        return prompt

    def infer_boundary_dip(self, boundary_idx: int) -> Optional[Dict]:
        """Use LLM to infer dip for a single boundary"""

        boundary = self.data['boundary'].iloc[boundary_idx]
        geom = boundary.geometry

        # Collect boundary data
        boundary_data = get_boundary_data(geom, self.data['litho'], DEM_FILE)
        if not boundary_data:
            return None

        # Get nearby foliation
        foliation_data = get_nearby_foliation(geom, self.data['foliation'])

        # Create prompt
        prompt = self._create_inference_prompt(boundary_data, foliation_data)

        # Call LLM
        try:
            api_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            if self.system_prompt:
                api_params["system"] = self.system_prompt

            response = self.client.messages.create(**api_params)

            response_text = response.content[0].text.strip()

            # Parse JSON response
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)

            # Add metadata
            result['boundary_idx'] = boundary_idx
            result['adjacent_lithologies'] = boundary_data['adjacent_lithologies']
            result['boundary_length'] = boundary_data['length']
            result['elevation_range'] = boundary_data['elevation_range']['range']

            return result

        except json.JSONDecodeError as e:
            print(f"    JSON parse error for boundary {boundary_idx}: {e}")
            print(f"    Response: {response_text[:200]}...")
            return None
        except Exception as e:
            print(f"    Error analyzing boundary {boundary_idx}: {e}")
            return None

    def analyze_boundaries(self, indices: List[int] = None, max_count: int = 20):
        """Analyze multiple boundaries"""

        if indices is None:
            # Select diverse boundaries
            indices = self._select_representative_boundaries(max_count)

        print(f"\nAnalyzing {len(indices)} boundaries with LLM...")

        results = []
        for i, idx in enumerate(indices):
            print(f"  [{i+1}/{len(indices)}] Boundary #{idx}...", end=" ")

            result = self.infer_boundary_dip(idx)

            if result:
                results.append(result)
                print(f"Dip: {result['dip_angle']}° → {result['dip_direction']}° ({result['confidence']})")
            else:
                print("Failed")

        self.results = results
        return results

    def _select_representative_boundaries(self, max_count: int) -> List[int]:
        """Select diverse representative boundaries for analysis"""

        selected = []

        # Group by adjacent lithology pairs
        litho_groups = {}

        for idx in range(len(self.data['boundary'])):
            geom = self.data['boundary'].iloc[idx].geometry
            if geom is None or geom.is_empty or geom.length < 100:
                continue

            buffer = geom.buffer(50)
            intersecting = self.data['litho'][self.data['litho'].intersects(buffer)]

            lithos = sorted([r.get('LITHOIDX', 'Unknown') for _, r in intersecting.iterrows()])[:2]
            key = tuple(lithos)

            if key not in litho_groups:
                litho_groups[key] = []
            litho_groups[key].append(idx)

        # Select from each group
        per_group = max(1, max_count // len(litho_groups))

        for key, indices in litho_groups.items():
            # Sort by length (prefer longer boundaries)
            indices_sorted = sorted(indices,
                                   key=lambda i: self.data['boundary'].iloc[i].geometry.length,
                                   reverse=True)
            selected.extend(indices_sorted[:per_group])

            if len(selected) >= max_count:
                break

        return selected[:max_count]

    def generate_report(self) -> str:
        """Generate analysis report"""

        if not self.results:
            return "No results available."

        report = []
        report.append("=" * 70)
        report.append("LLM GEOLOGIST AGENT - SUBSURFACE STRUCTURE INFERENCE REPORT")
        report.append("LLM 지질 에이전트 - 지하구조 추정 보고서")
        report.append("=" * 70)

        # Statistics
        dips = [r['dip_angle'] for r in self.results]
        confidences = [r['confidence'] for r in self.results]

        report.append(f"\n분석된 경계: {len(self.results)}")
        report.append(f"\n경사각 통계:")
        report.append(f"  평균: {np.mean(dips):.1f}°")
        report.append(f"  중앙값: {np.median(dips):.1f}°")
        report.append(f"  범위: {min(dips):.0f}° ~ {max(dips):.0f}°")

        report.append(f"\n신뢰도 분포:")
        for c in ['high', 'medium', 'low', 'very_low']:
            count = confidences.count(c)
            if count > 0:
                report.append(f"  {c}: {count} ({100*count/len(confidences):.0f}%)")

        # Contact types
        contact_types = [r.get('contact_type', 'unknown') for r in self.results]
        report.append(f"\n접촉 유형:")
        for ct in set(contact_types):
            count = contact_types.count(ct)
            report.append(f"  {ct}: {count}")

        # Detailed results
        report.append(f"\n" + "=" * 70)
        report.append("상세 분석 결과")
        report.append("=" * 70)

        for r in self.results:
            report.append(f"\n경계 #{r['boundary_idx']}: {' - '.join(r['adjacent_lithologies'])}")
            report.append(f"  길이: {r['boundary_length']:.0f}m, 표고차: {r['elevation_range']:.0f}m")
            report.append(f"  추정: 주향 {r['strike']:.0f}°, 경사 {r['dip_angle']:.0f}° → {r['dip_direction']:.0f}°")
            report.append(f"  접촉유형: {r.get('contact_type', 'unknown')}, 신뢰도: {r['confidence']}")
            report.append(f"  추론: {r['reasoning']}")

        return '\n'.join(report)

    def export_results(self, output_path: Path):
        """Export results to JSON"""

        output_data = {
            'summary': {
                'total_analyzed': len(self.results),
                'mean_dip': float(np.mean([r['dip_angle'] for r in self.results])) if self.results else 0,
                'model_used': self.model,
                'system_persona': bool(self.system_prompt)
            },
            'boundaries': self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\nResults exported to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  LLM GEOLOGIST AGENT")
    print("  LLM 기반 지하구조 추정 에이전트")
    print("=" * 70)

    # Initialize agent
    try:
        agent = LLMGeologistAgent()
        print("\nAnthropic API connected successfully.")
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set ANTHROPIC_API_KEY environment variable.")
        return None

    # Load data
    agent.load_data()

    # Analyze boundaries
    results = agent.analyze_boundaries(max_count=agent.max_boundaries)

    # Generate report
    report = agent.generate_report()
    print("\n" + report)

    # Export results
    agent.export_results(OUTPUT_DIR / "llm_geologist_results.json")

    print("\n" + "=" * 70)
    print("  COMPLETE!")
    print("=" * 70)

    return agent


if __name__ == "__main__":
    main()
