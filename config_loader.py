"""
Configuration Loader for Seoul Geology Cross-Section Project
설정 파일 로더
"""

import argparse
import yaml
import os
import colorsys
from pathlib import Path
from typing import Dict, Any, Optional, List


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments

    Args:
        args: List of arguments (for testing). If None, uses sys.argv.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Seoul Geology Cross-Section Project',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='Path to config.yaml file (default: ./config.yaml)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Override data directory path'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory path'
    )

    parser.add_argument(
        '--dem-file',
        type=str,
        default=None,
        help='Override DEM file path'
    )

    parser.add_argument(
        '--geology-dir',
        type=str,
        default=None,
        help='Override geology shapefile directory path'
    )

    # Shapefile overrides
    parser.add_argument(
        '--shp-litho',
        type=str,
        default=None,
        help='Override lithology shapefile name'
    )

    parser.add_argument(
        '--shp-boundary',
        type=str,
        default=None,
        help='Override boundary shapefile name'
    )

    parser.add_argument(
        '--shp-fault',
        type=str,
        default=None,
        help='Override fault shapefile name'
    )

    parser.add_argument(
        '--shp-foliation',
        type=str,
        default=None,
        help='Override foliation shapefile name'
    )

    parser.add_argument(
        '--shp-crosssection',
        type=str,
        default=None,
        help='Override crosssection shapefile name'
    )

    parser.add_argument(
        '--shp-frame',
        type=str,
        default=None,
        help='Override frame shapefile name'
    )

    return parser.parse_args(args)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        # Default: config.yaml in the same directory as this script
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def get_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Extract path settings from config and convert to Path objects

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with Path objects
    """
    paths = config.get('paths', {})

    return {
        'base_dir': Path(paths.get('base_dir', '.')),
        'data_dir': Path(paths.get('data_dir', './data')),
        'output_dir': Path(paths.get('output_dir', './output')),
        'geology_dir': Path(paths.get('geology_dir', './data/geology')),
        'dem_dir': Path(paths.get('dem_dir', './data/dem')),
        'dem_file': Path(paths.get('dem_file', './data/dem/dem.img')),
    }


def get_shapefiles(config: Dict[str, Any]) -> Dict[str, str]:
    """Get shapefile names from config

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with shapefile names
    """
    return config.get('shapefiles', {
        'litho': 'FG33_Geology_50K_Litho.shp',
        'boundary': 'FG33_Geology_50K_Boundary.shp',
        'fault': 'FG33_Geology_50K_Fault.shp',
        'foliation': 'FG33_Geology_50K_Foliation.shp',
        'crosssection': 'FG33_Geology_50K_Crosssectionline.shp',
        'frame': 'FG33_Geology_50K_Frame.shp',
    })


def get_crs(config: Dict[str, Any]) -> str:
    """Get target CRS from config

    Args:
        config: Configuration dictionary

    Returns:
        CRS string (e.g., "EPSG:5186")
    """
    crs_config = config.get('crs', {})
    return crs_config.get('target', 'EPSG:5186')


def get_litho_colors(config: Dict[str, Any]) -> Dict[str, str]:
    """Get lithology color scheme from config

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary mapping litho codes to colors
    """
    return config.get('litho_colors', {
        'PCEbngn': '#FFB6C1',
        'PCEggn': '#DDA0DD',
        'PCEls': '#87CEEB',
        'PCEqz': '#F0E68C',
        'PCEam': '#90EE90',
        'Pgr': '#FFA07A',
        'Jbgr': '#FF6347',
        'Kqp': '#DEB887',
        'Kqv': '#D2691E',
        'Kfl': '#BC8F8F',
        'Qa': '#FFFACD',
    })


def get_litho_names(config: Dict[str, Any], lang: str = 'kr') -> Dict[str, str]:
    """Get lithology names from config

    Args:
        config: Configuration dictionary
        lang: Language code ('kr' or 'en')

    Returns:
        Dictionary mapping litho codes to names
    """
    key = f'litho_names_{lang}'
    default_kr = {
        'PCEbngn': '호상흑운모편마암',
        'PCEggn': '화강편마암',
        'PCEls': '석회암',
        'PCEqz': '규암',
        'PCEam': '각섬암',
        'Pgr': '반상화강암',
        'Jbgr': '흑운모화강암',
        'Kqp': '석영반암',
        'Kqv': '석영맥',
        'Kfl': '규장암',
        'Qa': '충적층',
    }
    default_en = {
        'PCEbngn': 'Banded Bt Gneiss',
        'PCEggn': 'Granitic Gneiss',
        'PCEls': 'Limestone',
        'PCEqz': 'Quartzite',
        'PCEam': 'Amphibolite',
        'Pgr': 'Porphyritic Granite',
        'Jbgr': 'Bt Granite (Jurassic)',
        'Kqp': 'Quartz Porphyry',
        'Kqv': 'Quartz Vein',
        'Kfl': 'Felsite',
        'Qa': 'Alluvium',
    }

    return config.get(key, default_kr if lang == 'kr' else default_en)


def get_cross_section_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get cross-section parameters from config

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with cross-section parameters
    """
    return config.get('cross_section', {
        'sample_interval': 50,
        'default_depth': 300,
        'default_fault_dip': 75,
    })


def get_visualization_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get visualization parameters from config

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with visualization parameters
    """
    return config.get('visualization', {
        'dpi': 150,
        'font_family': 'Malgun Gothic',
    })


# Convenience function to load everything at once
def load_all_config(config_path: Optional[str] = None, cli_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load config and extract all settings, with CLI argument support

    Args:
        config_path: Path to config file (overrides CLI --config if provided)
        cli_args: List of CLI arguments (for testing). If None, uses sys.argv.

    Returns:
        Dictionary with all extracted settings

    Usage:
        # Default: use config.yaml in script directory
        config = load_all_config()

        # Specify config path directly
        config = load_all_config(config_path="/path/to/config.yaml")

        # Use CLI arguments
        # python script.py --config /path/to/config.yaml --output-dir /custom/output
        config = load_all_config()
    """
    # Parse CLI arguments
    args = parse_args(cli_args)

    # Determine config path: explicit parameter > CLI argument > default
    if config_path is None:
        config_path = args.config

    # Load base config
    config = load_config(config_path)

    # Get paths from config
    paths = get_paths(config)

    # Override paths from CLI arguments
    if args.data_dir:
        paths['data_dir'] = Path(args.data_dir)
        # geology_dir도 config에서 읽은 상대 관계를 유지 (CLI에서 별도 지정 가능)

    if args.output_dir:
        paths['output_dir'] = Path(args.output_dir)

    if args.dem_file:
        paths['dem_file'] = Path(args.dem_file)

    if args.geology_dir:
        paths['geology_dir'] = Path(args.geology_dir)

    # Ensure output directory exists
    paths['output_dir'].mkdir(parents=True, exist_ok=True)

    # Get shapefiles from config
    shapefiles = get_shapefiles(config)

    # Override shapefiles from CLI arguments
    if args.shp_litho:
        shapefiles['litho'] = args.shp_litho
    if args.shp_boundary:
        shapefiles['boundary'] = args.shp_boundary
    if args.shp_fault:
        shapefiles['fault'] = args.shp_fault
    if args.shp_foliation:
        shapefiles['foliation'] = args.shp_foliation
    if args.shp_crosssection:
        shapefiles['crosssection'] = args.shp_crosssection
    if args.shp_frame:
        shapefiles['frame'] = args.shp_frame

    # Build full shapefile paths
    shapefile_paths = {
        key: paths['geology_dir'] / filename
        for key, filename in shapefiles.items()
    }

    return {
        'paths': paths,
        'shapefiles': shapefiles,  # Just filenames
        'shapefile_paths': shapefile_paths,  # Full paths (geology_dir + filename)
        'crs': get_crs(config),
        'litho_colors': get_litho_colors(config),
        'litho_names_kr': get_litho_names(config, 'kr'),
        'litho_names_en': get_litho_names(config, 'en'),
        'cross_section': get_cross_section_params(config),
        'visualization': get_visualization_params(config),
        'raw': config,  # Original config for accessing other settings
        'cli_args': args,  # Expose parsed CLI arguments
    }


def init_config() -> Dict[str, Any]:
    """Initialize configuration with CLI argument support

    This is the recommended entry point for scripts.
    Handles argument parsing and config loading in one call.

    Returns:
        Dictionary with all configuration settings

    Example:
        from config_loader import init_config

        CONFIG = init_config()
        print(CONFIG['paths']['output_dir'])
    """
    return load_all_config()


# Geological color palette for auto-assignment (visually distinct)
_GEO_COLOR_PALETTE = [
    '#E6194B', '#3CB44B', '#4363D8', '#F58231', '#911EB4',
    '#42D4F4', '#F032E6', '#BFEF45', '#FABED4', '#469990',
    '#DCBEFF', '#9A6324', '#800000', '#AAFFC3', '#808000',
    '#FFD8B1', '#000075', '#CD853F', '#B0C4DE', '#8FBC8F',
    '#D2691E', '#BC8F8F', '#DDA0DD', '#87CEEB', '#F0E68C',
]


def auto_populate_litho_info(litho_gdf, litho_colors, litho_names):
    """Shapefile의 LITHOIDX/LITHONAME으로 색상·이름 자동 보충

    config.yaml에 정의되지 않은 암상 코드를 shapefile에서 자동 감지하여
    고유한 색상과 이름을 할당합니다.

    Args:
        litho_gdf: litho shapefile의 GeoDataFrame
        litho_colors: {LITHOIDX: hex_color} 딕셔너리 (in-place 수정)
        litho_names: {LITHOIDX: display_name} 딕셔너리 (in-place 수정)

    Returns:
        (litho_colors, litho_names) 튜플
    """
    if 'LITHOIDX' not in litho_gdf.columns:
        return litho_colors, litho_names

    # 이미 사용 중인 색상을 제외한 가용 색상 목록
    used_colors = set(litho_colors.values())
    available_colors = [c for c in _GEO_COLOR_PALETTE if c not in used_colors]
    color_idx = 0

    unique_lithos = litho_gdf.drop_duplicates(subset=['LITHOIDX'])

    for _, row in unique_lithos.iterrows():
        litho_idx = row.get('LITHOIDX')
        if litho_idx is None or str(litho_idx).strip() == '' or str(litho_idx) == 'nan':
            continue

        litho_idx = str(litho_idx).strip()

        # 색상 자동 할당
        if litho_idx not in litho_colors:
            if color_idx < len(available_colors):
                litho_colors[litho_idx] = available_colors[color_idx]
            else:
                # Fallback: hash 기반 색상
                h = hash(litho_idx) % 360
                r, g, b = colorsys.hls_to_rgb(h / 360, 0.6, 0.7)
                litho_colors[litho_idx] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            color_idx += 1

        # 이름 자동 할당 (LITHONAME 필드에서 읽기)
        if litho_idx not in litho_names and 'LITHONAME' in litho_gdf.columns:
            litho_name = row.get('LITHONAME')
            if litho_name and str(litho_name).strip() and str(litho_name) != 'nan':
                name_str = str(litho_name).strip()
                # 인코딩 깨짐 감지 (CJK 한자가 섞여있으면 깨진 것)
                has_hangul = any('\uac00' <= c <= '\ud7a3' for c in name_str)
                has_cjk_only = any('\u4e00' <= c <= '\u9fff' for c in name_str) and not has_hangul
                if has_cjk_only:
                    litho_names[litho_idx] = litho_idx  # 깨졌으면 코드 그대로
                else:
                    litho_names[litho_idx] = name_str
            else:
                litho_names[litho_idx] = litho_idx

    return litho_colors, litho_names


def auto_populate_age_order(litho_gdf, age_order_dict):
    """Shapefile의 AGE 필드와 코드 접두사로 시대 순서 자동 할당

    Args:
        litho_gdf: litho shapefile의 GeoDataFrame
        age_order_dict: {LITHOIDX: age_order_number} 딕셔너리 (in-place 수정)

    Returns:
        age_order_dict
    """
    if 'LITHOIDX' not in litho_gdf.columns:
        return age_order_dict

    unique_lithos = litho_gdf.drop_duplicates(subset=['LITHOIDX'])

    for _, row in unique_lithos.iterrows():
        litho_idx = str(row.get('LITHOIDX', '')).strip()
        if not litho_idx or litho_idx in age_order_dict:
            continue

        age = str(row.get('AGE', '')).strip() if 'AGE' in litho_gdf.columns else ''

        # AGE 필드와 코드 접두사 기반 시대 순서 자동 할당
        if litho_idx.startswith('Q') or '제4기' in age:
            age_order_dict[litho_idx] = 5   # 제4기 (최신)
        elif litho_idx.startswith('N') or '신제3기' in age or '제3기' in age:
            age_order_dict[litho_idx] = 4.5
        elif litho_idx.startswith('K') or '백악기' in age:
            age_order_dict[litho_idx] = 4   # 백악기
        elif litho_idx.startswith('J') or '쥐라기' in age or '쥬라기' in age:
            age_order_dict[litho_idx] = 3   # 쥐라기
        elif litho_idx.startswith('T') or '삼첩기' in age or '트라이아스기' in age:
            age_order_dict[litho_idx] = 2.5
        elif litho_idx.startswith('P') and not litho_idx.startswith('PCE'):
            age_order_dict[litho_idx] = 2   # 고생대
        elif litho_idx.startswith('PCE') or '선캠브리아' in age:
            age_order_dict[litho_idx] = 1   # 선캠브리아
        else:
            age_order_dict[litho_idx] = 3   # 기본값

    return age_order_dict


def read_shapefile_safe(path, target_crs=None):
    """인코딩 자동 감지로 shapefile 안전하게 읽기

    UTF-8 → EUC-KR → 기본 순서로 시도하여 한국어 shapefile을 올바르게 읽습니다.
    SHAPE_ENCODING='' 설정으로 GDAL의 잘못된 인코딩 자동 변환을 방지합니다.

    Args:
        path: shapefile 경로
        target_crs: 변환할 좌표계 (예: "EPSG:5186")

    Returns:
        GeoDataFrame
    """
    import geopandas as _gpd

    old_env = os.environ.get('SHAPE_ENCODING')
    os.environ['SHAPE_ENCODING'] = ''

    gdf = None
    # UTF-8 우선 시도, 실패 시 euc-kr, 최종 기본값
    for enc in ['utf-8', 'euc-kr', None]:
        try:
            if enc:
                gdf = _gpd.read_file(path, encoding=enc)
            else:
                gdf = _gpd.read_file(path)

            # 한글 LITHONAME이 정상 읽혔는지 검증
            if 'LITHONAME' in gdf.columns:
                sample = gdf['LITHONAME'].dropna().head(1)
                if len(sample) > 0:
                    text = str(sample.iloc[0])
                    # 한글 포함 여부 확인 (가-힣 범위)
                    has_hangul = any('\uac00' <= c <= '\ud7a3' for c in text)
                    if has_hangul:
                        break  # 정상 읽힘
                    # 다음 인코딩 시도
                    continue
            break  # LITHONAME 없으면 인코딩 무관
        except Exception:
            continue

    # 모든 시도 실패 시 기본 읽기
    if gdf is None:
        gdf = _gpd.read_file(path)

    # 환경변수 복원
    if old_env is not None:
        os.environ['SHAPE_ENCODING'] = old_env
    elif 'SHAPE_ENCODING' in os.environ:
        del os.environ['SHAPE_ENCODING']

    if target_crs:
        gdf = gdf.to_crs(target_crs)

    return gdf


def reconfigure(
    data_dir: Optional[str] = None,
    geology_dir: Optional[str] = None,
    dem_file: Optional[str] = None,
    dem_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    target_crs: Optional[str] = None,
    shapefiles: Optional[Dict[str, str]] = None,
    region_name_kr: Optional[str] = None,
    region_name_en: Optional[str] = None,
) -> None:
    """런타임에 레거시 모듈들의 설정을 업데이트 (지역별 경로 반영)

    data_registry에서 가져온 지역별 경로로 이미 임포트된 레거시 스크립트들의
    모듈 전역변수를 패치합니다.

    Args:
        data_dir: 데이터 디렉토리 경로
        geology_dir: 지질도 shapefile 디렉토리 경로
        dem_file: DEM 파일 경로
        dem_dir: DEM 디렉토리 경로
        output_dir: 출력 디렉토리 경로
        target_crs: 목표 좌표계 (예: "EPSG:5186")
        shapefiles: shapefile 이름 딕셔너리 (예: {"litho": "FG33_Geology_50K_Litho.shp", ...})
    """
    import sys

    # 업데이트할 경로 모음
    paths_update = {}
    if data_dir:
        paths_update['data_dir'] = Path(data_dir)
    if geology_dir:
        paths_update['geology_dir'] = Path(geology_dir)
    if dem_file:
        paths_update['dem_file'] = Path(dem_file)
    if dem_dir:
        paths_update['dem_dir'] = Path(dem_dir)
    if output_dir:
        paths_update['output_dir'] = Path(output_dir)

    # shapefile 전체경로 빌드
    shapefile_paths = {}
    if geology_dir and shapefiles:
        geo_path = Path(geology_dir)
        for key, filename in shapefiles.items():
            shapefile_paths[key] = geo_path / filename

    # 레거시 모듈 목록
    module_names = [
        'cross_section_analysis',
        'geologist_agent_llm',
        'apply_llm_results',
        'review_agent_llm',
    ]

    for mod_name in module_names:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue

        # 개별 모듈 전역변수 업데이트
        if data_dir and hasattr(mod, 'BASE_DIR'):
            mod.BASE_DIR = Path(data_dir)
        if geology_dir and hasattr(mod, 'GEOLOGY_DIR'):
            mod.GEOLOGY_DIR = Path(geology_dir)
        if dem_file and hasattr(mod, 'DEM_FILE'):
            mod.DEM_FILE = Path(dem_file)
        if dem_dir and hasattr(mod, 'DEM_DIR'):
            mod.DEM_DIR = Path(dem_dir)
        if output_dir and hasattr(mod, 'OUTPUT_DIR'):
            mod.OUTPUT_DIR = Path(output_dir)
        if target_crs and hasattr(mod, 'TARGET_CRS'):
            mod.TARGET_CRS = target_crs
        if region_name_kr:
            mod.REGION_NAME_KR = region_name_kr
        if region_name_en:
            mod.REGION_NAME_EN = region_name_en

        # CONFIG dict 업데이트
        if hasattr(mod, 'CONFIG') and isinstance(mod.CONFIG, dict):
            if 'paths' in mod.CONFIG:
                for key, value in paths_update.items():
                    mod.CONFIG['paths'][key] = value
            if target_crs:
                mod.CONFIG['crs'] = target_crs
            if shapefiles:
                mod.CONFIG['shapefiles'] = shapefiles
            if shapefile_paths:
                mod.CONFIG['shapefile_paths'] = shapefile_paths

        # PATHS dict 업데이트
        if hasattr(mod, 'PATHS') and isinstance(mod.PATHS, dict):
            for key, value in paths_update.items():
                mod.PATHS[key] = value

    # geologist_agent_llm의 지식베이스를 지역별로 리로드
    # data_dir (원본 지식베이스) → output_dir (자동 생성 지식베이스) 순서로 검색
    search_dirs = [d for d in [data_dir, output_dir] if d]
    if search_dirs:
        geo_mod = sys.modules.get('geologist_agent_llm')
        if geo_mod and hasattr(geo_mod, 'load_knowledge_base'):
            for search_dir in search_dirs:
                if geo_mod.load_knowledge_base(search_dir):
                    break  # 첫 번째 성공한 디렉토리에서 중단
