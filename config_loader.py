"""
Configuration Loader for Seoul Geology Cross-Section Project
설정 파일 로더
"""

import argparse
import yaml
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
        # Also update geology_dir if it was relative to data_dir
        paths['geology_dir'] = paths['data_dir'] / "수치지질도_5만축척_FG33_서울"

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
