"""
Seoul Geology Cross-Section Automation - Section Line Analysis
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load configuration (supports CLI arguments: --config, --data-dir, --output-dir, --dem-file)
from config_loader import init_config

CONFIG = init_config()
PATHS = CONFIG['paths']
SHAPEFILES = CONFIG['shapefiles']

# Path settings from config
BASE_DIR = PATHS['data_dir']
GEOLOGY_DIR = PATHS['geology_dir']
DEM_DIR = PATHS['dem_dir']
OUTPUT_DIR = PATHS['output_dir']

# Target CRS for metric calculations (Korea 2000 / Central Belt)
TARGET_CRS = CONFIG['crs']

# Korean font setting
viz_params = CONFIG['visualization']
plt.rcParams['font.family'] = viz_params.get('font_family', 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

def load_shapefiles():
    """Load geology data and reproject to metric CRS"""
    print("=" * 60)
    print("1. Loading Data")
    print("=" * 60)

    shp_paths = CONFIG['shapefile_paths']
    data = {}

    # Existing cross-sections
    data['crosssection'] = gpd.read_file(shp_paths['crosssection']).to_crs(TARGET_CRS)
    print(f"  - Existing cross-sections: {len(data['crosssection'])}")

    # Foliation data
    data['foliation'] = gpd.read_file(shp_paths['foliation']).to_crs(TARGET_CRS)
    print(f"  - Foliation data: {len(data['foliation'])}")

    # Lithology
    data['litho'] = gpd.read_file(shp_paths['litho']).to_crs(TARGET_CRS)
    print(f"  - Lithology units: {len(data['litho'])}")

    # Geological boundaries
    data['boundary'] = gpd.read_file(shp_paths['boundary']).to_crs(TARGET_CRS)
    print(f"  - Geological boundaries: {len(data['boundary'])}")

    # Faults
    data['fault'] = gpd.read_file(shp_paths['fault']).to_crs(TARGET_CRS)
    print(f"  - Faults: {len(data['fault'])}")

    # Frame
    data['frame'] = gpd.read_file(shp_paths['frame']).to_crs(TARGET_CRS)

    print(f"\n  Target CRS: {TARGET_CRS}")

    # Get bounds
    bounds = data['litho'].total_bounds
    print(f"  Extent: X({bounds[0]:.0f} ~ {bounds[2]:.0f}), Y({bounds[1]:.0f} ~ {bounds[3]:.0f})")

    return data

def analyze_existing_crosssection(data):
    """Analyze existing cross-sections"""
    print("\n" + "=" * 60)
    print("2. Existing Cross-Section Analysis")
    print("=" * 60)

    cs = data['crosssection']

    print(f"\n  Columns: {list(cs.columns)}")
    print(f"  Number of sections: {len(cs)}")

    results = []
    for idx, row in cs.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            start = coords[0]
            end = coords[-1]

            # Calculate azimuth
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            azimuth = np.degrees(np.arctan2(dx, dy)) % 360
            length = geom.length

            name = row.get('CSLID', f'Line_{idx}')
            if name is None or str(name) == 'None':
                name = f'Section_{idx+1}'

            result = {
                'name': str(name),
                'start': start,
                'end': end,
                'azimuth': azimuth,
                'length': length
            }
            results.append(result)

            print(f"\n  Section {idx + 1} ({name}):")
            print(f"    Start: ({start[0]:.1f}, {start[1]:.1f})")
            print(f"    End: ({end[0]:.1f}, {end[1]:.1f})")
            print(f"    Azimuth: {azimuth:.1f} deg")
            print(f"    Length: {length:.1f} m")

    return results

def direction_to_azimuth(direction_str):
    """Convert direction string (NW, NE, etc.) to azimuth"""
    direction_map = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    if isinstance(direction_str, str):
        return direction_map.get(direction_str.upper(), None)
    return None

def analyze_foliation(data):
    """Analyze foliation data for dominant structural direction"""
    print("\n" + "=" * 60)
    print("3. Foliation (Structural) Analysis")
    print("=" * 60)

    fol = data['foliation']

    print(f"\n  Columns: {list(fol.columns)}")

    azimuths = []

    # Try DIPAZI (dip azimuth) first - this is numeric
    if 'DIPAZI' in fol.columns:
        dipazi_values = fol['DIPAZI'].dropna()
        # Convert to numeric, ignoring errors
        dipazi_numeric = []
        for val in dipazi_values:
            try:
                dipazi_numeric.append(float(val))
            except (ValueError, TypeError):
                pass

        if dipazi_numeric:
            # Strike = DipAzi - 90
            strikes = [(d - 90) % 360 for d in dipazi_numeric]
            azimuths.extend(strikes)
            print(f"\n  Using DIPAZI to calculate strike:")
            print(f"    Valid records: {len(dipazi_numeric)}")

    # Try STRIKE column (might be direction strings)
    if 'STRIKE' in fol.columns and len(azimuths) == 0:
        strike_values = fol['STRIKE'].dropna()
        for val in strike_values:
            az = direction_to_azimuth(val)
            if az is not None:
                azimuths.append(az)

        if azimuths:
            print(f"\n  Using STRIKE direction strings:")
            print(f"    Valid records: {len(azimuths)}")

    # If still no azimuths, estimate from boundary lines
    if len(azimuths) == 0:
        print("\n  Estimating strike from geological boundaries...")
        azimuths = estimate_strike_from_boundaries(data['boundary'])

    if azimuths:
        # Circular mean for azimuths
        sin_sum = np.sum([np.sin(np.radians(2 * a)) for a in azimuths])
        cos_sum = np.sum([np.cos(np.radians(2 * a)) for a in azimuths])
        mean_strike = (np.degrees(np.arctan2(sin_sum, cos_sum)) / 2) % 180

        print(f"\n  Strike Statistics:")
        print(f"    Number of measurements: {len(azimuths)}")
        print(f"    Mean strike: {mean_strike:.1f} deg")
        print(f"    (Range: {min(azimuths):.1f} ~ {max(azimuths):.1f} deg)")
    else:
        # Default based on PDF description (SE dipping, strike ~NE-SW)
        mean_strike = 45  # NE-SW direction
        print(f"\n  Using default strike from literature: {mean_strike} deg (NE-SW)")

    # Optimal section azimuth (perpendicular to strike)
    optimal_azimuth = (mean_strike + 90) % 360

    print(f"\n  >>> Optimal section azimuth (perpendicular to strike): {optimal_azimuth:.1f} deg")

    return mean_strike, optimal_azimuth

def estimate_strike_from_boundaries(boundary_gdf):
    """Estimate dominant strike from geological boundary lines"""
    azimuths = []

    for idx, row in boundary_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            coords = []
            for line in geom.geoms:
                coords.extend(list(line.coords))
        else:
            continue

        if len(coords) >= 2:
            for i in range(len(coords) - 1):
                dx = coords[i+1][0] - coords[i][0]
                dy = coords[i+1][1] - coords[i][1]
                seg_len = np.sqrt(dx**2 + dy**2)
                if seg_len > 100:  # Only segments longer than 100m
                    az = np.degrees(np.arctan2(dx, dy)) % 180
                    azimuths.append(az)

    print(f"    Analyzed {len(azimuths)} boundary segments")
    return azimuths

def generate_auto_crosssection(data, optimal_azimuth, num_sections=3):
    """Generate automatic cross-sections based on geological structure"""
    print("\n" + "=" * 60)
    print("4. Auto Cross-Section Generation")
    print("=" * 60)

    # Get map extent
    bounds = data['litho'].total_bounds  # [minx, miny, maxx, maxy]

    print(f"\n  Map extent:")
    print(f"    X: {bounds[0]:.1f} ~ {bounds[2]:.1f} m")
    print(f"    Y: {bounds[1]:.1f} ~ {bounds[3]:.1f} m")

    # Center point
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2

    # Section length = 70% of diagonal
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    diag_length = np.sqrt(width**2 + height**2)
    section_length = diag_length * 0.7

    # Azimuth in radians
    az_rad = np.radians(optimal_azimuth)

    auto_sections = []

    # Perpendicular direction for spacing
    perpendicular_az = np.radians((optimal_azimuth + 90) % 360)
    spacing = min(width, height) * 0.3  # 30% of smaller dimension

    for i in range(num_sections):
        offset = (i - (num_sections - 1) / 2) * spacing

        # Offset center point
        offset_center_x = center_x + offset * np.sin(perpendicular_az)
        offset_center_y = center_y + offset * np.cos(perpendicular_az)

        # Section start and end points
        half_len = section_length / 2
        start_x = offset_center_x - half_len * np.sin(az_rad)
        start_y = offset_center_y - half_len * np.cos(az_rad)
        end_x = offset_center_x + half_len * np.sin(az_rad)
        end_y = offset_center_y + half_len * np.cos(az_rad)

        section = {
            'name': f'Auto_{chr(65+i)}-{chr(65+i)}\'',  # A-A', B-B', C-C'
            'start': (start_x, start_y),
            'end': (end_x, end_y),
            'azimuth': optimal_azimuth,
            'length': section_length
        }
        auto_sections.append(section)

        print(f"\n  Auto Section {i + 1} ({section['name']}):")
        print(f"    Start: ({start_x:.1f}, {start_y:.1f})")
        print(f"    End: ({end_x:.1f}, {end_y:.1f})")
        print(f"    Azimuth: {optimal_azimuth:.1f} deg")
        print(f"    Length: {section_length:.1f} m")

    return auto_sections

def compare_crosssections(existing, auto):
    """Compare existing and auto-generated cross-sections"""
    print("\n" + "=" * 60)
    print("5. Cross-Section Comparison")
    print("=" * 60)

    comparison = {
        'existing': existing,
        'auto': auto,
        'analysis': {}
    }

    if existing:
        existing_azimuths = [s['azimuth'] for s in existing]
        existing_mean_az = np.mean(existing_azimuths)
        existing_lengths = [s['length'] for s in existing]

        print(f"\n  Existing Sections:")
        print(f"    Count: {len(existing)}")
        print(f"    Mean azimuth: {existing_mean_az:.1f} deg")
        print(f"    Mean length: {np.mean(existing_lengths):.1f} m")
        for s in existing:
            print(f"    - {s['name']}: Az={s['azimuth']:.1f} deg, L={s['length']:.1f} m")
    else:
        existing_mean_az = None
        print("\n  Existing Sections: None")

    auto_mean_az = np.mean([s['azimuth'] for s in auto])
    auto_lengths = [s['length'] for s in auto]

    print(f"\n  Auto-Generated Sections:")
    print(f"    Count: {len(auto)}")
    print(f"    Azimuth: {auto_mean_az:.1f} deg (perpendicular to structure)")
    print(f"    Length: {np.mean(auto_lengths):.1f} m")

    if existing_mean_az is not None:
        angle_diff = abs(existing_mean_az - auto_mean_az)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        print(f"\n  === COMPARISON ===")
        print(f"  Azimuth difference: {angle_diff:.1f} deg")

        if angle_diff < 15:
            assessment = "Very similar - existing sections well-aligned with structure"
            assessment_kr = "매우 유사 - 기존 측선이 지질 구조를 잘 반영"
        elif angle_diff < 30:
            assessment = "Similar - minor difference, both acceptable"
            assessment_kr = "유사 - 약간의 차이, 둘 다 적절"
        elif angle_diff < 60:
            assessment = "Different - auto sections may better represent structure"
            assessment_kr = "차이 있음 - 자동 측선이 구조를 더 잘 반영할 수 있음"
        else:
            assessment = "Very different - choose based on purpose"
            assessment_kr = "크게 다름 - 목적에 따라 선택 필요"

        print(f"  Assessment: {assessment}")
        print(f"  평가: {assessment_kr}")

        comparison['analysis'] = {
            'existing_mean_azimuth': float(existing_mean_az),
            'auto_mean_azimuth': float(auto_mean_az),
            'angle_difference': float(angle_diff),
            'assessment': assessment,
            'assessment_kr': assessment_kr
        }

    return comparison

def visualize_comparison(data, existing, auto, comparison):
    """Visualize cross-section comparison"""
    print("\n" + "=" * 60)
    print("6. Visualization")
    print("=" * 60)

    fig = plt.figure(figsize=(18, 10))

    # Left: Plan view
    ax1 = fig.add_subplot(121)

    # Lithology colors from config
    litho_colors = CONFIG['litho_colors']
    litho_names = CONFIG['litho_names_en']

    # Plot lithology
    plotted_lithos = set()
    for idx, row in data['litho'].iterrows():
        litho_idx = row.get('LITHOIDX', 'Unknown')
        color = litho_colors.get(litho_idx, '#CCCCCC')
        label = litho_names.get(litho_idx, litho_idx) if litho_idx not in plotted_lithos else None

        if row.geometry.geom_type == 'Polygon':
            ax1.fill(*row.geometry.exterior.xy, color=color, alpha=0.6,
                    edgecolor='gray', linewidth=0.3, label=label)
        elif row.geometry.geom_type == 'MultiPolygon':
            for j, poly in enumerate(row.geometry.geoms):
                ax1.fill(*poly.exterior.xy, color=color, alpha=0.6,
                        edgecolor='gray', linewidth=0.3,
                        label=label if j == 0 else None)

        if litho_idx not in plotted_lithos:
            plotted_lithos.add(litho_idx)

    # Plot faults
    for idx, row in data['fault'].iterrows():
        if row.geometry is not None and row.geometry.geom_type == 'LineString':
            ax1.plot(*row.geometry.xy, 'k-', linewidth=1.5,
                    label='Fault' if idx == 0 else '')

    # Plot existing sections
    for i, section in enumerate(existing):
        ax1.plot([section['start'][0], section['end'][0]],
                [section['start'][1], section['end'][1]],
                'b-', linewidth=3, label='Existing Section' if i == 0 else '')
        ax1.plot(section['start'][0], section['start'][1], 'bo', markersize=10)
        ax1.plot(section['end'][0], section['end'][1], 'b^', markersize=10)
        # Label
        mid_x = (section['start'][0] + section['end'][0]) / 2
        mid_y = (section['start'][1] + section['end'][1]) / 2
        ax1.annotate(section['name'], (mid_x, mid_y), fontsize=10,
                    color='blue', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot auto sections
    for i, section in enumerate(auto):
        ax1.plot([section['start'][0], section['end'][0]],
                [section['start'][1], section['end'][1]],
                'r--', linewidth=3, label='Auto Section (Structure-based)' if i == 0 else '')
        ax1.plot(section['start'][0], section['start'][1], 'ro', markersize=10)
        ax1.plot(section['end'][0], section['end'][1], 'r^', markersize=10)
        # Label
        mid_x = (section['start'][0] + section['end'][0]) / 2
        mid_y = (section['start'][1] + section['end'][1]) / 2
        ax1.annotate(section['name'], (mid_x, mid_y), fontsize=10,
                    color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_title('Seoul Geological Map - Cross-Section Comparison\n'
                  '서울 지질도 - 측선 비교 (Blue: Existing, Red: Auto)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Right: Rose diagram for azimuth comparison
    ax2 = fig.add_subplot(122, projection='polar')

    # Existing section azimuths
    if existing:
        for i, s in enumerate(existing):
            az_rad = np.radians(90 - s['azimuth'])  # Convert to math angle
            ax2.annotate('', xy=(az_rad, 0.9), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=3))
            ax2.text(az_rad, 1.05, f"{s['azimuth']:.0f}°", ha='center', va='center',
                    fontsize=9, color='blue')

    # Auto section azimuths
    for i, s in enumerate(auto):
        az_rad = np.radians(90 - s['azimuth'])
        ax2.annotate('', xy=(az_rad, 0.7), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3, ls='--'))
        if i == 0:
            ax2.text(az_rad, 0.85, f"{s['azimuth']:.0f}°", ha='center', va='center',
                    fontsize=9, color='red')

    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_title('Section Azimuth Comparison\n측선 방위각 비교\n(Blue: Existing, Red: Auto)', fontsize=12)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([])

    # Add comparison text
    if comparison.get('analysis'):
        analysis = comparison['analysis']
        text = f"Azimuth Difference: {analysis['angle_difference']:.1f}°\n"
        text += f"Assessment: {analysis.get('assessment_kr', '')}"
        fig.text(0.5, 0.02, text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save
    output_path = OUTPUT_DIR / "crosssection_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n  Figure saved: {output_path}")

    plt.close()

    return str(output_path)

def save_results(comparison, mean_strike, optimal_azimuth):
    """Save results to files"""
    print("\n" + "=" * 60)
    print("7. Saving Results")
    print("=" * 60)

    # JSON output
    output_data = {
        'geological_structure': {
            'mean_strike': float(mean_strike),
            'optimal_section_azimuth': float(optimal_azimuth),
            'description': f'Perpendicular to strike {mean_strike:.1f} deg',
            'description_kr': f'주향 {mean_strike:.1f}도에 수직인 방향'
        },
        'existing_crosssections': [
            {
                'name': s['name'],
                'start': {'x': float(s['start'][0]), 'y': float(s['start'][1])},
                'end': {'x': float(s['end'][0]), 'y': float(s['end'][1])},
                'azimuth': float(s['azimuth']),
                'length': float(s['length'])
            } for s in comparison['existing']
        ],
        'auto_crosssections': [
            {
                'name': s['name'],
                'start': {'x': float(s['start'][0]), 'y': float(s['start'][1])},
                'end': {'x': float(s['end'][0]), 'y': float(s['end'][1])},
                'azimuth': float(s['azimuth']),
                'length': float(s['length'])
            } for s in comparison['auto']
        ],
        'comparison': comparison.get('analysis', {}),
        'crs': TARGET_CRS
    }

    json_path = OUTPUT_DIR / "crosssection_analysis.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved: {json_path}")

    # GeoJSON output
    features = []

    for s in comparison['existing']:
        features.append({
            'type': 'Feature',
            'properties': {
                'name': s['name'],
                'type': 'existing',
                'azimuth': s['azimuth'],
                'length': s['length']
            },
            'geometry': {
                'type': 'LineString',
                'coordinates': [list(s['start']), list(s['end'])]
            }
        })

    for s in comparison['auto']:
        features.append({
            'type': 'Feature',
            'properties': {
                'name': s['name'],
                'type': 'auto',
                'azimuth': s['azimuth'],
                'length': s['length']
            },
            'geometry': {
                'type': 'LineString',
                'coordinates': [list(s['start']), list(s['end'])]
            }
        })

    geojson = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {'name': TARGET_CRS}
        },
        'features': features
    }

    geojson_path = OUTPUT_DIR / "crosssections.geojson"
    with open(geojson_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"  GeoJSON saved: {geojson_path}")

    return json_path, geojson_path

def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("  Seoul Geology Cross-Section Automation")
    print("  서울 지질 단면 자동화 - 측선 분석")
    print("=" * 60)

    # 1. Load data
    data = load_shapefiles()

    # 2. Analyze existing cross-sections
    existing_sections = analyze_existing_crosssection(data)

    # 3. Analyze foliation for structural direction
    mean_strike, optimal_azimuth = analyze_foliation(data)

    # 4. Generate auto cross-sections
    auto_sections = generate_auto_crosssection(data, optimal_azimuth, num_sections=3)

    # 5. Compare sections
    comparison = compare_crosssections(existing_sections, auto_sections)

    # 6. Visualize
    fig_path = visualize_comparison(data, existing_sections, auto_sections, comparison)

    # 7. Save results
    json_path, geojson_path = save_results(comparison, mean_strike, optimal_azimuth)

    print("\n" + "=" * 60)
    print("  COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Comparison figure: {fig_path}")
    print(f"  - Analysis data (JSON): {json_path}")
    print(f"  - Cross-sections (GeoJSON): {geojson_path}")

    return comparison

if __name__ == "__main__":
    main()
