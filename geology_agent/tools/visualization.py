"""
Visualization Tool
단면도 시각화 및 HTML 보고서 생성 도구 (apply_llm_results.py 래핑)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List


def _add_parent_to_path():
    """부모 디렉토리를 Python 경로에 추가"""
    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))


def generate_visualization(
    output_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    geology_dir: Optional[str] = None,
    dem_file: Optional[str] = None,
    dem_dir: Optional[str] = None,
    target_crs: Optional[str] = None,
    shapefiles: Optional[Dict] = None,
    region_name_kr: Optional[str] = None,
    region_name_en: Optional[str] = None,
) -> Dict[str, Any]:
    """
    시각화 생성 직접 호출

    Returns:
        {
            "success": bool,
            "report_path": str or None,
            "section_images": dict (section_name -> image_path),
            "map_path": str or None,
            "error": str or None
        }
    """
    _add_parent_to_path()

    result = {
        "success": False,
        "report_path": None,
        "section_images": {},
        "map_path": None,
        "error": None,
    }

    # CLI 인자를 임시로 비워서 config_loader 충돌 방지
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]

    # stdout 보존 (apply_llm_results가 sys.stdout을 수정함)
    original_stdout = sys.stdout

    try:
        # 지역별 경로로 레거시 모듈 설정 업데이트
        from config_loader import reconfigure
        reconfigure(
            data_dir=data_dir,
            geology_dir=geology_dir,
            dem_file=dem_file,
            dem_dir=dem_dir,
            output_dir=output_dir,
            target_crs=target_crs,
            shapefiles=shapefiles,
            region_name_kr=region_name_kr,
            region_name_en=region_name_en,
        )

        # apply_llm_results 모듈 임포트
        from apply_llm_results import (
            load_data,
            extract_profile,
            project_geology_with_llm_dip,
            create_section_figure,
            create_summary_plots,
            generate_html_report,
        )

        # stdout 복원 (apply_llm_results가 TextIOWrapper로 교체함)
        sys.stdout = original_stdout

        # 임포트 후 다시 reconfigure (임포트 시 init_config가 덮어쓸 수 있음)
        reconfigure(
            data_dir=data_dir,
            geology_dir=geology_dir,
            dem_file=dem_file,
            dem_dir=dem_dir,
            output_dir=output_dir,
            target_crs=target_crs,
            shapefiles=shapefiles,
            region_name_kr=region_name_kr,
            region_name_en=region_name_en,
        )

        output_path = Path(output_dir) if output_dir else Path("output")

        # 데이터 로드
        print("Loading data...")
        data = load_data()

        # 이전 실행의 단면도 PNG 삭제 (필터링된 단면이 남아있는 것 방지)
        import glob as glob_mod
        for old_png in glob_mod.glob(str(output_path / '*_llm_section.png')):
            os.remove(old_png)
            print(f"  Cleaned up: {Path(old_png).name}")

        # 단면도 생성
        print("Processing cross-sections...")
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

        result["section_images"] = section_images

        # 요약 플롯 생성
        print("Creating summary plots...")
        plots = create_summary_plots(data)

        # 지도 이미지 경로
        map_path = output_path / 'boundary_location_map.png'
        if map_path.exists():
            result["map_path"] = str(map_path)

        # HTML 보고서 생성
        print("Generating HTML report...")
        html_content = generate_html_report(data, section_images, plots)

        html_path = output_path / "llm_geologist_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        result["report_path"] = str(html_path)
        result["success"] = True

        print(f"Report generated: {html_path}")

    except Exception as e:
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    finally:
        # sys.argv 복원
        sys.argv = original_argv

    return result


def generate_section_only(
    section_name: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    특정 단면만 생성
    """
    _add_parent_to_path()

    result = {
        "success": False,
        "image_path": None,
        "error": None,
    }

    # CLI 인자를 임시로 비워서 config_loader 충돌 방지
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]

    # stdout 보존 (apply_llm_results가 sys.stdout을 수정함)
    original_stdout = sys.stdout

    try:
        from apply_llm_results import (
            load_data,
            extract_profile,
            project_geology_with_llm_dip,
            create_section_figure,
        )

        # stdout 복원 (apply_llm_results가 TextIOWrapper로 교체함)
        sys.stdout = original_stdout

        data = load_data()

        # 지정된 단면 찾기
        target_section = None
        for section in data['sections']:
            if section['name'] == section_name:
                target_section = section
                break

        if not target_section:
            result["error"] = f"단면 '{section_name}'을(를) 찾을 수 없습니다."
            return result

        profile_data = extract_profile(target_section)
        projections = project_geology_with_llm_dip(target_section, data, profile_data)

        safe_name = section_name.replace("'", "").replace("-", "_").replace(" ", "_")
        img_path = create_section_figure(target_section, profile_data, projections, safe_name)

        result["image_path"] = img_path
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    finally:
        # sys.argv 복원
        sys.argv = original_argv

    return result


# langchain 도구 버전 (선택적)
try:
    from langchain_core.tools import tool

    @tool
    def generate_visualization_tool(
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        LLM 분석 결과를 적용하여 단면도와 HTML 보고서를 생성합니다.

        Args:
            output_dir: 출력 디렉토리 경로 (llm_geologist_results.json이 있어야 함)

        Returns:
            생성 결과 정보
        """
        return generate_visualization(output_dir=output_dir)

except ImportError:
    generate_visualization_tool = None
