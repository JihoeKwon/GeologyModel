"""
Cross-Section Analysis Tool
단면 분석 도구 (cross_section_analysis.py + geologist_agent_llm.py 래핑)
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List


def _add_parent_to_path():
    """부모 디렉토리를 Python 경로에 추가"""
    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))


def analyze_sections(
    data_dir: str,
    output_dir: str,
    max_boundaries: int = 20,
    config_path: Optional[str] = None,
    geology_dir: Optional[str] = None,
    dem_file: Optional[str] = None,
    dem_dir: Optional[str] = None,
    target_crs: Optional[str] = None,
    shapefiles: Optional[Dict] = None,
    region_name_kr: Optional[str] = None,
    region_name_en: Optional[str] = None,
) -> Dict[str, Any]:
    """
    단면 분석 직접 호출

    Returns:
        {
            "success": bool,
            "crosssection_analysis_path": str or None,
            "llm_results_path": str or None,
            "existing_sections": list,
            "auto_sections": list,
            "mean_strike": float or None,
            "optimal_azimuth": float or None,
            "boundaries_analyzed": int,
            "error": str or None
        }
    """
    _add_parent_to_path()

    result = {
        "success": False,
        "crosssection_analysis_path": None,
        "llm_results_path": None,
        "existing_sections": [],
        "auto_sections": [],
        "mean_strike": None,
        "optimal_azimuth": None,
        "boundaries_analyzed": 0,
        "error": None,
    }

    try:
        import os
        import sys

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # CLI 인자를 임시로 비워서 config_loader 충돌 방지
        original_argv = sys.argv
        sys.argv = [sys.argv[0]]

        # stdout 보존 (cross_section_analysis와 geologist_agent_llm이 sys.stdout을 수정함)
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

            # 1단계: 단면선 분석 (cross_section_analysis.py 기능)
            print("1. Running cross-section analysis...")
            from cross_section_analysis import (
                load_shapefiles,
                analyze_existing_crosssection,
                analyze_foliation,
                generate_auto_crosssection,
                compare_crosssections,
                save_results,
            )

            # reconfigure 후 다시 호출 (이미 임포트된 모듈의 전역변수 반영)
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

            # 데이터 로드
            data = load_shapefiles()

            # 기존 단면선 분석
            existing_sections = analyze_existing_crosssection(data)
            result["existing_sections"] = existing_sections

            # 엽리 분석으로 최적 방위각 결정
            mean_strike, optimal_azimuth = analyze_foliation(data)
            result["mean_strike"] = float(mean_strike)
            result["optimal_azimuth"] = float(optimal_azimuth)

            # 자동 단면선 생성
            auto_sections = generate_auto_crosssection(data, optimal_azimuth, num_sections=3)
            result["auto_sections"] = auto_sections

            # 비교 및 저장
            comparison = compare_crosssections(existing_sections, auto_sections)
            json_path, geojson_path = save_results(comparison, mean_strike, optimal_azimuth)
            result["crosssection_analysis_path"] = str(json_path)

            # 2단계: LLM 경계 분석 (geologist_agent_llm.py 기능)
            print("2. Running LLM boundary analysis...")
            from geologist_agent_llm import LLMGeologistAgent

            agent = LLMGeologistAgent(Path(config_path) if config_path else None)
            agent.load_data()

            # 경계 분석
            llm_results = agent.analyze_boundaries(max_count=max_boundaries)
            result["boundaries_analyzed"] = len(llm_results)

            # 결과 내보내기
            llm_output_path = output_path / "llm_geologist_results.json"
            agent.export_results(llm_output_path)
            result["llm_results_path"] = str(llm_output_path)

            result["success"] = True

        finally:
            # sys.argv 복원
            sys.argv = original_argv
            # stdout 복원
            sys.stdout = original_stdout

    except Exception as e:
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    return result


# langchain 도구 버전 (선택적)
try:
    from langchain_core.tools import tool

    @tool
    def analyze_sections_tool(
        data_dir: str,
        output_dir: str,
        max_boundaries: int = 20,
    ) -> Dict[str, Any]:
        """
        지질 단면을 분석하고 LLM으로 경계 경사를 추정합니다.

        Args:
            data_dir: 데이터 디렉토리 경로
            output_dir: 출력 디렉토리 경로
            max_boundaries: 분석할 최대 경계 수

        Returns:
            분석 결과 정보
        """
        return analyze_sections(
            data_dir=data_dir,
            output_dir=output_dir,
            max_boundaries=max_boundaries,
        )

except ImportError:
    analyze_sections_tool = None


def run_llm_analysis_only(
    max_boundaries: int = 20,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    LLM 경계 분석만 실행 (단면선 분석이 이미 완료된 경우)
    """
    _add_parent_to_path()

    result = {
        "success": False,
        "llm_results_path": None,
        "boundaries_analyzed": 0,
        "mean_dip": None,
        "error": None,
    }

    # CLI 인자를 임시로 비워서 config_loader 충돌 방지
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]

    try:
        from geologist_agent_llm import LLMGeologistAgent

        agent = LLMGeologistAgent(Path(config_path) if config_path else None)
        agent.load_data()

        llm_results = agent.analyze_boundaries(max_count=max_boundaries)
        result["boundaries_analyzed"] = len(llm_results)

        if llm_results:
            import numpy as np
            dips = [r['dip_angle'] for r in llm_results]
            result["mean_dip"] = float(np.mean(dips))

        # 결과 내보내기
        from config_loader import init_config
        config = init_config()
        output_dir = config['paths']['output_dir']

        llm_output_path = output_dir / "llm_geologist_results.json"
        agent.export_results(llm_output_path)
        result["llm_results_path"] = str(llm_output_path)

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    finally:
        # sys.argv 복원
        sys.argv = original_argv

    return result
