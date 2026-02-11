"""
Review Tool
단면도 검토 도구 (review_agent_llm.py 래핑)
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List



def _add_parent_to_path():
    """부모 디렉토리를 Python 경로에 추가"""
    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))


def review_sections(
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
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
    단면도 검토 직접 호출

    Returns:
        {
            "success": bool,
            "total_reviewed": int,
            "average_score": float,
            "reviews": list,
            "critical_issues": list,
            "report_path": str or None,
            "html_report_path": str or None,
            "error": str or None
        }
    """
    _add_parent_to_path()

    result = {
        "success": False,
        "total_reviewed": 0,
        "average_score": 0.0,
        "reviews": [],
        "critical_issues": [],
        "report_path": None,
        "html_report_path": None,
        "error": None,
    }

    # CLI 인자를 임시로 비워서 config_loader 충돌 방지
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]

    # stdout 보존 (review_agent_llm이 sys.stdout을 수정함)
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

        from review_agent_llm import GeologyReviewAgent

        # stdout 복원
        sys.stdout = original_stdout

        # 임포트 후 다시 reconfigure
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

        # 검토 에이전트 초기화
        print("Initializing Review Agent...")
        agent = GeologyReviewAgent(Path(config_path) if config_path else None)

        # 모든 단면도 검토
        print("Reviewing cross-sections...")
        reviews = agent.review_all_sections(output_path)

        if reviews:
            result["reviews"] = reviews
            result["total_reviewed"] = len(reviews)

            # 통계 계산
            scores = [r.get('overall_score', 0) for r in reviews]
            result["average_score"] = sum(scores) / len(scores) if scores else 0.0

            # 중요 문제점 수집
            for r in reviews:
                for issue in r.get('critical_issues', []):
                    issue['section'] = r.get('section_name', 'Unknown')
                    result["critical_issues"].append(issue)

            # 결과 내보내기
            json_path = agent.export_results(output_path / "review_results.json")
            result["report_path"] = str(json_path)

            # 텍스트 보고서 생성
            report = agent.generate_report()
            report_txt_path = output_path / "review_report.txt"
            with open(report_txt_path, 'w', encoding='utf-8') as f:
                f.write(report)

            # HTML 보고서 생성
            html_path = agent.generate_html_report(output_path / "review_report.html")
            result["html_report_path"] = str(html_path)

            result["success"] = True
            print(f"Review complete. Average score: {result['average_score']:.1f}/10")
        else:
            result["error"] = "검토할 단면도를 찾을 수 없습니다."

    except Exception as e:
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    finally:
        # sys.argv 복원
        sys.argv = original_argv

    return result


def review_single_section(
    image_path: str,
    section_name: str,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    단일 단면도 검토
    """
    _add_parent_to_path()

    result = {
        "success": False,
        "score": None,
        "assessment": None,
        "critical_issues": [],
        "summary": None,
        "error": None,
    }

    # CLI 인자를 임시로 비워서 config_loader 충돌 방지
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]

    # stdout 보존 (review_agent_llm이 sys.stdout을 수정함)
    original_stdout = sys.stdout

    try:
        from review_agent_llm import GeologyReviewAgent

        # stdout 복원
        sys.stdout = original_stdout

        agent = GeologyReviewAgent(Path(config_path) if config_path else None)
        review = agent.review_section(Path(image_path), section_name)

        if review:
            result["score"] = review.get('overall_score', 0)
            result["assessment"] = review.get('overall_assessment', 'unknown')
            result["critical_issues"] = review.get('critical_issues', [])
            result["summary"] = review.get('summary_korean', '')
            result["success"] = True
        else:
            result["error"] = "검토 실패"

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
    def review_sections_tool(
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        생성된 단면도를 검토하고 품질 평가를 수행합니다.

        Args:
            output_dir: 출력 디렉토리 경로 (단면도 이미지가 있어야 함)

        Returns:
            검토 결과 정보
        """
        return review_sections(output_dir=output_dir)

except ImportError:
    review_sections_tool = None
