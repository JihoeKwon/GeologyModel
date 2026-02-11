"""
LangGraph Workflow for Geology Agent
지질단면 분석 워크플로우 정의
"""

import uuid
from typing import Dict, Any, Optional, Literal
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import GeologyAgentState
from .data_registry import check_region_data, get_base_dir
from .tools.data_check import check_data
from .tools.knowledge import extract_knowledge
from .tools.analysis import analyze_sections
from .tools.visualization import generate_visualization
from .tools.review import review_sections


def _sanitize_for_serialization(obj):
    """numpy/특수 타입을 Python 네이티브 타입으로 변환 (msgpack 직렬화용)"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_serialization(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_serialization(v) for v in obj]
    # numpy 타입 처리
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    return obj


# =============================================================================
# Node Functions
# =============================================================================

def check_data_node(state: GeologyAgentState) -> Dict[str, Any]:
    """데이터 존재 확인 노드"""
    region = state.get("region", "서울")

    print(f"\n{'='*60}")
    print(f"[Step 1/6] Checking data for region: {region}")
    print(f"{'='*60}")

    result = check_data(region)

    return {
        "data_available": result["available"],
        "data_paths": result["paths"],
        "current_step": "check_data",
        "error": None if result["available"] else result["message"],
    }


def extract_knowledge_node(state: GeologyAgentState) -> Dict[str, Any]:
    """지식베이스 추출 노드"""
    print(f"\n{'='*60}")
    print("[Step 2/6] Extracting geological knowledge from documents")
    print(f"{'='*60}")

    data_paths = state.get("data_paths", {})
    data_dir = data_paths.get("data_dir")
    output_dir = data_paths.get("output_dir")

    # 기존 지식베이스가 있으면 스킵
    existing_kb = data_paths.get("knowledge_base_path")
    if existing_kb:
        print(f"Using existing knowledge base: {existing_kb}")
        return {
            "knowledge_extracted": True,
            "knowledge_base_path": existing_kb,
            "current_step": "extract_knowledge",
        }

    # 새로 추출
    region = state.get("region", "seoul")
    result = extract_knowledge(
        data_dir=data_dir,
        output_dir=output_dir,
        region_name=region.lower(),
        max_pages=60,
    )

    return {
        "knowledge_extracted": result["success"],
        "knowledge_base_path": result.get("knowledge_base_path"),
        "current_step": "extract_knowledge",
        "error": result.get("error"),
    }


def analyze_sections_node(state: GeologyAgentState) -> Dict[str, Any]:
    """단면 분석 노드"""
    print(f"\n{'='*60}")
    print("[Step 3/6] Analyzing cross-sections and geological boundaries")
    print(f"{'='*60}")

    data_paths = state.get("data_paths", {})
    data_dir = data_paths.get("data_dir")
    output_dir = data_paths.get("output_dir")

    result = analyze_sections(
        data_dir=data_dir,
        output_dir=output_dir,
        max_boundaries=5,
        geology_dir=data_paths.get("geology_dir"),
        dem_file=data_paths.get("dem_file"),
        dem_dir=data_paths.get("dem_dir"),
        target_crs=data_paths.get("crs"),
        shapefiles=data_paths.get("shapefiles"),
        region_name_kr=data_paths.get("region_name_kr"),
        region_name_en=data_paths.get("region_name_en"),
    )

    sections_data = None
    if result["success"]:
        sections_data = {
            "existing_sections": result.get("existing_sections", []),
            "auto_sections": result.get("auto_sections", []),
            "mean_strike": result.get("mean_strike"),
            "optimal_azimuth": result.get("optimal_azimuth"),
            "boundaries_analyzed": result.get("boundaries_analyzed", 0),
        }

    return _sanitize_for_serialization({
        "sections_analyzed": result["success"],
        "boundaries_analyzed": result["success"],
        "sections_data": sections_data,
        "llm_results_path": result.get("llm_results_path"),
        "current_step": "analyze_sections",
        "error": result.get("error"),
    })


def generate_visualization_node(state: GeologyAgentState) -> Dict[str, Any]:
    """시각화 생성 노드"""
    print(f"\n{'='*60}")
    print("[Step 4/6] Generating cross-section visualizations and report")
    print(f"{'='*60}")

    data_paths = state.get("data_paths", {})
    output_dir = data_paths.get("output_dir")

    result = generate_visualization(
        output_dir=output_dir,
        data_dir=data_paths.get("data_dir"),
        geology_dir=data_paths.get("geology_dir"),
        dem_file=data_paths.get("dem_file"),
        dem_dir=data_paths.get("dem_dir"),
        target_crs=data_paths.get("crs"),
        shapefiles=data_paths.get("shapefiles"),
        region_name_kr=data_paths.get("region_name_kr"),
        region_name_en=data_paths.get("region_name_en"),
    )

    return _sanitize_for_serialization({
        "visualization_generated": result["success"],
        "section_images": result.get("section_images"),
        "report_path": result.get("report_path"),
        "current_step": "generate_visualization",
        "error": result.get("error"),
    })


def review_node(state: GeologyAgentState) -> Dict[str, Any]:
    """검토 노드"""
    print(f"\n{'='*60}")
    print("[Step 5/6] Reviewing generated cross-sections")
    print(f"{'='*60}")

    data_paths = state.get("data_paths", {})
    output_dir = data_paths.get("output_dir")

    result = review_sections(
        output_dir=output_dir,
        data_dir=data_paths.get("data_dir"),
        geology_dir=data_paths.get("geology_dir"),
        dem_file=data_paths.get("dem_file"),
        dem_dir=data_paths.get("dem_dir"),
        target_crs=data_paths.get("crs"),
        shapefiles=data_paths.get("shapefiles"),
        region_name_kr=data_paths.get("region_name_kr"),
        region_name_en=data_paths.get("region_name_en"),
    )

    review_results = None
    if result["success"]:
        review_results = {
            "total_reviewed": result.get("total_reviewed", 0),
            "average_score": result.get("average_score", 0.0),
            "reviews": result.get("reviews", []),
            "critical_issues": result.get("critical_issues", []),
            "html_report_path": result.get("html_report_path"),
        }

    return _sanitize_for_serialization({
        "review_completed": result["success"],
        "review_results": review_results,
        "current_step": "review",
        "error": result.get("error"),
    })


def finalize_node(state: GeologyAgentState) -> Dict[str, Any]:
    """최종 출력 생성 노드"""
    print(f"\n{'='*60}")
    print("[Step 6/6] Finalizing output")
    print(f"{'='*60}")

    region = state.get("region", "Unknown")
    data_paths = state.get("data_paths", {})
    section_images = state.get("section_images", {})
    report_path = state.get("report_path")
    review_results = state.get("review_results", {})
    sections_data = state.get("sections_data", {})

    # 출력 구성
    outputs = {
        "knowledge_base": state.get("knowledge_base_path"),
        "cross_sections": list(section_images.values()) if section_images else [],
        "report": report_path,
        "review_report": review_results.get("html_report_path") if review_results else None,
    }

    # 요약 생성
    num_sections = len(section_images) if section_images else 0
    avg_score = review_results.get("average_score", 0) if review_results else 0
    summary = f"{region} 지역 지질단면도 {num_sections}개 생성 완료. 평균 품질점수 {avg_score:.1f}/10"

    final_output = {
        "status": "completed",
        "region": region,
        "outputs": outputs,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'='*60}")
    print("WORKFLOW COMPLETE!")
    print(f"{'='*60}")
    print(f"  Region: {region}")
    print(f"  Sections generated: {num_sections}")
    print(f"  Average quality score: {avg_score:.1f}/10")
    print(f"  Report: {report_path}")

    return {
        "final_output": final_output,
        "current_step": "finalize",
    }


def error_node(state: GeologyAgentState) -> Dict[str, Any]:
    """에러 처리 노드"""
    error = state.get("error", "Unknown error")

    print(f"\n{'='*60}")
    print(f"ERROR: {error}")
    print(f"{'='*60}")

    return {
        "final_output": {
            "status": "error",
            "region": state.get("region", "Unknown"),
            "error": error,
            "timestamp": datetime.now().isoformat(),
        },
        "current_step": "error",
    }


# =============================================================================
# Routing Functions
# =============================================================================

def route_after_check_data(state: GeologyAgentState) -> Literal["extract_knowledge", "error"]:
    """데이터 확인 후 라우팅"""
    if state.get("data_available", False):
        return "extract_knowledge"
    return "error"


def route_after_knowledge(state: GeologyAgentState) -> Literal["analyze_sections", "error"]:
    """지식베이스 추출 후 라우팅"""
    if state.get("knowledge_extracted", False) or state.get("knowledge_base_path"):
        return "analyze_sections"
    return "error"


def route_after_analysis(state: GeologyAgentState) -> Literal["generate_visualization", "error"]:
    """단면 분석 후 라우팅"""
    if state.get("sections_analyzed", False):
        return "generate_visualization"
    return "error"


def route_after_visualization(state: GeologyAgentState) -> Literal["review", "finalize"]:
    """시각화 생성 후 라우팅 - 검토는 선택적"""
    if state.get("visualization_generated", False):
        return "review"
    return "finalize"


def route_after_review(state: GeologyAgentState) -> Literal["finalize"]:
    """검토 후 라우팅"""
    return "finalize"


# =============================================================================
# Workflow Creation
# =============================================================================

def create_workflow() -> StateGraph:
    """
    LangGraph 워크플로우 생성

    Returns:
        컴파일된 StateGraph
    """
    # StateGraph 생성
    workflow = StateGraph(GeologyAgentState)

    # 노드 추가
    workflow.add_node("check_data", check_data_node)
    workflow.add_node("extract_knowledge", extract_knowledge_node)
    workflow.add_node("analyze_sections", analyze_sections_node)
    workflow.add_node("generate_visualization", generate_visualization_node)
    workflow.add_node("review", review_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("error", error_node)

    # 시작점
    workflow.add_edge(START, "check_data")

    # 조건부 엣지
    workflow.add_conditional_edges(
        "check_data",
        route_after_check_data,
        {
            "extract_knowledge": "extract_knowledge",
            "error": "error",
        }
    )

    workflow.add_conditional_edges(
        "extract_knowledge",
        route_after_knowledge,
        {
            "analyze_sections": "analyze_sections",
            "error": "error",
        }
    )

    workflow.add_conditional_edges(
        "analyze_sections",
        route_after_analysis,
        {
            "generate_visualization": "generate_visualization",
            "error": "error",
        }
    )

    workflow.add_conditional_edges(
        "generate_visualization",
        route_after_visualization,
        {
            "review": "review",
            "finalize": "finalize",
        }
    )

    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {
            "finalize": "finalize",
        }
    )

    # 종료 엣지
    workflow.add_edge("finalize", END)
    workflow.add_edge("error", END)

    return workflow


def compile_workflow(checkpointer: Optional[Any] = None):
    """
    워크플로우 컴파일

    Args:
        checkpointer: 체크포인터 (기본값: MemorySaver)

    Returns:
        컴파일된 그래프
    """
    workflow = create_workflow()

    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


def run_workflow(
    region: str = "서울",
    task_id: Optional[str] = None,
    skip_knowledge_extraction: bool = False,
    skip_review: bool = False,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    워크플로우 실행

    Args:
        region: 분석할 지역명
        task_id: 태스크 ID (기본값: UUID 생성)
        skip_knowledge_extraction: 지식베이스 추출 스킵 여부
        skip_review: 검토 스킵 여부
        progress_callback: 각 노드 완료 시 호출되는 콜백 (step_name: str) -> None

    Returns:
        최종 결과
    """
    # 그래프 컴파일
    graph = compile_workflow()

    # 초기 상태
    initial_state = {
        "region": region,
        "task_id": task_id or str(uuid.uuid4()),
        "messages": [],
        "data_available": False,
        "knowledge_extracted": skip_knowledge_extraction,
        "sections_analyzed": False,
        "boundaries_analyzed": False,
        "visualization_generated": False,
        "review_completed": skip_review,
    }

    # 설정
    config = {
        "configurable": {
            "thread_id": initial_state["task_id"],
        }
    }

    # 실행
    print(f"\n{'#'*60}")
    print(f"# GEOLOGY CROSS-SECTION AGENT")
    print(f"# Region: {region}")
    print(f"# Task ID: {initial_state['task_id']}")
    print(f"{'#'*60}")

    # stream으로 실행하여 각 노드 완료 시 진행 상황 콜백
    final_state = None
    for chunk in graph.stream(initial_state, config):
        # chunk = {node_name: state_update}
        for node_name, state_update in chunk.items():
            step = state_update.get("current_step") if isinstance(state_update, dict) else None
            if step and progress_callback:
                progress_callback(step)
        final_state = chunk

    # 최종 상태에서 결과 추출
    # stream의 마지막 chunk에서 final_output 가져오기
    if final_state:
        for node_name, state_update in final_state.items():
            if isinstance(state_update, dict) and "final_output" in state_update:
                return state_update["final_output"]

    # fallback: 체크포인터에서 최종 상태 가져오기
    try:
        full_state = graph.get_state(config)
        if full_state and full_state.values:
            return full_state.values.get("final_output", {})
    except Exception:
        pass

    return {}


# 모듈 레벨에서 그래프 인스턴스 생성 (싱글톤)
_graph_instance = None


def get_graph():
    """싱글톤 그래프 인스턴스 반환"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = compile_workflow()
    return _graph_instance
