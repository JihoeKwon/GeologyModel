"""
Geology Agent State Definition
LangGraph 워크플로우 상태 정의
"""

from typing import TypedDict, Optional, List, Any

# Annotated는 typing에서, add_messages는 langgraph에서 가져옴
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

try:
    from langgraph.graph.message import add_messages
except ImportError:
    # langgraph가 없으면 더미 함수 사용
    def add_messages(left, right):
        """Fallback add_messages when langgraph is not installed"""
        if left is None:
            return right
        return left + right


class DataPaths(TypedDict, total=False):
    """데이터 경로 정보"""
    data_dir: str
    pdf_path: Optional[str]
    geology_dir: str
    dem_dir: str
    dem_file: str
    output_dir: str
    knowledge_base_path: Optional[str]


class SectionData(TypedDict, total=False):
    """단면 분석 데이터"""
    existing_sections: List[dict]
    auto_sections: List[dict]
    mean_strike: float
    optimal_azimuth: float
    comparison: dict


class ReviewResults(TypedDict, total=False):
    """검토 결과"""
    total_reviewed: int
    average_score: float
    reviews: List[dict]
    critical_issues: List[dict]


class FinalOutput(TypedDict, total=False):
    """최종 출력"""
    status: str
    region: str
    outputs: dict
    summary: str


class GeologyAgentState(TypedDict, total=False):
    """
    Geology Agent 워크플로우 상태

    각 노드에서 상태를 업데이트하며, 전체 파이프라인 진행 상황을 추적합니다.
    """
    # 메시지 히스토리 (LangGraph add_messages reducer 사용)
    messages: Annotated[list, add_messages]

    # 기본 정보
    region: str
    task_id: str

    # 데이터 확인 단계
    data_available: bool
    data_paths: DataPaths

    # 지식베이스 추출 단계
    knowledge_extracted: bool
    knowledge_base_path: Optional[str]

    # 단면 분석 단계
    sections_analyzed: bool
    sections_data: Optional[SectionData]

    # LLM 경계 분석 단계
    boundaries_analyzed: bool
    llm_results_path: Optional[str]

    # 시각화 생성 단계
    visualization_generated: bool
    section_images: Optional[dict]
    report_path: Optional[str]

    # 검토 단계
    review_completed: bool
    review_results: Optional[ReviewResults]

    # 최종 출력
    final_output: Optional[FinalOutput]

    # 에러 처리
    error: Optional[str]
    current_step: str
