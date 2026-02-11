"""
Geology Cross-Section Agent
LangGraph + FastAPI 기반 지질단면 자동화 에이전트

사용 예시:
    from geology_agent import create_app, run_workflow

    # FastAPI 앱 실행
    app = create_app()

    # 또는 워크플로우 직접 실행
    result = run_workflow(region="서울")
"""

__version__ = "0.1.0"

# 기본 모듈은 항상 로드 (의존성 없음)
from .data_registry import REGIONS, check_region_data

# 선택적 로드 (의존성 있음)
try:
    from .state import GeologyAgentState
except ImportError:
    GeologyAgentState = None

try:
    from .workflow import create_workflow, run_workflow
except ImportError:
    create_workflow = None
    run_workflow = None


def get_workflow_functions():
    """워크플로우 함수들을 지연 로드로 반환"""
    from .workflow import create_workflow, run_workflow
    return create_workflow, run_workflow


__all__ = [
    "create_workflow",
    "run_workflow",
    "REGIONS",
    "check_region_data",
    "GeologyAgentState",
    "get_workflow_functions",
]
