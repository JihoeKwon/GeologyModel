"""
Geology Agent Tools
기존 스크립트를 래핑한 LangGraph 도구들
"""

# 직접 함수들은 항상 사용 가능
from .data_check import check_data
from .knowledge import extract_knowledge
from .analysis import analyze_sections
from .visualization import generate_visualization
from .review import review_sections

# langchain 도구들은 선택적 로드
try:
    from .data_check import check_data_tool
    from .knowledge import extract_knowledge_tool
    from .analysis import analyze_sections_tool
    from .visualization import generate_visualization_tool
    from .review import review_sections_tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    check_data_tool = None
    extract_knowledge_tool = None
    analyze_sections_tool = None
    generate_visualization_tool = None
    review_sections_tool = None
    LANGCHAIN_AVAILABLE = False

__all__ = [
    # 직접 함수
    "check_data",
    "extract_knowledge",
    "analyze_sections",
    "generate_visualization",
    "review_sections",
    # langchain 도구
    "check_data_tool",
    "extract_knowledge_tool",
    "analyze_sections_tool",
    "generate_visualization_tool",
    "review_sections_tool",
    "LANGCHAIN_AVAILABLE",
]
