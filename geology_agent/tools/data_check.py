"""
Data Check Tool
데이터 존재 여부 확인 도구
"""

from typing import Dict, Any

from ..data_registry import check_region_data, get_base_dir


def check_data(region: str) -> Dict[str, Any]:
    """
    지역 데이터 존재 여부를 확인합니다.

    Args:
        region: 지역명 (예: "서울", "Seoul")

    Returns:
        데이터 가용성 정보
    """
    return check_region_data(region, get_base_dir())


# langchain 도구 버전 (선택적)
try:
    from langchain_core.tools import tool

    @tool
    def check_data_tool(region: str) -> Dict[str, Any]:
        """
        지역 데이터 존재 여부를 확인합니다.

        Args:
            region: 지역명 (예: "서울", "Seoul")

        Returns:
            데이터 가용성 정보
        """
        return check_data(region)

except ImportError:
    check_data_tool = None
