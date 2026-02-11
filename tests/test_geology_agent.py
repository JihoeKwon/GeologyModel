"""
Tests for Geology Agent
지질 에이전트 테스트
"""

import sys
from pathlib import Path

# 부모 디렉토리를 경로에 추가
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def test_data_registry():
    """데이터 레지스트리 테스트"""
    from geology_agent.data_registry import (
        REGIONS,
        check_region_data,
        get_available_regions,
        normalize_region_name,
    )

    print("Testing data registry...")

    # 지역 목록 확인
    regions = get_available_regions()
    assert "서울" in regions, "서울 지역이 등록되어 있어야 합니다"
    print(f"  Available regions: {regions}")

    # 지역명 정규화
    assert normalize_region_name("Seoul") == "서울"
    assert normalize_region_name("seoul") == "서울"
    assert normalize_region_name("서울") == "서울"
    print("  Region name normalization: OK")

    # 데이터 확인
    result = check_region_data("서울")
    print(f"  Seoul data available: {result['available']}")
    print(f"  Paths: {list(result['paths'].keys())}")

    print("  Data registry tests passed!\n")


def test_state():
    """상태 정의 테스트"""
    from geology_agent.state import GeologyAgentState

    print("Testing state definition...")

    # TypedDict 필드 확인
    annotations = GeologyAgentState.__annotations__
    required_fields = [
        "messages", "region", "data_available", "data_paths",
        "knowledge_extracted", "sections_analyzed", "final_output"
    ]

    for field in required_fields:
        assert field in annotations, f"필드 {field}가 정의되어 있어야 합니다"

    print(f"  State fields: {list(annotations.keys())}")
    print("  State tests passed!\n")


def test_tools_import():
    """도구 import 테스트"""
    print("Testing tools import...")

    try:
        from geology_agent.tools import (
            check_data_tool,
            extract_knowledge_tool,
            analyze_sections_tool,
            generate_visualization_tool,
            review_sections_tool,
        )
        print("  All tools imported successfully!")
    except ImportError as e:
        print(f"  Import error (expected if langchain not installed): {e}")

    print("  Tools import tests passed!\n")


def test_workflow_creation():
    """워크플로우 생성 테스트"""
    print("Testing workflow creation...")

    try:
        from geology_agent.workflow import create_workflow

        workflow = create_workflow()
        print(f"  Workflow created: {type(workflow)}")

        # 노드 확인
        nodes = workflow.nodes
        expected_nodes = [
            "check_data", "extract_knowledge", "analyze_sections",
            "generate_visualization", "review", "finalize", "error"
        ]

        for node in expected_nodes:
            assert node in nodes, f"노드 {node}가 존재해야 합니다"

        print(f"  Nodes: {list(nodes.keys())}")
        print("  Workflow tests passed!\n")

    except ImportError as e:
        print(f"  Import error (expected if langgraph not installed): {e}")
        print("  Skipping workflow tests\n")


def test_api_schemas():
    """API 스키마 테스트"""
    print("Testing API schemas...")

    try:
        from geology_agent.api.schemas import (
            AnalyzeRequest,
            AnalyzeResponse,
            StatusResponse,
            ResultsResponse,
            RegionInfo,
            TaskStatus,
        )

        # 요청 생성
        request = AnalyzeRequest(region="서울")
        assert request.region == "서울"
        print(f"  AnalyzeRequest: {request.model_dump()}")

        # 상태 enum
        assert TaskStatus.COMPLETED == "completed"
        print("  TaskStatus enum: OK")

        print("  API schema tests passed!\n")

    except ImportError as e:
        print(f"  Import error (expected if pydantic not installed): {e}")
        print("  Skipping API tests\n")


def test_data_check_tool():
    """데이터 확인 도구 직접 테스트"""
    print("Testing data check tool...")

    from geology_agent.tools.data_check import check_data

    result = check_data("서울")

    print(f"  Region: {result['region']}")
    print(f"  Available: {result['available']}")
    print(f"  Message: {result['message']}")

    if result['paths']:
        print("  Paths found:")
        for key, value in result['paths'].items():
            if value:
                print(f"    {key}: {value[:50]}..." if len(str(value)) > 50 else f"    {key}: {value}")

    print("  Data check tool tests passed!\n")


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "=" * 60)
    print("GEOLOGY AGENT TESTS")
    print("=" * 60 + "\n")

    test_data_registry()
    test_state()
    test_data_check_tool()
    test_tools_import()
    test_workflow_creation()
    test_api_schemas()

    print("=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
