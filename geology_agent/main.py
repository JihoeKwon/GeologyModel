"""
Geology Cross-Section Agent - Main Entry Point
지질단면 분석 에이전트 메인 실행 파일

Usage:
    # CLI 실행
    python -m geology_agent.main --region "서울"

    # 또는 모듈로 import
    from geology_agent import run_workflow
    result = run_workflow(region="서울")
"""

import argparse
import sys
from pathlib import Path

# 부모 디렉토리를 경로에 추가 (패키지 외부에서 실행 시)
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Geology Cross-Section Agent - 지질단면 자동 분석',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 실행 (서울 지역)
  python -m geology_agent.main --region "서울"

  # 지식베이스 추출 포함
  python -m geology_agent.main --region "서울" --extract-knowledge

  # 검토 스킵
  python -m geology_agent.main --region "서울" --skip-review

  # API 서버 실행
  python -m geology_agent.main --server

  # 사용 가능한 지역 확인
  python -m geology_agent.main --list-regions
        """
    )

    parser.add_argument(
        '--region', '-r',
        type=str,
        default="서울",
        help='분석할 지역명 (기본값: 서울)'
    )

    parser.add_argument(
        '--extract-knowledge',
        action='store_true',
        help='지식베이스를 새로 추출 (기본: 기존 지식베이스 사용)'
    )

    parser.add_argument(
        '--skip-review',
        action='store_true',
        help='검토 단계 스킵'
    )

    parser.add_argument(
        '--server',
        action='store_true',
        help='FastAPI 서버 실행'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='서버 호스트 (기본값: 0.0.0.0)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='서버 포트 (기본값: 8000)'
    )

    parser.add_argument(
        '--list-regions',
        action='store_true',
        help='사용 가능한 지역 목록 출력'
    )

    parser.add_argument(
        '--check-data',
        action='store_true',
        help='지역 데이터 존재 여부만 확인'
    )

    return parser.parse_args()


def list_regions():
    """사용 가능한 지역 목록 출력"""
    from .data_registry import REGIONS, check_region_data, get_base_dir

    print("\n" + "=" * 60)
    print("사용 가능한 지역 목록")
    print("=" * 60)

    for region_name, region_config in REGIONS.items():
        data_check = check_region_data(region_name, get_base_dir())

        status = "[O] Available" if data_check["available"] else "[X] No data"
        pdf_status = "[PDF]" if data_check["paths"].get("pdf_path") else "     "
        kb_status = "[KB]" if data_check["paths"].get("knowledge_base_path") else "    "

        print(f"\n{region_name} ({region_config.get('name_en', '')})")
        print(f"  상태: {status}")
        print(f"  {pdf_status} PDF 도폭설명서")
        print(f"  {kb_status} 지식베이스")

        if not data_check["available"] and data_check["missing"]:
            print(f"  누락: {', '.join(data_check['missing'][:3])}")


def check_data(region: str):
    """데이터 존재 여부 확인"""
    from .data_registry import check_region_data, get_base_dir

    print(f"\n지역 '{region}' 데이터 확인 중...")

    result = check_region_data(region, get_base_dir())

    print("\n" + "=" * 60)
    print(f"지역: {region}")
    print("=" * 60)

    if result["available"]:
        print("Status: [OK] Available\n")
        print("Paths:")
        for key, value in result["paths"].items():
            if value:
                print(f"  {key}: {value}")
    else:
        print("Status: [X] Not available\n")
        print(f"메시지: {result['message']}\n")

        if result["missing"]:
            print("누락된 항목:")
            for item in result["missing"]:
                print(f"  - {item}")


def run_server(host: str, port: int):
    """FastAPI 서버 실행"""
    import uvicorn

    print("\n" + "=" * 60)
    print("Geology Cross-Section Agent API Server")
    print("=" * 60)
    print(f"\n서버 시작: http://{host}:{port}")
    print(f"API 문서: http://{host}:{port}/docs")
    print("\n종료하려면 Ctrl+C를 누르세요.\n")

    uvicorn.run(
        "geology_agent.api.server:app",
        host=host,
        port=port,
        reload=False,  # Disabled to avoid multiprocessing issues on Windows
    )


def run_workflow_cli(region: str, skip_knowledge: bool, skip_review: bool):
    """워크플로우 CLI 실행"""
    from .workflow import run_workflow

    print("\n" + "#" * 60)
    print("# GEOLOGY CROSS-SECTION AGENT")
    print(f"# Region: {region}")
    print(f"# Skip Knowledge Extraction: {skip_knowledge}")
    print(f"# Skip Review: {skip_review}")
    print("#" * 60)

    result = run_workflow(
        region=region,
        skip_knowledge_extraction=skip_knowledge,
        skip_review=skip_review,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    if result.get("status") == "completed":
        print(f"\n[OK] Completed: {result.get('summary', '')}\n")

        outputs = result.get("outputs", {})
        if outputs:
            print("생성된 파일:")
            if outputs.get("knowledge_base"):
                print(f"  - 지식베이스: {outputs['knowledge_base']}")
            if outputs.get("report"):
                print(f"  - HTML 보고서: {outputs['report']}")
            if outputs.get("cross_sections"):
                print(f"  - 단면도 {len(outputs['cross_sections'])}개:")
                for img in outputs["cross_sections"][:5]:
                    print(f"      {img}")
                if len(outputs["cross_sections"]) > 5:
                    print(f"      ... 외 {len(outputs['cross_sections']) - 5}개")
    else:
        print(f"\n[FAILED]: {result.get('error', 'Unknown error')}")

    return result


def main():
    """메인 함수"""
    args = parse_args()

    if args.list_regions:
        list_regions()
        return

    if args.check_data:
        check_data(args.region)
        return

    if args.server:
        run_server(args.host, args.port)
        return

    # 워크플로우 실행
    run_workflow_cli(
        region=args.region,
        skip_knowledge=not args.extract_knowledge,
        skip_review=args.skip_review,
    )


if __name__ == "__main__":
    main()
