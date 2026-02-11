"""
Knowledge Extraction Tool
지식베이스 추출 도구 (knowledge_extractor_agent.py 래핑)
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional


def _add_parent_to_path():
    """부모 디렉토리를 Python 경로에 추가"""
    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))


def extract_knowledge(
    data_dir: str,
    output_dir: str,
    region_name: str = "seoul",
    max_pages: int = 60,
    use_vision: bool = False,
    vision_pages: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    지식베이스 추출 직접 호출

    Returns:
        {
            "success": bool,
            "knowledge_base_path": str or None,
            "ocr_text_path": str or None,
            "extraction_stats": dict,
            "error": str or None
        }
    """
    _add_parent_to_path()

    result = {
        "success": False,
        "knowledge_base_path": None,
        "ocr_text_path": None,
        "extraction_stats": {},
        "error": None,
    }

    try:
        # 기존 knowledge_extractor_agent 모듈 임포트
        from knowledge_extractor_agent import (
            HybridTextExtractor,
            KnowledgeExtractorAgent,
            VisionAnalyzer,
            find_geological_documents,
            load_api_config,
        )

        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 문서 찾기
        documents = find_geological_documents(data_path)
        total_docs = sum(len(v) for v in documents.values())

        if total_docs == 0:
            result["error"] = f"지질 문서를 찾을 수 없습니다: {data_dir}"
            return result

        # 텍스트 추출
        extractor = HybridTextExtractor(use_gpu=False)

        all_text = []
        total_stats = {
            'pymupdf_pages': 0,
            'tesseract_pages': 0,
            'paddleocr_pages': 0,
            'failed_pages': 0
        }

        for pdf_path in documents['pdf']:
            text, stats = extractor.extract_from_pdf(
                pdf_path, max_pages=max_pages, dpi=150
            )
            all_text.append(f"\n\n{'='*60}\nDocument: {pdf_path.name}\n{'='*60}\n")
            all_text.append(text)

            for key in total_stats:
                total_stats[key] += stats.get(key, 0)

        document_text = '\n'.join(all_text)
        result["extraction_stats"] = total_stats

        # OCR 텍스트 저장
        ocr_output_path = output_path / f"{region_name}_ocr_text.txt"
        with open(ocr_output_path, 'w', encoding='utf-8') as f:
            f.write(document_text)
        result["ocr_text_path"] = str(ocr_output_path)

        # Vision API 분석 (선택적)
        diagram_analysis = ""
        if use_vision and vision_pages:
            try:
                config = load_api_config(Path(config_path) if config_path else None)
                vision = VisionAnalyzer(
                    api_key=config['anthropic']['api_key'],
                    model=config.get('model', {}).get('name', 'claude-sonnet-4-20250514')
                )

                page_nums = [int(p.strip()) - 1 for p in vision_pages.split(',')]
                analysis_results = []

                for pdf_path in documents['pdf'][:1]:
                    for page_num in page_nums:
                        try:
                            img_bytes = extractor.extract_page_as_image(pdf_path, page_num, dpi=200)
                            analysis = vision.analyze_diagram(img_bytes)
                            analysis_results.append(f"### Page {page_num + 1} Diagram Analysis:\n{analysis}")
                        except Exception as e:
                            print(f"Vision analysis error on page {page_num}: {e}")

                diagram_analysis = "\n\n".join(analysis_results)
            except Exception as e:
                print(f"Vision API error: {e}")

        # LLM 지식 추출
        config_file = Path(config_path) if config_path else None
        agent = KnowledgeExtractorAgent(config_file)
        knowledge_code = agent.extract_knowledge(
            document_text,
            region_name,
            chart_data="",
            diagram_analysis=diagram_analysis
        )

        # 지식베이스 저장 (DataRepository 데이터 디렉토리에 저장)
        output_filename = f"{region_name}_geological_knowledge_auto.py"
        data_path = Path(data_dir)
        kb_output_path = data_path / output_filename

        with open(kb_output_path, 'w', encoding='utf-8') as f:
            f.write(knowledge_code)

        result["knowledge_base_path"] = str(kb_output_path)

        # 문법 검증
        try:
            compile(knowledge_code, output_filename, 'exec')
            result["success"] = True
        except SyntaxError as e:
            result["error"] = f"생성된 코드에 문법 오류가 있습니다: {e}"
            result["success"] = True  # 파일은 생성됨

    except Exception as e:
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    return result


# langchain 도구 버전 (선택적)
try:
    from langchain_core.tools import tool

    @tool
    def extract_knowledge_tool(
        data_dir: str,
        output_dir: str,
        region_name: str = "seoul",
        max_pages: int = 60,
        use_vision: bool = False,
        vision_pages: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        PDF 문서에서 지질 지식베이스를 추출합니다.

        Args:
            data_dir: 데이터 디렉토리 경로
            output_dir: 출력 디렉토리 경로
            region_name: 지역명 (출력 파일명에 사용)
            max_pages: 최대 처리 페이지 수
            use_vision: Vision API 사용 여부
            vision_pages: Vision 분석할 페이지 번호 (쉼표 구분)

        Returns:
            추출 결과 정보
        """
        return extract_knowledge(
            data_dir=data_dir,
            output_dir=output_dir,
            region_name=region_name,
            max_pages=max_pages,
            use_vision=use_vision,
            vision_pages=vision_pages,
        )

except ImportError:
    extract_knowledge_tool = None
