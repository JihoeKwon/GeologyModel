"""
Geological Knowledge Base Extractor Agent (Hybrid Multi-OCR + Vision)
지질도폭 설명서에서 지식베이스를 자동 추출하는 에이전트

Hybrid approach:
1. Text Extraction: PyMuPDF → Tesseract → PaddleOCR (3-stage fallback)
2. Chart Extraction: DePlot (local, free)
3. Diagram Analysis: Claude Vision API (geological interpretation)
4. Knowledge Generation: Claude API

Usage:
    python knowledge_extractor_agent.py --data-dir "D:/Data/SeoulGeology"
    python knowledge_extractor_agent.py --data-dir "D:/Data/BusanGeology" --output-name "busan"
    python knowledge_extractor_agent.py --data-dir "D:/Data" --use-vision --vision-pages 1,5,10
    python knowledge_extractor_agent.py --data-dir "D:/Data" --use-deplot --chart-pages 3,7
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import json
import re
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import anthropic
import yaml
import gc

# =============================================================================
# Dependency Check
# =============================================================================

# PyMuPDF (required)
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. Install with: pip install pymupdf")

# Tesseract OCR
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
    # Windows에서 Tesseract 경로 설정
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
            Path.home().name
        ),
    ]
    for path in tesseract_paths:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            break
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. Install with: pip install pytesseract pillow")

# PaddleOCR (optional, for complex layouts)
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("Info: PaddleOCR not installed (optional). Install with: pip install paddlepaddle paddleocr")

# DePlot (optional, for chart extraction)
try:
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
    DEPLOT_AVAILABLE = True
except ImportError:
    DEPLOT_AVAILABLE = False
    print("Info: DePlot not installed (optional). Install with: pip install transformers")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_api_config(config_path: Path = None) -> dict:
    """Load API configuration from config.yaml"""
    config_path = config_path or DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


# =============================================================================
# Page Classification
# =============================================================================

class PageType(Enum):
    TEXT = "text"
    CHART = "chart"
    DIAGRAM = "diagram"
    MIXED = "mixed"


@dataclass
class PageInfo:
    page_num: int
    page_type: PageType
    has_text: bool
    has_images: bool
    text_density: float  # 0.0 ~ 1.0


def classify_page(page, page_num: int) -> PageInfo:
    """Classify a PDF page by its content type"""
    text = page.get_text().strip()
    images = page.get_images()

    text_len = len(text)
    has_text = text_len > 50
    has_images = len(images) > 0

    # Calculate text density (characters per page area)
    rect = page.rect
    page_area = rect.width * rect.height
    text_density = min(text_len / max(page_area, 1) * 1000, 1.0)

    # Determine page type
    if text_density > 0.3 and not has_images:
        page_type = PageType.TEXT
    elif has_images and text_density < 0.1:
        # Could be chart or diagram - will need further analysis
        page_type = PageType.DIAGRAM
    elif has_images and text_density >= 0.1:
        page_type = PageType.MIXED
    else:
        page_type = PageType.TEXT

    return PageInfo(
        page_num=page_num,
        page_type=page_type,
        has_text=has_text,
        has_images=has_images,
        text_density=text_density
    )


# =============================================================================
# Document Finder
# =============================================================================

def find_geological_documents(data_dir: Path) -> Dict[str, List[Path]]:
    """Find geological explanation documents in the data directory"""
    documents = {
        'pdf': [],
        'hwp': [],
        'txt': []
    }

    keywords = ['설명서', '도폭', 'explanation', 'geological', '지질']

    for pdf_file in data_dir.rglob('*.pdf'):
        filename_lower = pdf_file.name.lower()
        if any(kw in filename_lower or kw in str(pdf_file).lower() for kw in keywords):
            documents['pdf'].append(pdf_file)
        elif '지질' in pdf_file.name or 'geolog' in filename_lower:
            documents['pdf'].append(pdf_file)

    for hwp_file in data_dir.rglob('*.hwp'):
        filename_lower = hwp_file.name.lower()
        if any(kw in filename_lower or kw in str(hwp_file).lower() for kw in keywords):
            documents['hwp'].append(hwp_file)

    for txt_file in data_dir.rglob('*.txt'):
        filename_lower = txt_file.name.lower()
        if any(kw in filename_lower or kw in str(txt_file).lower() for kw in keywords):
            documents['txt'].append(txt_file)

    return documents


# =============================================================================
# Text Extraction (3-Stage Fallback)
# =============================================================================

class HybridTextExtractor:
    """
    Extract text from PDFs using 3-stage fallback:
    1. PyMuPDF (direct text extraction)
    2. Tesseract OCR (lightweight, fast)
    3. PaddleOCR (accurate, for complex layouts)
    """

    def __init__(self, use_gpu: bool = False):
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF is required but not installed")

        self.use_gpu = use_gpu
        self.paddle_ocr = None
        self.min_text_threshold = 100  # Minimum characters to consider extraction successful
        self.ocr_confidence_threshold = 0.5

        print("  Text Extractor initialized")
        print(f"    - PyMuPDF: Available")
        print(f"    - Tesseract: {'Available' if TESSERACT_AVAILABLE else 'Not available'}")
        print(f"    - PaddleOCR: {'Available' if PADDLEOCR_AVAILABLE else 'Not available'}")

    def _init_paddle_ocr(self):
        """Lazy initialization of PaddleOCR"""
        if self.paddle_ocr is None and PADDLEOCR_AVAILABLE:
            print("    Initializing PaddleOCR...")
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=self.use_gpu)
            print("    PaddleOCR ready!")

    def extract_from_pdf(self, pdf_path: Path, max_pages: int = 100, dpi: int = 150) -> Tuple[str, Dict]:
        """
        Extract text from PDF with 3-stage fallback
        Returns: (extracted_text, extraction_stats)
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages)

        print(f"    Processing {pages_to_process}/{total_pages} pages...")

        all_text = []
        stats = {
            'total_pages': pages_to_process,
            'pymupdf_pages': 0,
            'tesseract_pages': 0,
            'paddleocr_pages': 0,
            'failed_pages': 0
        }

        for i in range(pages_to_process):
            page = doc[i]
            page_text = ""
            extraction_method = None

            # Stage 1: Try PyMuPDF direct extraction
            page_text = page.get_text().strip()
            if len(page_text) >= self.min_text_threshold:
                extraction_method = "pymupdf"
                stats['pymupdf_pages'] += 1
            else:
                # Need OCR - convert page to image
                pix = page.get_pixmap(dpi=dpi)
                img_bytes = pix.tobytes("png")
                del pix
                gc.collect()

                # Save temp image
                temp_path = pdf_path.parent / f"_temp_ocr_page_{i}.png"
                try:
                    with open(temp_path, 'wb') as f:
                        f.write(img_bytes)
                    del img_bytes

                    # Stage 2: Try Tesseract
                    if TESSERACT_AVAILABLE:
                        try:
                            img = Image.open(temp_path)
                            page_text = pytesseract.image_to_string(img, lang='kor+eng')
                            img.close()

                            if len(page_text.strip()) >= self.min_text_threshold:
                                extraction_method = "tesseract"
                                stats['tesseract_pages'] += 1
                        except Exception as e:
                            print(f"      Page {i+1} Tesseract error: {e}")

                    # Stage 3: Try PaddleOCR (if Tesseract failed or unavailable)
                    if extraction_method is None and PADDLEOCR_AVAILABLE:
                        try:
                            self._init_paddle_ocr()
                            result = self.paddle_ocr.ocr(str(temp_path), cls=True)

                            if result and result[0]:
                                texts = []
                                for line in result[0]:
                                    if line[1][1] > self.ocr_confidence_threshold:
                                        texts.append(line[1][0])
                                page_text = ' '.join(texts)

                                if len(page_text.strip()) >= self.min_text_threshold // 2:
                                    extraction_method = "paddleocr"
                                    stats['paddleocr_pages'] += 1
                        except Exception as e:
                            print(f"      Page {i+1} PaddleOCR error: {e}")

                finally:
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()

            # Record result
            if extraction_method:
                all_text.append(f"\n--- Page {i+1} [{extraction_method}] ---\n")
                all_text.append(page_text)
            else:
                stats['failed_pages'] += 1
                all_text.append(f"\n--- Page {i+1} [FAILED] ---\n")

            # Progress and memory management
            if (i + 1) % 5 == 0:
                gc.collect()
            if (i + 1) % 10 == 0:
                print(f"      Processed {i+1}/{pages_to_process} pages...")

        doc.close()
        return '\n'.join(all_text), stats

    def extract_page_as_image(self, pdf_path: Path, page_num: int, dpi: int = 200) -> bytes:
        """Extract a single page as image bytes"""
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            doc.close()
            raise ValueError(f"Page {page_num} does not exist (total: {len(doc)})")

        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        del pix
        doc.close()
        return img_bytes


# =============================================================================
# Chart Extraction (DePlot)
# =============================================================================

class ChartExtractor:
    """Extract data from charts using DePlot"""

    def __init__(self):
        if not DEPLOT_AVAILABLE:
            raise RuntimeError("DePlot (transformers) not installed")

        print("  Initializing DePlot...")
        self.processor = Pix2StructProcessor.from_pretrained('google/deplot')
        self.model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
        print("  DePlot ready!")

    def extract_chart_data(self, image_bytes: bytes) -> str:
        """Extract tabular data from chart image"""
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        try:
            image = Image.open(temp_path)
            inputs = self.processor(
                images=image,
                text="Generate underlying data table of the figure below:",
                return_tensors="pt"
            )

            predictions = self.model.generate(**inputs, max_new_tokens=512)
            table_text = self.processor.decode(predictions[0], skip_special_tokens=True)

            image.close()
            return table_text
        finally:
            Path(temp_path).unlink()

    def parse_table(self, table_text: str) -> Dict[str, List]:
        """Parse DePlot output to structured data"""
        lines = table_text.strip().split('\n')
        if not lines:
            return {}

        # DePlot outputs: "header1 | header2 | ... <0x0A> val1 | val2 | ..."
        # or sometimes: "col1 | col2 \n val1 | val2"
        result = {'raw': table_text, 'data': []}

        for line in lines:
            if '|' in line:
                values = [v.strip() for v in line.split('|')]
                result['data'].append(values)

        return result


# =============================================================================
# Vision API for Geological Diagrams
# =============================================================================

DIAGRAM_ANALYSIS_PROMPT = """이 지질 다이어그램을 분석하세요. 다음 정보를 추출해주세요:

1. **암석 단위 (Rock Units)**:
   - 기호/범례에 표시된 암석 종류
   - 각 암석의 색상 또는 패턴
   - 한글/영문 명칭

2. **지층 순서 (Stratigraphic Order)**:
   - 상부에서 하부로의 지층 순서
   - 각 지층의 대략적 두께

3. **구조 (Structures)**:
   - 단층 (fault): 종류, 방향
   - 습곡 (fold): 종류, 축 방향
   - 관입 관계

4. **접촉 관계 (Contact Relationships)**:
   - 정합/부정합/관입 접촉
   - 접촉면의 특징

5. **경사/주향 (Dip/Strike)**:
   - 표시된 경사 방향과 각도
   - 주향선 방향

JSON 형식으로 출력해주세요."""


class VisionAnalyzer:
    """Analyze geological diagrams using Claude Vision API"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def analyze_diagram(self, image_bytes: bytes, prompt: str = None) -> str:
        """Analyze a geological diagram using Vision API"""
        prompt = prompt or DIAGRAM_ANALYSIS_PROMPT
        image_b64 = base64.standard_b64encode(image_bytes).decode('utf-8')

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        return response.content[0].text


# =============================================================================
# LLM Knowledge Extraction
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are a geological knowledge extraction specialist. Your task is to extract structured geological information from geological map explanation documents (지질도폭 설명서) and convert it into a Python knowledge base format.

## Your Expertise
- Korean Peninsula geology and stratigraphy
- Structural geology terminology in Korean and English
- Rock unit classification and nomenclature
- Contact relationships between rock units
- Metamorphic and igneous petrology

## Output Format
You must output valid Python code that defines the following data structures:

1. **ROCK_UNITS**: Dictionary with rock unit codes as keys
   - name_kr: Korean name
   - name_en: English name
   - age: Geological age
   - description: Brief description from the document
   - lithology: Rock characteristics (texture, composition, etc.)
   - structure: Structural characteristics (foliation, etc.)
   - expected_dip_range: Typical dip angle range (e.g., "40-70°")
   - expected_contact_type: "conformable", "intrusive", or "unconformable"

2. **STRUCTURAL_GEOLOGY**: Regional structural information
   - regional_setting: Location, dominant trends
   - foliation: Typical orientations with numerical values
   - folding: Fold phases and characteristics
   - faulting: Fault types and orientations

3. **CONTACT_RULES**: Lists of rock unit pairs
   - conformable_pairs: Units with conformable contacts
   - intrusive_pairs: Units with intrusive contacts
   - unconformable_pairs: Units with unconformable contacts

4. **DIP_ESTIMATION_RULES**: Dip angle estimation guidelines

5. **CHART_DATA**: Extracted chart/table data (if available)

6. Helper functions:
   - get_contact_type(rock1, rock2)
   - get_expected_dip_range(rock1, rock2)
   - generate_context_for_boundary(rock_codes)
   - generate_regional_context()

## Important Guidelines
- Extract LITHOIDX codes from the document (e.g., PCEbngn, Jbgr, Qa)
- Pay attention to Korean rock names and their codes
- Include both Korean and English terminology
- Preserve numerical data (dip angles, strikes, ages in Ma)
- Extract specific measurements mentioned in the text
- Output ONLY valid Python code, no markdown formatting
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """Based on the following geological map explanation document, create a comprehensive knowledge base.

Note: The text was extracted using OCR, so there may be some recognition errors. Please use your geological expertise to interpret the content correctly.

## Source Document Text:
{document_text}

## Chart Data (extracted from figures):
{chart_data}

## Diagram Analysis (from geological diagrams):
{diagram_analysis}

## Instructions:
1. Identify all rock units mentioned (look for patterns like "호상흑운모편마암", "흑운모화강암", etc.)
2. Extract structural geology information (foliation strikes/dips, fold phases, faulting)
3. Determine contact relationships between rock units
4. Extract typical dip angles and directions mentioned in the text
5. Note any numerical data (ages, measurements, etc.)
6. Incorporate chart data and diagram analysis if provided

## Required Output:
Generate a complete Python file with:
- ROCK_UNITS dictionary with detailed information for each rock type
- STRUCTURAL_GEOLOGY dictionary with regional structure info
- CONTACT_RULES dictionary with rock unit pairs
- DIP_ESTIMATION_RULES dictionary
- CHART_DATA dictionary (if chart data was provided)
- Helper functions (get_contact_type, get_expected_dip_range, etc.)

Start with the module docstring. Make it a complete, runnable Python file.
"""


class KnowledgeExtractorAgent:
    """Agent to extract geological knowledge from documents using LLM"""

    def __init__(self, config_path: Path = None):
        config = load_api_config(config_path)

        self.api_key = config.get('anthropic', {}).get('api_key')
        self.model = config.get('model', {}).get('name', 'claude-sonnet-4-20250514')
        self.max_tokens = 8192

        if not self.api_key:
            raise ValueError("Anthropic API key not found in config")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        print(f"  Knowledge Extractor Agent initialized (Model: {self.model})")

    def extract_knowledge(self, document_text: str, region_name: str,
                         chart_data: str = "", diagram_analysis: str = "") -> str:
        """Extract knowledge base from document text"""
        max_chars = 150000
        if len(document_text) > max_chars:
            print(f"  Warning: Document text truncated from {len(document_text)} to {max_chars} chars")
            document_text = document_text[:max_chars]

        user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
            document_text=document_text,
            chart_data=chart_data or "No chart data extracted.",
            diagram_analysis=diagram_analysis or "No diagram analysis available."
        )

        print(f"  Sending to LLM ({len(document_text):,} chars)...")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )

        response_text = response.content[0].text
        response_text = self._clean_python_code(response_text)

        header = f'''"""
{region_name.title()} Geological Map Knowledge Base
{region_name.title()} 지질도폭 설명서 기반 지질 지식 베이스

Auto-generated by knowledge_extractor_agent.py (Hybrid Multi-OCR + Vision)
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Extraction Methods Used:
- Text: PyMuPDF / Tesseract / PaddleOCR (3-stage fallback)
- Charts: DePlot (if enabled)
- Diagrams: Claude Vision API (if enabled)
"""

'''
        return header + response_text

    def _clean_python_code(self, text: str) -> str:
        """Remove markdown code blocks"""
        text = re.sub(r'^```python\s*\n', '', text)
        text = re.sub(r'^```\s*\n', '', text)
        text = re.sub(r'\n```\s*$', '', text)
        text = re.sub(r'^```\s*', '', text)
        return text.strip()


# =============================================================================
# Main Execution
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Extract geological knowledge base from map explanation documents (Hybrid Multi-OCR + Vision)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction (PyMuPDF → Tesseract → PaddleOCR)
  python knowledge_extractor_agent.py --data-dir "D:/Data/SeoulGeology"

  # With chart extraction using DePlot
  python knowledge_extractor_agent.py --data-dir "D:/Data" --use-deplot --chart-pages 3,7,15

  # With diagram analysis using Vision API
  python knowledge_extractor_agent.py --data-dir "D:/Data" --use-vision --vision-pages 1,5,10

  # Full hybrid (all features)
  python knowledge_extractor_agent.py --data-dir "D:/Data" --use-deplot --chart-pages 3,7 --use-vision --vision-pages 1,5
        """
    )

    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing geological data and explanation documents')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: data-dir)')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Region name for output filename')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml')
    parser.add_argument('--max-pages', type=int, default=60,
                       help='Maximum PDF pages to process (default: 60)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for OCR (default: 150)')

    # DePlot options
    parser.add_argument('--use-deplot', action='store_true',
                       help='Use DePlot for chart data extraction (free, local)')
    parser.add_argument('--chart-pages', type=str, default=None,
                       help='Comma-separated page numbers containing charts (e.g., "3,7,15")')

    # Vision API options
    parser.add_argument('--use-vision', action='store_true',
                       help='Use Claude Vision API for diagram analysis (additional cost)')
    parser.add_argument('--vision-pages', type=str, default=None,
                       help='Comma-separated page numbers containing diagrams (e.g., "1,5,10")')

    parser.add_argument('--dry-run', action='store_true',
                       help='Find documents but do not extract')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU for OCR/DePlot (if available)')

    return parser.parse_args()


def infer_region_name(data_dir: Path) -> str:
    """Infer region name from directory path"""
    dir_name = data_dir.name.lower()

    regions = {
        'seoul': '서울', 'busan': '부산', 'daegu': '대구',
        'incheon': '인천', 'gwangju': '광주', 'daejeon': '대전',
        'ulsan': '울산', 'sejong': '세종', 'gyeonggi': '경기',
        'gangwon': '강원', 'chungbuk': '충북', 'chungnam': '충남',
        'jeonbuk': '전북', 'jeonnam': '전남', 'gyeongbuk': '경북',
        'gyeongnam': '경남', 'jeju': '제주'
    }

    for eng, kor in regions.items():
        if eng in dir_name or kor in dir_name:
            return eng

    for pdf in data_dir.rglob('*.pdf'):
        for eng, kor in regions.items():
            if kor in pdf.name:
                return eng

    return re.sub(r'[^a-zA-Z0-9]', '_', dir_name).lower()


def main():
    print("\n" + "=" * 70)
    print("  Geological Knowledge Base Extractor (Hybrid Multi-OCR + Vision)")
    print("  지질 지식베이스 자동 추출 에이전트")
    print("=" * 70)

    args = parse_arguments()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config) if args.config else None
    region_name = args.output_name or infer_region_name(data_dir)

    print(f"\n  Configuration:")
    print(f"    Data directory: {data_dir}")
    print(f"    Region name: {region_name}")
    print(f"    Output directory: {output_dir}")
    print(f"    Max pages: {args.max_pages}")
    print(f"    DPI: {args.dpi}")
    print(f"    Use DePlot: {args.use_deplot}")
    print(f"    Use Vision API: {args.use_vision}")

    # ==========================================================================
    # Step 1: Find documents
    # ==========================================================================
    print("\n" + "-" * 50)
    print("1. Finding geological documents...")
    print("-" * 50)

    documents = find_geological_documents(data_dir)
    total_docs = sum(len(v) for v in documents.values())

    print(f"\n  Found {total_docs} document(s):")
    for doc_type, files in documents.items():
        for f in files:
            print(f"    [{doc_type.upper()}] {f.name}")

    if total_docs == 0:
        print("\n  No geological documents found!")
        sys.exit(1)

    if args.dry_run:
        print("\n  [Dry run - stopping before extraction]")
        return

    # ==========================================================================
    # Step 2: Text extraction (3-stage fallback)
    # ==========================================================================
    print("\n" + "-" * 50)
    print("2. Extracting text (PyMuPDF → Tesseract → PaddleOCR)...")
    print("-" * 50)

    extractor = HybridTextExtractor(use_gpu=args.use_gpu)

    all_text = []
    total_stats = {'pymupdf_pages': 0, 'tesseract_pages': 0, 'paddleocr_pages': 0, 'failed_pages': 0}

    for pdf_path in documents['pdf']:
        print(f"\n  Processing: {pdf_path.name}")
        text, stats = extractor.extract_from_pdf(pdf_path, max_pages=args.max_pages, dpi=args.dpi)
        all_text.append(f"\n\n{'='*60}\nDocument: {pdf_path.name}\n{'='*60}\n")
        all_text.append(text)

        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    document_text = '\n'.join(all_text)

    print(f"\n  Extraction Statistics:")
    print(f"    - PyMuPDF (direct): {total_stats['pymupdf_pages']} pages")
    print(f"    - Tesseract OCR: {total_stats['tesseract_pages']} pages")
    print(f"    - PaddleOCR: {total_stats['paddleocr_pages']} pages")
    print(f"    - Failed: {total_stats['failed_pages']} pages")
    print(f"    - Total text: {len(document_text):,} characters")

    # Save OCR result for inspection
    ocr_output_path = output_dir / f"{region_name}_ocr_text.txt"
    with open(ocr_output_path, 'w', encoding='utf-8') as f:
        f.write(document_text)
    print(f"  Text saved: {ocr_output_path}")

    # ==========================================================================
    # Step 3: Chart extraction with DePlot (optional)
    # ==========================================================================
    chart_data = ""
    if args.use_deplot and args.chart_pages:
        print("\n" + "-" * 50)
        print("3. Extracting chart data with DePlot...")
        print("-" * 50)

        if not DEPLOT_AVAILABLE:
            print("  Warning: DePlot not available, skipping chart extraction")
        else:
            try:
                chart_extractor = ChartExtractor()
                page_nums = [int(p.strip()) - 1 for p in args.chart_pages.split(',')]

                chart_results = []
                for pdf_path in documents['pdf'][:1]:
                    for page_num in page_nums:
                        print(f"  Extracting chart from page {page_num + 1}...")
                        try:
                            img_bytes = extractor.extract_page_as_image(pdf_path, page_num, dpi=200)
                            table_text = chart_extractor.extract_chart_data(img_bytes)
                            chart_results.append(f"### Page {page_num + 1} Chart Data:\n{table_text}")
                            print(f"    Extracted: {len(table_text)} chars")
                        except Exception as e:
                            print(f"    Error: {e}")

                chart_data = "\n\n".join(chart_results)
            except Exception as e:
                print(f"  DePlot initialization error: {e}")

    # ==========================================================================
    # Step 4: Diagram analysis with Vision API (optional)
    # ==========================================================================
    diagram_analysis = ""
    if args.use_vision and args.vision_pages:
        print("\n" + "-" * 50)
        print("4. Analyzing diagrams with Claude Vision API...")
        print("-" * 50)

        config = load_api_config(config_path)
        vision = VisionAnalyzer(
            api_key=config['anthropic']['api_key'],
            model=config.get('model', {}).get('name', 'claude-sonnet-4-20250514')
        )

        page_nums = [int(p.strip()) - 1 for p in args.vision_pages.split(',')]
        analysis_results = []

        for pdf_path in documents['pdf'][:1]:
            for page_num in page_nums:
                print(f"  Analyzing page {page_num + 1}...")
                try:
                    img_bytes = extractor.extract_page_as_image(pdf_path, page_num, dpi=200)
                    analysis = vision.analyze_diagram(img_bytes)
                    analysis_results.append(f"### Page {page_num + 1} Diagram Analysis:\n{analysis}")
                    print(f"    Analysis complete: {len(analysis)} chars")
                except Exception as e:
                    print(f"    Error: {e}")

        diagram_analysis = "\n\n".join(analysis_results)

    # ==========================================================================
    # Step 5: LLM knowledge extraction
    # ==========================================================================
    step_num = 5 if (args.use_deplot or args.use_vision) else 3
    print("\n" + "-" * 50)
    print(f"{step_num}. Extracting knowledge with LLM...")
    print("-" * 50)

    agent = KnowledgeExtractorAgent(config_path)
    knowledge_code = agent.extract_knowledge(
        document_text,
        region_name,
        chart_data=chart_data,
        diagram_analysis=diagram_analysis
    )

    # ==========================================================================
    # Step 6: Save output
    # ==========================================================================
    step_num += 1
    print("\n" + "-" * 50)
    print(f"{step_num}. Saving knowledge base...")
    print("-" * 50)

    output_filename = f"{region_name}_geological_knowledge_auto.py"
    output_path = output_dir / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(knowledge_code)

    print(f"\n  Saved: {output_path}")

    # Validate
    step_num += 1
    print("\n" + "-" * 50)
    print(f"{step_num}. Validating generated code...")
    print("-" * 50)

    try:
        compile(knowledge_code, output_filename, 'exec')
        print("  Syntax check: PASSED")
    except SyntaxError as e:
        print(f"  Syntax check: FAILED - {e}")
        print("  The generated file may need manual fixes.")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  COMPLETE!")
    print("=" * 70)
    print(f"\n  Output files:")
    print(f"    - Knowledge base: {output_path}")
    print(f"    - OCR text: {ocr_output_path}")
    print(f"    - File size: {output_path.stat().st_size:,} bytes")

    print(f"\n  Extraction summary:")
    print(f"    - Text pages: {total_stats['pymupdf_pages'] + total_stats['tesseract_pages'] + total_stats['paddleocr_pages']}")
    print(f"    - Chart data: {'Yes' if chart_data else 'No'}")
    print(f"    - Diagram analysis: {'Yes' if diagram_analysis else 'No'}")

    print(f"\n  Next steps:")
    print(f"    1. Review: {output_filename}")
    print(f"    2. Compare with existing knowledge base")
    print(f"    3. Import: from {region_name}_geological_knowledge_auto import ROCK_UNITS")

    return output_path


if __name__ == "__main__":
    main()
