# Seoul Geology Cross-Section Project
# 서울 지질 단면도 자동 생성 프로젝트

지질도폭 설명서와 수치지질도 데이터를 기반으로 LLM을 활용하여 지질 단면도를 자동 생성하는 파이프라인입니다.

## 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │ PDF 설명서   │  │ Shapefiles  │  │ DEM 데이터   │                      │
│  └─────────────┘  └─────────────┘  └─────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1. knowledge_extractor_agent.py                                        │
│     지식베이스 추출 (OCR + Vision API)                                    │
│     Output: {region}_geological_knowledge.py                            │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. cross_section_analysis.py                                           │
│     단면선 분석 및 정의                                                   │
│     Output: crosssection_analysis.json                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. geologist_agent_llm.py                                              │
│     LLM 기반 경계면 경사각 추정                                           │
│     Output: llm_geologist_results.json                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. apply_llm_results.py                                                │
│     단면도 생성 및 HTML 보고서                                            │
│     Output: *_llm_section.png, llm_geologist_report.html                │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  5. review_agent_llm.py                                                 │
│     품질 검토 및 개선점 제안                                               │
│     Output: review_report.html                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 파일 설명

### 핵심 모듈

| 파일 | 역할 | 입력 | 출력 |
|------|------|------|------|
| `config.yaml` | 전역 설정 (경로, API 키, 모델) | - | - |
| `config_loader.py` | 설정 파일 로더 (공통 유틸리티) | config.yaml | CONFIG 객체 |

### 파이프라인 스크립트

#### 1. knowledge_extractor_agent.py
**지식베이스 자동 추출 에이전트**

지질도폭 설명서(PDF)에서 암석 단위, 구조지질, 접촉관계 등의 정보를 자동 추출하여 Python 지식베이스를 생성합니다.

- **텍스트 추출**: 3단계 폴백 (PyMuPDF → Tesseract → PaddleOCR)
- **차트 추출**: DePlot (선택적)
- **다이어그램 분석**: Claude Vision API (선택적)

```bash
# 기본 실행 (OCR만)
python knowledge_extractor_agent.py --data-dir "D:/Data/SeoulData"

# Vision API 사용
python knowledge_extractor_agent.py --data-dir "D:/Data/SeoulData" --use-vision --vision-pages 1,5,10
```

**출력**: `{region}_geological_knowledge.py`, `{region}_ocr_text.txt`

---

#### 2. cross_section_analysis.py
**단면선 분석 및 정의**

Shapefile의 기존 단면선을 분석하고, 추가 단면선을 자동 생성합니다.

- 기존 단면선 분석 (azimuth, 길이, 교차 암석)
- 자동 단면선 생성 (지질 경계 횡단)
- DEM 기반 고도 프로파일 추출

```bash
python cross_section_analysis.py --data-dir "D:/Data/SeoulData"
```

**출력**: `crosssection_analysis.json`

---

#### 3. geologist_agent_llm.py
**LLM 기반 지하구조 추정 에이전트**

지질 경계면의 경사각, 접촉 유형, 3D 기하를 LLM을 활용하여 추정합니다.

- 지식베이스 참조 (seoul_geological_knowledge.py)
- 경계면별 경사각/주향 추정
- 접촉 유형 및 깊이 변화 분석

```bash
python geologist_agent_llm.py --data-dir "D:/Data/SeoulData"
```

**출력**: `llm_geologist_results.json`

---

#### 4. apply_llm_results.py
**단면도 생성 및 HTML 보고서**

LLM 추정 결과를 적용하여 지질 단면도를 생성하고 종합 HTML 보고서를 작성합니다.

- LLM 경사각 적용한 단면도 시각화
- 암석 단위별 색상 및 패턴
- 단층, 접촉면 표현
- 종합 HTML 보고서 생성

```bash
python apply_llm_results.py --data-dir "D:/Data/SeoulData"
```

**출력**: `*_llm_section.png`, `llm_geologist_report.html`

---

#### 5. review_agent_llm.py
**품질 검토 에이전트**

생성된 단면도의 지질학적 타당성을 Vision API로 검토하고 개선점을 제안합니다.

- 단면도 이미지 분석
- 지질학적 오류 검출
- 개선 권고사항 제시

```bash
python review_agent_llm.py --data-dir "D:/Data/SeoulData"
```

**출력**: `review_report.html`, `review_report.txt`

---

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install anthropic pyyaml geopandas rasterio fitz pytesseract pillow matplotlib shapely numpy
```

**선택적 의존성**:
```bash
pip install paddlepaddle paddleocr  # PaddleOCR (복잡한 레이아웃용)
pip install transformers            # DePlot (차트 추출용)
```

**Tesseract OCR 설치** (Windows):
```bash
winget install tesseract-ocr.tesseract
```
언어팩 다운로드: [tessdata](https://github.com/tesseract-ocr/tessdata)에서 `eng.traineddata`, `kor.traineddata`를 `C:\Program Files\Tesseract-OCR\tessdata\`에 복사

### 2. 설정 파일 (config.yaml)

```yaml
paths:
  base_dir: "D:/Claude/GeologyModel"
  data_dir: "D:/Claude/GeologyModel/SeoulData"
  output_dir: "D:/Claude/GeologyModel/output"
  geology_dir: "D:/Claude/GeologyModel/SeoulData/수치지질도_5만축척_FG33_서울"
  dem_file: "D:/Claude/GeologyModel/SeoulData/DEM5m_2018_서울_5186/DEM5m_2018_서울.img"

anthropic:
  api_key: "your-api-key-here"

model:
  name: "claude-sonnet-4-20250514"
```

### 3. 데이터 구조

```
SeoulData/
├── 지질도_도폭설명서_5만축척_FG33_서울.pdf
├── 수치지질도_5만축척_FG33_서울/
│   ├── FG33_Geology_50K_Litho.shp
│   ├── FG33_Geology_50K_Boundary.shp
│   ├── FG33_Geology_50K_Fault.shp
│   ├── FG33_Geology_50K_Foliation.shp
│   └── FG33_Geology_50K_Crosssectionline.shp
├── DEM5m_2018_서울_5186/
│   └── DEM5m_2018_서울.img
└── seoul_geological_knowledge.py  (생성됨)
```

---

## 전체 실행 예시

```bash
# 1. 지식베이스 추출 (최초 1회)
python knowledge_extractor_agent.py --data-dir "D:/Data/SeoulData" --use-vision --vision-pages 5,10,15,20

# 2. 단면선 분석
python cross_section_analysis.py --data-dir "D:/Data/SeoulData"

# 3. LLM 경사각 추정
python geologist_agent_llm.py --data-dir "D:/Data/SeoulData"

# 4. 단면도 및 보고서 생성
python apply_llm_results.py --data-dir "D:/Data/SeoulData"

# 5. 품질 검토
python review_agent_llm.py --data-dir "D:/Data/SeoulData"
```

---

## 출력 결과물

| 파일 | 설명 |
|------|------|
| `seoul_geological_knowledge.py` | 지질 지식베이스 (Python 모듈) |
| `crosssection_analysis.json` | 단면선 정의 데이터 |
| `llm_geologist_results.json` | LLM 경사각 추정 결과 |
| `*_llm_section.png` | 단면도 이미지 |
| `llm_geologist_report.html` | 종합 HTML 보고서 |
| `review_report.html` | 품질 검토 보고서 |

---

## 비용 추정 (Claude Sonnet 4 기준)

| 단계 | 예상 비용 |
|------|----------|
| knowledge_extractor (Vision, 15페이지) | ~$0.23 |
| geologist_agent (20 경계면) | ~$0.10 |
| apply_llm_results | ~$0.02 |
| review_agent (5 단면도) | ~$0.15 |
| **총계** | **~$0.50/실행** |

---

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 개발되었습니다.

## 참고문헌

- 지질도폭설명서 서울 1:50,000 (홍승호, 이병주, 황상기, 1982), 한국동력자원연구소
