# Geology Cross-Section Agent
# 지질 단면도 자동 생성 에이전트

수치지질도, DEM, 지질도폭설명서(PDF)를 입력받아 LLM 기반으로 지질 단면도를 자동 생성하는 파이프라인입니다.
LangGraph 워크플로우와 FastAPI 서버를 통해 다중 지역(서울, 부산 등)의 지질 분석을 자동화합니다.

---

## 목차

1. [시스템 아키텍처](#1-시스템-아키텍처)
2. [파이프라인 상세](#2-파이프라인-상세)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [설치 및 설정](#4-설치-및-설정)
5. [실행 방법](#5-실행-방법)
6. [API 레퍼런스](#6-api-레퍼런스)
7. [다중 지역 지원](#7-다중-지역-지원)
8. [핵심 기술 상세](#8-핵심-기술-상세)
9. [출력 결과물](#9-출력-결과물)
10. [비용 추정](#10-비용-추정)
11. [변경 이력](#11-변경-이력)

---

## 1. 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│                          입력 데이터                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  PDF 설명서   │  │  Shapefiles  │  │  DEM (5m)    │               │
│  │  (도폭설명서) │  │  (수치지질도) │  │  (수치표고)   │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
└─────────┼─────────────────┼─────────────────┼───────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      LangGraph Workflow                              │
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │① 데이터  │───▶│② 지식    │───▶│③ 단면    │───▶│④ 시각화  │      │
│  │   확인   │    │   추출   │    │   분석   │    │   생성   │      │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘      │
│                                                       │             │
│                                  ┌──────────┐    ┌────▼─────┐      │
│                                  │⑥ 최종화  │◀───│⑤ 품질    │      │
│                                  │          │    │   검토   │      │
│                                  └──────────┘    └──────────┘      │
└──────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          출력 결과물                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  단면도 PNG   │  │  HTML 보고서  │  │  검토 보고서  │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

### 실행 모드

| 모드 | 명령 | 설명 |
|------|------|------|
| CLI | `python -m geology_agent.main --region "서울"` | 커맨드라인에서 직접 실행 |
| API Server | `python -m geology_agent.main --server` | FastAPI REST API 서버 |
| Dashboard | `http://localhost:8000/dashboard` | 웹 브라우저 대시보드 UI |
| Import | `from geology_agent import run_workflow` | Python 코드에서 직접 호출 |

---

## 2. 파이프라인 상세

### Step 1: 데이터 확인 (`check_data`)

`data_registry.py`를 통해 지역별 데이터 존재 여부를 검증합니다.

```
입력: 지역명 ("서울", "부산", ...)
검증: Shapefile(6종), DEM, PDF(선택), 지식베이스(선택)
출력: data_available (bool), data_paths (dict)
```

- 지역별 CRS 자동 설정 (서울: EPSG:5186, 부산: EPSG:5179)
- Shapefile 인코딩 자동 감지 (UTF-8 → EUC-KR 폴백)
- 누락 파일 목록 반환

---

### Step 2: 지식베이스 추출 (`extract_knowledge`)

**스크립트**: `knowledge_extractor_agent.py`

지질도폭설명서(PDF)에서 암석 단위, 구조지질, 접촉관계 정보를 자동 추출하여 Python 지식베이스를 생성합니다.

```
입력: PDF 도폭설명서 (예: 지질도_도폭설명서_5만축척_FG33_서울.pdf)
방법: 3단계 OCR 폴백 (PyMuPDF → Tesseract → PaddleOCR) + Claude Vision API
출력: {region}_geological_knowledge_auto.py
```

**생성되는 지식베이스 구조**:
```python
ROCK_UNITS = {
    "Ka": {
        "name_kr": "안산암류",
        "expected_contact_type": "conformable",  # 접촉 유형
        "expected_dip_range": "거의 수평-30°",     # 예상 경사 범위
        ...
    },
}
STRUCTURAL_GEOLOGY = { ... }     # 구조지질 정보
CONTACT_RULES = { ... }          # 접촉관계 규칙 (쌍별)
DIP_ESTIMATION_RULES = { ... }   # 경사각 추정 규칙
LITHOIDX_TO_KB = { ... }         # Shapefile→KB 코드 매핑
```

---

### Step 3: 단면 분석 (`analyze_sections`)

**스크립트**: `cross_section_analysis.py` → `geologist_agent_llm.py`

두 단계로 구성됩니다:

#### 3-A. 단면선 정의 및 지표 분석

```
입력: Shapefiles (Litho, Boundary, Fault, Foliation, CrossSection), DEM
처리:
  1. 기존 단면선 분석 (지질도에 표시된 공식 단면선)
  2. 자동 단면선 생성 (주향에 수직 방향, 지질 경계 최대 횡단)
  3. DEM 기반 고도 프로파일 추출
  4. 단면선별 교차 암상/경계/단층 추출
출력: crosssection_analysis.json
```

**자동 단면선 생성 로직**:
- 엽리(Foliation) 데이터에서 평균 주향(mean strike) 계산
- 주향에 수직인 최적 단면 방위각 결정
- 지질도 범위 내에서 3개 자동 단면선 생성
- 해안/수역 제외: `litho_union.buffer(500)` 클리핑
- 암상 커버리지 30% 미만 단면 자동 제외
- 암상 다양성 필터: 고유 암상 2종 미만 단면 제외 (해양 위주 단면 방지)

#### 3-B. LLM 기반 경계면 경사각 추정

```
입력: crosssection_analysis.json + 지식베이스
처리:
  1. 주요 지질 경계면 선별 (경계 길이/고도차 기준)
  2. 경계면별 지질학적 맥락 생성 (인접 암상, 엽리 데이터, 고도 프로파일)
  3. Claude API로 구조지질학 전문가 분석 요청
  4. 경사각, 주향, 접촉유형, 심부 거동 등 추정
출력: llm_geologist_results.json
```

**LLM 추정 항목**:
```json
{
  "dip_angle": 75,
  "dip_direction": 179,
  "strike": 89,
  "contact_type": "intrusive",
  "contact_geometry": "curved",
  "depth_behavior": { "dip_change": "flattening", "estimated_depth_to_flatten": 2000 },
  "intrusion_shape": "stock",
  "fold_influence": { "present": false },
  "confidence": "medium",
  "reasoning": "경상분지 내 백악기 화성활동과 연관된 관입접촉으로 해석됨..."
}
```

---

### Step 4: 시각화 생성 (`generate_visualization`)

**스크립트**: `apply_llm_results.py`

LLM 분석 결과를 적용하여 지질 단면도를 생성하고 종합 HTML 보고서를 작성합니다.

```
입력: llm_geologist_results.json + crosssection_analysis.json + Shapefiles + DEM
처리:
  1. 단면선별 지형 프로파일 추출
  2. 단면선별 지표 암상 투영 (Shapefile → 단면선 교차)
  3. 4단계 우선순위 암상 분류로 접촉 유형/경사각 결정
  4. 심부 지질 구조 생성 (관입체 형태, 변성암 경계, 부정합면)
  5. Matplotlib 기반 단면도 렌더링
     - 암석체 에지 윤곽선(검정) + 암석 코드 라벨 표시
     - 관입체 경계: `base_dip` 기반 동적 경사 프로파일 (심부 플레어)
  6. 경계위치도(boundary_location_map) 생성
     - 이중 라인(흰색 배경 + 검정 전경) 단면선 표시
     - 시작점(○)/끝점(□) 마커 + 노란색 라벨
     - 암상 범례 + 접촉유형 범례 + 경계 라벨
  7. 종합 HTML 보고서 생성
출력: *_llm_section.png, boundary_location_map.png, llm_geologist_report.html
```

#### 4단계 우선순위 암상 분류 시스템

단면도 생성 시 각 암상 경계의 접촉 유형과 경사각을 결정하는 핵심 로직입니다:

```
Priority 1: LLM 분석 결과 (정확한 경계 매칭 + KB 교차검증)
  → llm_geologist_results.json에서 인접 암상이 정확히 일치하는 경계 검색
  → KB의 접촉유형과 교차검증하여 불일치 시 LLM 결과 기각
  → 신뢰도: medium~high

Priority 2: 지식베이스 동적 분류 (ROCK_UNITS + LITHOIDX_TO_KB)
  → KB의 CONTACT_RULES에서 암상 쌍별 접촉유형 검색
  → KB의 ROCK_UNITS.expected_contact_type에서 개별 암상 분류
  → LITHOIDX_TO_KB 매핑으로 Shapefile↔KB 코드 불일치 해결
  → 신뢰도: low (KB 기반)

Priority 3: 하드코딩 베이스 (누적 확장형)
  → 서울: Pgr, Jbgr, Kqp, Kqv, Kfl (관입), PCEbngn 등 (변성)
  → 부산: Kbgr, Khgdi, Kga, Kgp 등 (관입), Kan, Kanb, Kts 등 (정합)
  → KB가 없는 지역에서도 기본 분류 가능
  → 신뢰도: low

Priority 4: 제네릭 폴백
  → 55° 경사, unknown 유형, very_low 신뢰도
```

#### 접촉 유형별 단면도 표현

| 접촉 유형 | 색상 | 경계 형태 | 기본 경사 |
|-----------|------|----------|----------|
| intrusive (관입) | 빨강 | 곡면, 심부 확장 | 75° |
| unconformable (부정합) | 파랑 | 불규칙 침식면 | 8° |
| conformable (정합) | 초록 | 평면~완곡면 | 30° |
| fault (단층) | 검정 | 직선 | 75° |

---

### Step 5: 품질 검토 (`review`)

**스크립트**: `review_agent_llm.py`

생성된 단면도의 지질학적 타당성을 Claude Vision API로 검토합니다.

```
입력: *_llm_section.png 이미지
처리:
  1. 단면도 이미지를 Vision API로 분석
  2. 10점 척도 품질 평가 (암상 분포, 접촉관계, 구조 합리성)
  3. 구체적 오류 지적 및 개선 권고
출력: review_report.html, review_results.json
```

**검토 기준**:
- 암상 분포의 지질학적 합리성
- 접촉면 기하의 적절성 (경사각, 형태)
- 충적층 표현의 정확성
- 관입체 형태의 현실성
- 단층 표현의 적절성

---

### Step 6: 최종화 (`finalize`)

모든 출력물을 취합하고 요약 정보를 생성합니다.

```
출력:
  - outputs.knowledge_base: 지식베이스 경로
  - outputs.cross_sections: 단면도 PNG 경로 목록
  - outputs.report: HTML 보고서 경로
  - outputs.review_report: 검토 보고서 경로
  - summary: "{지역} 지질단면도 N개 생성 완료. 평균 품질점수 X.X/10"
```

---

## 3. 프로젝트 구조

```
GeologyModel/
├── config.yaml                          # 전역 설정 (경로, API 키, 모델, 색상)
├── config_loader.py                     # 설정 로더 + reconfigure() 다중지역 패치
├── requirements.txt                     # Python 의존성
│
├── knowledge_extractor_agent.py         # [Step 2] 지식베이스 추출 (OCR+Vision)
├── cross_section_analysis.py            # [Step 3-A] 단면선 분석/자동 생성
├── geologist_agent_llm.py               # [Step 3-B] LLM 경사각 추정
├── apply_llm_results.py                 # [Step 4] 단면도 생성/HTML 보고서
├── review_agent_llm.py                  # [Step 5] Vision API 품질 검토
│
├── geology_agent/                       # Agent Framework (LangGraph + FastAPI)
│   ├── __init__.py                      #   패키지 초기화, run_workflow 노출
│   ├── __main__.py                      #   python -m geology_agent 진입점
│   ├── main.py                          #   CLI 파서 + 실행 디스패처
│   ├── state.py                         #   LangGraph 상태 정의 (TypedDict)
│   ├── data_registry.py                 #   지역별 데이터 등록/조회
│   ├── workflow.py                      #   LangGraph 6-노드 워크플로우
│   │
│   ├── api/                             #   FastAPI REST API
│   │   ├── server.py                    #     앱 생성, CORS, 정적파일 마운트
│   │   ├── routes.py                    #     API 엔드포인트 (analyze, status, results)
│   │   ├── schemas.py                   #     Pydantic 스키마
│   │   └── static/
│   │       └── dashboard.html           #     웹 대시보드 UI
│   │
│   └── tools/                           #   워크플로우 노드 실행기
│       ├── data_check.py                #     데이터 존재 확인
│       ├── knowledge.py                 #     지식베이스 추출 래퍼
│       ├── analysis.py                  #     단면 분석 래퍼
│       ├── visualization.py             #     시각화 생성 래퍼
│       └── review.py                    #     품질 검토 래퍼
│
├── DataRepository/                      # 입력 데이터 (지역별)
│   ├── SeoulData/
│   │   ├── 지질도_도폭설명서_5만축척_FG33_서울.pdf
│   │   ├── 수치지질도_5만축척_FG33_서울/   (Shapefiles 6종)
│   │   ├── DEM5m_2018_서울_5186/          (DEM .img)
│   │   └── seoul_geological_knowledge.py   (지식베이스)
│   └── BusanData/
│       ├── 지질도_도폭설명서_5만축척_IE00_부산.pdf
│       ├── 수치지질도_5만축척_IE00_부산/   (Shapefiles 6종)
│       ├── (B080)공개DEM_35913_img_2025/  (DEM .img)
│       └── busan_geological_knowledge_auto.py  (지식베이스)
│
├── docs/                               # 작업 로그 및 문서
│   ├── work_log_20260203.md            #   서울 관입체 형태 개선
│   └── work_log_20260211.md            #   부산 다중지역 + 시각화 전면 개선
│
├── output/                              # 출력 결과물 (지역별)
│   ├── Seoul/                           #   서울 지역 결과
│   └── Busan/                           #   부산 지역 결과
│
└── tests/
    └── test_geology_agent.py            # 테스트
```

---

## 4. 설치 및 설정

### 4.1 의존성 설치

```bash
pip install -r requirements.txt
```

**주요 의존성**:

| 패키지 | 용도 |
|--------|------|
| `langgraph`, `langchain-core`, `langchain-anthropic` | LangGraph 워크플로우 |
| `fastapi`, `uvicorn`, `pydantic` | REST API 서버 |
| `anthropic` | Claude API (LLM 분석 + Vision 검토) |
| `geopandas`, `rasterio`, `shapely`, `pyproj` | 공간 데이터 처리 |
| `matplotlib`, `numpy` | 시각화 |
| `pymupdf`, `pytesseract`, `pillow` | PDF/OCR 처리 |
| `pyyaml` | 설정 파일 |

**선택적 의존성** (지식베이스 추출 고급 기능):
```bash
pip install paddlepaddle paddleocr  # 복잡한 레이아웃 OCR
pip install transformers            # DePlot 차트 추출
```

**Tesseract OCR** (Windows):
```bash
winget install tesseract-ocr.tesseract
```
언어팩: [tessdata](https://github.com/tesseract-ocr/tessdata)에서 `kor.traineddata`를 `C:\Program Files\Tesseract-OCR\tessdata\`에 복사

### 4.2 설정 파일 (config.yaml)

```yaml
# 경로 설정
paths:
  base_dir: "D:/Claude/GeologyModel"
  data_dir: "D:/Claude/GeologyModel/DataRepository/SeoulData"
  output_dir: "D:/Claude/GeologyModel/output"
  geology_dir: ".../수치지질도_5만축척_FG33_서울"
  dem_file: ".../DEM5m_2018_서울.img"

# Shapefile 이름
shapefiles:
  litho: "FG33_Geology_50K_Litho.shp"
  boundary: "FG33_Geology_50K_Boundary.shp"
  fault: "FG33_Geology_50K_Fault.shp"
  foliation: "FG33_Geology_50K_Foliation.shp"
  crosssection: "FG33_Geology_50K_Crosssectionline.shp"
  frame: "FG33_Geology_50K_Frame.shp"

# 좌표계
crs:
  target: "EPSG:5186"

# Anthropic API
anthropic:
  api_key: "sk-ant-..."

# 모델 설정
model:
  name: "claude-sonnet-4-20250514"
  max_tokens: 2048
  review_model: "claude-sonnet-4-20250514"

# 암상 색상 (서울 기본값, 다른 지역은 자동 보충)
litho_colors:
  PCEbngn: "#FFB6C1"
  Qa: "#FFFACD"
  ...
```

> **참고**: `config.yaml`은 서울 지역 기본값을 포함합니다. 다른 지역 실행 시 `reconfigure()`가 경로, CRS, Shapefile명을 자동으로 패치하며, 미등록 암상의 색상/이름은 Shapefile에서 자동 보충됩니다.

### 4.3 데이터 준비

각 지역별로 다음 데이터가 필요합니다:

| 데이터 | 필수 | 설명 |
|--------|------|------|
| 수치지질도 Shapefiles (6종) | O | Litho, Boundary, Fault, Foliation, CrossSection, Frame |
| DEM (수치표고모델) | O | .img 형식, 5m 해상도 권장 |
| 지질도폭설명서 PDF | - | 지식베이스 자동 추출에 사용 |
| 지식베이스 .py | - | PDF 없이 수동 작성 가능 |

---

## 5. 실행 방법

### 5.1 CLI 실행

```bash
# 서울 지역 전체 파이프라인 (검토 포함)
python -m geology_agent.main --region "서울"

# 부산 지역 (지식베이스 새로 추출 + 검토 스킵)
python -m geology_agent.main --region "부산" --extract-knowledge --skip-review

# 사용 가능한 지역 확인
python -m geology_agent.main --list-regions

# 데이터 존재 여부만 확인
python -m geology_agent.main --region "부산" --check-data
```

### 5.2 API 서버

```bash
# 서버 시작 (기본: 0.0.0.0:8000)
python -m geology_agent.main --server

# 포트 변경
python -m geology_agent.main --server --port 9000
```

서버 시작 후:
- API 문서: http://localhost:8000/docs
- 대시보드: http://localhost:8000/dashboard

### 5.3 레거시 스크립트 개별 실행

```bash
# 1. 지식베이스 추출
python knowledge_extractor_agent.py --data-dir "DataRepository/SeoulData"

# 2. 단면선 분석
python cross_section_analysis.py --data-dir "DataRepository/SeoulData"

# 3. LLM 경사각 추정
python geologist_agent_llm.py --data-dir "DataRepository/SeoulData"

# 4. 단면도 및 보고서 생성
python apply_llm_results.py --data-dir "DataRepository/SeoulData"

# 5. 품질 검토
python review_agent_llm.py --data-dir "DataRepository/SeoulData"
```

### 5.4 Python Import

```python
from geology_agent import run_workflow

result = run_workflow(
    region="부산",
    skip_knowledge_extraction=True,
    skip_review=False,
)

print(result["status"])       # "completed"
print(result["outputs"])      # 생성된 파일 경로들
print(result["summary"])      # 요약 텍스트
```

---

## 6. API 레퍼런스

### 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/api/v1/health` | 서버 상태 확인 |
| `GET` | `/api/v1/regions` | 사용 가능한 지역 목록 |
| `POST` | `/api/v1/analyze` | 분석 요청 (비동기) |
| `GET` | `/api/v1/status/{task_id}` | 진행 상황 조회 |
| `GET` | `/api/v1/results/{task_id}` | 결과 조회 |
| `GET` | `/files/{path}` | 출력 파일 정적 서빙 |
| `GET` | `/dashboard` | 웹 대시보드 UI |

### 분석 요청 예시

```bash
# 분석 시작
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"region": "Busan", "skip_review": true}'

# 응답: {"task_id": "abc-123", "status": "pending", "message": "..."}

# 진행 상황 확인
curl http://localhost:8000/api/v1/status/abc-123

# 응답: {"task_id": "abc-123", "status": "running", "current_step": "generate_visualization", "progress": 0.67}

# 결과 조회
curl http://localhost:8000/api/v1/results/abc-123

# 응답: {"task_id": "abc-123", "status": "completed", "outputs": {...}, "summary": "..."}
```

### Python 클라이언트

```python
import requests

# 분석 요청
resp = requests.post("http://localhost:8000/api/v1/analyze", json={
    "region": "서울",
    "skip_knowledge_extraction": True,
    "skip_review": False,
})
task_id = resp.json()["task_id"]

# 폴링으로 완료 대기
import time
while True:
    status = requests.get(f"http://localhost:8000/api/v1/status/{task_id}").json()
    if status["status"] in ("completed", "error"):
        break
    print(f"진행: {status['current_step']} ({status['progress']:.0%})")
    time.sleep(5)

# 결과 조회
results = requests.get(f"http://localhost:8000/api/v1/results/{task_id}").json()
print(results["summary"])
```

---

## 7. 다중 지역 지원

### 7.1 등록된 지역

| 지역 | CRS | Shapefile 접두사 | DEM |
|------|-----|----------------|-----|
| 서울 | EPSG:5186 | FG33_ | DEM5m_2018_서울.img |
| 부산 | EPSG:5179 | IE00_ | 35913.img |

### 7.2 다중 지역 구현 방식

```
data_registry.py                    config_loader.py
┌─────────────┐                     ┌──────────────────────┐
│ REGIONS = {  │   check_region_data │ reconfigure(region)  │
│   "서울": {  │──────────────────▶  │  ├─ PATHS 패치       │
│     path,    │   paths, CRS,       │  ├─ CRS 패치         │
│     CRS,     │   shapefiles        │  ├─ SHAPEFILES 패치  │
│     shapefiles│                    │  ├─ REGION_NAME 패치  │
│   },         │                     │  └─ KB 리로드         │
│   "부산": {  │                     └──────────────────────┘
│     ...      │                              │
│   }          │                              ▼
└─────────────┘                     레거시 스크립트 모듈 변수 일괄 패치
                                    (cross_section_analysis, geologist_agent_llm,
                                     apply_llm_results, review_agent_llm)
```

`reconfigure()`는 `sys.modules`에 로드된 레거시 스크립트의 모듈 레벨 변수(`PATHS`, `CONFIG`, `TARGET_CRS`, `REGION_NAME_KR` 등)를 지역별 값으로 동적 패치합니다.

### 7.3 새 지역 추가 방법

1. **데이터 준비**: `DataRepository/{RegionName}Data/` 아래에 Shapefiles, DEM 배치
2. **data_registry.py 등록**: `REGIONS` 딕셔너리에 지역 정보 추가
3. **지식베이스 생성**: PDF가 있으면 자동 추출, 없으면 수동 작성
4. **LITHOIDX_TO_KB 매핑**: Shapefile의 LITHOIDX 코드와 KB의 ROCK_UNITS 코드가 다른 경우 매핑 딕셔너리 추가
5. **실행**: `python -m geology_agent.main --region "새지역" --extract-knowledge`

---

## 8. 핵심 기술 상세

### 8.1 Shapefile 인코딩 처리

한국 지질도 Shapefile은 인코딩이 통일되어 있지 않습니다:

```python
# config_loader.py → read_shapefile_safe()
def read_shapefile_safe(shp_path, target_crs=None):
    os.environ['SHAPE_ENCODING'] = ''   # GDAL 자동 추측 방지
    for encoding in ['utf-8', 'euc-kr']:
        try:
            gdf = gpd.read_file(shp_path, encoding=encoding)
            # 한글 검증 로직...
            return gdf.to_crs(target_crs) if target_crs else gdf
        except:
            continue
    return gpd.read_file(shp_path)  # 기본 인코딩 폴백
```

### 8.2 DEM CRS 변환

DEM의 좌표계가 목표 CRS와 다를 수 있으므로 좌표 변환이 필요합니다:

```python
from pyproj import Transformer
transformer = Transformer.from_crs(target_crs, dem_crs, always_xy=True)
dem_x, dem_y = transformer.transform(x, y)
row, col = rowcol(dem_transform, dem_x, dem_y)
```

### 8.3 암상 정보 자동 보충

`config.yaml`에는 서울 지역 암상만 등록되어 있으므로, 다른 지역 실행 시 Shapefile에서 자동으로 보충합니다:

```python
# config_loader.py
auto_populate_litho_info(litho_gdf, LITHO_COLORS, LITHO_NAMES)
# → Shapefile의 LITHOIDX/LITHONAME 속성에서 미등록 암상 색상·이름 자동 할당

auto_populate_age_order(litho_gdf, LITHO_AGE_ORDER)
# → AGE 필드 + 코드 접두사 기반으로 시대 순서 자동 배정
```

### 8.4 LITHOIDX↔KB 코드 매핑

Shapefile의 암상 코드(LITHOIDX)와 지식베이스의 ROCK_UNITS 키가 다른 경우가 있습니다:

| Shapefile | KB | 암석명 |
|-----------|-----|--------|
| Kan | Ka | 안산암류 |
| Kanb | Kavb | 안산암질 화산각력암 |
| Kbgr | Kgr | 흑운모화강암 |
| Khgdi | Kgd | 각섬석화강섬록암 |
| Qa | Qal | 충적층 |

지식베이스의 `LITHOIDX_TO_KB` 딕셔너리가 이 변환을 담당합니다.

### 8.5 LangGraph 워크플로우 라우팅

```
START → check_data ─┬─ data_available → extract_knowledge ─┬─ knowledge_extracted → analyze_sections
                     └─ no data → error                     └─ failed → error
                                                                │
                     ┌─ review_completed → finalize ← ─ ─ ─ ─ ─┤
                     │                                          │
                     └─ visualization OK → review               │
                                                                │
                     error ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ sections_analyzed → generate_visualization
                                                                         │
                                                              finalize ← ┘ (visualization failed)
```

각 노드 실패 시 `error` 노드로 라우팅되며, 에러 메시지와 함께 종료됩니다.

---

## 9. 출력 결과물

### 지역별 출력 디렉토리 (`output/{Region}/`)

| 파일 | 설명 |
|------|------|
| `*_llm_section.png` | 지질 단면도 이미지 (단면별 1개) |
| `llm_geologist_report.html` | 종합 HTML 보고서 (지도, 단면도, 범례, AI 분석 상세) |
| `llm_geologist_results.json` | LLM 경사각 추정 원본 데이터 |
| `crosssection_analysis.json` | 단면선 정의 및 지표 분석 데이터 |
| `boundary_location_map.png` | 분석된 경계면 위치도 |
| `review_report.html` | 품질 검토 보고서 |
| `review_results.json` | 검토 결과 JSON |

### HTML 보고서 구성

```
┌─ 헤더 (지역명, 생성 일시) ──────────────────────────┐
├─ 지표 지질도 (boundary_location_map.png 삽입)        │
├─ 암상 범례 (Shapefile에서 동적 생성)                  │
├─ 지질 개요 (지식베이스 GEOLOGICAL_CONTEXT)            │
├─ 단면도 갤러리 (각 단면별 PNG 삽입)                   │
├─ AI 분석 요약표 (접촉유형, 경사, 주향, 신뢰도)        │
├─ AI 분석 상세 (경계별 추론 과정)                      │
├─ 신뢰도 범례                                         │
└─ 푸터 (모델명, 지역명)                                │
```

---

## 10. 비용 추정

Claude Sonnet 4 기준, 1회 전체 파이프라인 실행 비용:

| 단계 | API 호출 | 예상 비용 |
|------|---------|----------|
| 지식베이스 추출 (Vision, 15pp) | ~3 calls | ~$0.23 |
| LLM 경사각 추정 (5 boundaries) | ~5 calls | ~$0.10 |
| 시각화 생성 | 0 calls | $0.00 |
| 품질 검토 (3 sections) | ~3 calls | ~$0.10 |
| **총계** | | **~$0.43/실행** |

> 지식베이스 추출은 최초 1회만 필요하므로, 반복 실행 시 ~$0.20/실행.

---

## 11. 변경 이력

### 2026-02-11: 부산 다중지역 지원 + 시각화 전면 개선

**다중지역 암상 분류 시스템 (apply_llm_results.py)**
- 4단계 우선순위 분류체계 구현 (LLM+KB교차검증 → KB동적분류 → 하드코딩 → 폴백)
- 부산 17개 LITHOIDX 코드 전부 정상 분류 (이전: 1/17만 분류)
- `LITHOIDX_TO_KB` 매핑으로 Shapefile↔KB 코드 불일치 해결
- LLM Partial Match Override 제거 (KB 정밀 경사각 보존)
- `generate_intrusion_boundary()`에서 `base_dip` 파라미터 활용 동적 경사 프로파일

**시각화 개선 (apply_llm_results.py)**
- 암석체에 검정 에지 윤곽선(linewidth=0.8) + 암석 코드 라벨 표시
- 기반암 배경 alpha 0.6→0.35로 감소 (전경 암석체 가시성 향상)
- 경계위치도: 이중 라인 단면선(흰5px+검2.5px), 마커, 노란 라벨, 암상/접촉유형 범례

**자동단면 품질 필터 (cross_section_analysis.py)**
- 고유 암상 2종 미만 단면 자동 제외 (해양 위주 단면 방지)
- 이전 실행 PNG 자동 정리 (필터링된 단면 잔존 방지)

**대시보드 수정 (dashboard.html, workflow.py)**
- `toFileUrl()` 수정: 지역별 서브디렉토리 경로 보존 (`/files/Busan/report.html`)
- Review Report 링크 연동 (`html_report_path` state 전달)

### 2026-02-03: 서울 관입체 형태 개선

- `generate_intrusion_boundary()` 경사각 조정 (67°→35°, 심부 5°)
- `generate_metamorphic_boundary()` 경사각 조정 (40-60°→25-35°)
- 충적층 최대 두께 40m→20m, 하부 부정합면 점선 추가
- 관입암 접촉타입 강제 적용 (LLM 결과 무관)
- 배경 기반암 채우기 추가

### 2026-02-01: 초기 커밋

- 서울 지역 단일 파이프라인 구축
- Claude API 기반 경사각 추정 + Vision 기반 품질 검토
- LangGraph 워크플로우 + FastAPI 서버 + 웹 대시보드

---

## 참고문헌

- 서울 도폭: 지질도폭설명서 서울 1:50,000 (홍승호, 이병주, 황상기, 1982), 한국동력자원연구소
- 부산 도폭: 지질도폭설명서 부산-가덕 1:50,000 (장태우, 강필종, 박석환, 황상구, 이동우, 1983), 한국동력자원연구소
