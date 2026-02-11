"""
FastAPI Server for Geology Agent
지질단면 분석 API 서버
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .routes import router


def _safe_import(module_name, original_stdout):
    """
    모듈을 임포트하되, sys.stdout이 교체된 경우 새 wrapper의 buffer를
    detach하여 GC 시 buffer가 닫히지 않도록 처리.
    """
    try:
        __import__(module_name)
    except Exception:
        pass

    # 모듈이 sys.stdout을 교체했으면 새 wrapper를 안전하게 폐기
    if sys.stdout is not original_stdout:
        try:
            sys.stdout.detach()  # buffer를 분리하여 GC 시 close 방지
        except Exception:
            pass
        sys.stdout = original_stdout


def _preimport_geology_modules():
    """
    지질 분석 모듈을 미리 임포트하여 sys.stdout 수정을 한 번만 수행.
    각 모듈의 sys.stdout = io.TextIOWrapper(...) 호출 후 buffer를 detach하여
    원래 stdout이 손상되지 않도록 보호.
    """
    # 부모 디렉토리를 경로에 추가
    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # stdout/argv 보존
    original_stdout = sys.stdout
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]

    try:
        # 이 모듈들이 sys.stdout을 수정하므로 미리 임포트
        _safe_import('cross_section_analysis', original_stdout)
        _safe_import('geologist_agent_llm', original_stdout)
        _safe_import('apply_llm_results', original_stdout)
        _safe_import('review_agent_llm', original_stdout)
    finally:
        # stdout/argv 완전 복원
        sys.stdout = original_stdout
        sys.argv = original_argv


def create_app() -> FastAPI:
    """
    FastAPI 앱 생성

    Returns:
        FastAPI 앱 인스턴스
    """
    app = FastAPI(
        title="Geology Cross-Section Agent API",
        description="""
## 지질단면 자동 분석 에이전트 API

이 API는 LangGraph 기반의 지질단면 분석 워크플로우를 제공합니다.

### 주요 기능

- **지질 데이터 분석**: 수치지질도 shapefile과 DEM 데이터를 분석
- **지식베이스 추출**: PDF 도폭설명서에서 지질 지식 추출
- **LLM 경계 분석**: Claude API를 사용한 지하구조 추정
- **단면도 생성**: 고품질 지질단면도 PNG 생성
- **HTML 보고서**: 상세 분석 보고서 생성
- **품질 검토**: AI 기반 단면도 품질 평가

### 사용 예시

```python
import requests

# 분석 요청
response = requests.post(
    "http://localhost:8000/analyze",
    json={"region": "서울"}
)
task_id = response.json()["task_id"]

# 상태 확인
status = requests.get(f"http://localhost:8000/status/{task_id}")
print(status.json())

# 결과 조회
results = requests.get(f"http://localhost:8000/results/{task_id}")
print(results.json())
```
        """,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_preimport():
        """서버 시작 시 지질 모듈 미리 임포트"""
        _preimport_geology_modules()

    # 라우터 등록
    app.include_router(router, prefix="/api/v1")

    # 대시보드 HTML 서빙
    static_dir = Path(__file__).parent / "static"

    @app.get("/dashboard", tags=["Dashboard"])
    async def dashboard():
        """대시보드 UI"""
        html_path = static_dir / "dashboard.html"
        content = html_path.read_text(encoding="utf-8")
        return HTMLResponse(
            content=content,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # 정적 파일 서빙 (출력 파일 접근용)
    output_dir = Path(__file__).parent.parent.parent / "output"
    if output_dir.exists():
        app.mount("/files", StaticFiles(directory=str(output_dir)), name="files")

    @app.get("/", tags=["Root"])
    async def root():
        """루트 엔드포인트"""
        return {
            "message": "Geology Cross-Section Agent API",
            "version": "0.1.0",
            "docs": "/docs",
            "redoc": "/redoc",
        }

    return app


# 앱 인스턴스 (uvicorn에서 직접 사용)
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "geology_agent.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
