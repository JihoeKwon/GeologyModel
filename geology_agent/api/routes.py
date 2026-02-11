"""
API Routes for Geology Agent
API 엔드포인트 정의
"""

import uuid
import asyncio
from typing import Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, BackgroundTasks

from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    StatusResponse,
    ResultsResponse,
    RegionInfo,
    RegionsResponse,
    HealthResponse,
    TaskStatus,
    OutputFiles,
    ReviewSummary,
    StepInfo,
    ErrorResponse,
)
from ..data_registry import REGIONS, check_region_data, get_base_dir
from ..workflow import run_workflow

# 라우터 생성
router = APIRouter()

# 태스크 저장소 (실제 프로덕션에서는 Redis 등 사용)
_tasks: Dict[str, Dict[str, Any]] = {}

# ThreadPoolExecutor for running workflows
_executor = ThreadPoolExecutor(max_workers=2)


def _run_workflow_task(task_id: str, region: str, skip_knowledge: bool, skip_review: bool):
    """백그라운드에서 워크플로우 실행"""
    try:
        _tasks[task_id]["status"] = TaskStatus.RUNNING
        _tasks[task_id]["current_step"] = "check_data"

        def on_progress(step_name: str):
            _tasks[task_id]["current_step"] = step_name

        result = run_workflow(
            region=region,
            task_id=task_id,
            skip_knowledge_extraction=skip_knowledge,
            skip_review=skip_review,
            progress_callback=on_progress,
        )

        _tasks[task_id]["result"] = result
        _tasks[task_id]["status"] = TaskStatus.COMPLETED if result.get("status") == "completed" else TaskStatus.FAILED
        _tasks[task_id]["completed_at"] = datetime.now()

        if result.get("status") == "error":
            _tasks[task_id]["error"] = result.get("error")

    except Exception as e:
        _tasks[task_id]["status"] = TaskStatus.FAILED
        _tasks[task_id]["error"] = str(e)
        _tasks[task_id]["completed_at"] = datetime.now()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """서비스 상태 확인"""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now(),
    )


@router.get("/regions", response_model=RegionsResponse, tags=["Regions"])
async def get_regions():
    """사용 가능한 지역 목록 조회"""
    regions = []

    for region_name, region_config in REGIONS.items():
        data_check = check_region_data(region_name, get_base_dir())

        regions.append(RegionInfo(
            name=region_name,
            name_kr=region_config.get("name_kr", region_name),
            name_en=region_config.get("name_en", region_name),
            available=data_check["available"],
            has_pdf=data_check["paths"].get("pdf_path") is not None,
            has_knowledge_base=data_check["paths"].get("knowledge_base_path") is not None,
        ))

    return RegionsResponse(regions=regions)


@router.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    지질단면 분석 요청

    지역의 지질 데이터를 분석하여 단면도를 생성합니다.
    """
    # 지역 데이터 확인
    data_check = check_region_data(request.region, get_base_dir())

    if not data_check["available"]:
        raise HTTPException(
            status_code=400,
            detail=f"지역 '{request.region}' 데이터를 사용할 수 없습니다: {data_check['message']}"
        )

    # 태스크 생성
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "task_id": task_id,
        "region": request.region,
        "status": TaskStatus.PENDING,
        "current_step": None,
        "created_at": datetime.now(),
        "completed_at": None,
        "result": None,
        "error": None,
    }

    # 백그라운드에서 워크플로우 실행
    _executor.submit(
        _run_workflow_task,
        task_id,
        request.region,
        request.skip_knowledge_extraction,
        request.skip_review,
    )

    return AnalyzeResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"분석이 시작되었습니다. GET /status/{task_id} 로 진행 상황을 확인하세요.",
    )


@router.get("/status/{task_id}", response_model=StatusResponse, tags=["Analysis"])
async def get_status(task_id: str):
    """
    분석 진행 상황 조회
    """
    if task_id not in _tasks:
        raise HTTPException(
            status_code=404,
            detail=f"태스크를 찾을 수 없습니다: {task_id}"
        )

    task = _tasks[task_id]

    # 진행률 계산
    step_order = ["check_data", "extract_knowledge", "analyze_sections", "generate_visualization", "review", "finalize"]
    current_step = task.get("current_step") or task.get("result", {}).get("current_step")
    progress = None

    if current_step and current_step in step_order:
        progress = (step_order.index(current_step) + 1) / len(step_order)
    elif task["status"] == TaskStatus.COMPLETED:
        progress = 1.0

    return StatusResponse(
        task_id=task_id,
        status=task["status"],
        current_step=current_step,
        progress=progress,
        error=task.get("error"),
    )


@router.get("/results/{task_id}", response_model=ResultsResponse, tags=["Analysis"])
async def get_results(task_id: str):
    """
    분석 결과 조회
    """
    if task_id not in _tasks:
        raise HTTPException(
            status_code=404,
            detail=f"태스크를 찾을 수 없습니다: {task_id}"
        )

    task = _tasks[task_id]

    if task["status"] not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail="분석이 아직 완료되지 않았습니다. /status/{task_id}로 진행 상황을 확인하세요."
        )

    result = task.get("result", {})
    outputs_data = result.get("outputs", {})

    outputs = None
    if outputs_data:
        outputs = OutputFiles(
            knowledge_base=outputs_data.get("knowledge_base"),
            cross_sections=outputs_data.get("cross_sections", []),
            report=outputs_data.get("report"),
            review_report=outputs_data.get("review_report"),
        )

    review_summary = None
    # 검토 결과가 있으면 요약 생성
    if result.get("review_results"):
        review_data = result["review_results"]
        review_summary = ReviewSummary(
            total_reviewed=review_data.get("total_reviewed", 0),
            average_score=review_data.get("average_score", 0.0),
            critical_issues_count=len(review_data.get("critical_issues", [])),
        )

    return ResultsResponse(
        task_id=task_id,
        status=task["status"],
        region=task["region"],
        outputs=outputs,
        review_summary=review_summary,
        summary=result.get("summary"),
        completed_at=task.get("completed_at"),
    )


@router.delete("/tasks/{task_id}", tags=["Analysis"])
async def delete_task(task_id: str):
    """
    태스크 삭제 (완료된 태스크만)
    """
    if task_id not in _tasks:
        raise HTTPException(
            status_code=404,
            detail=f"태스크를 찾을 수 없습니다: {task_id}"
        )

    task = _tasks[task_id]

    if task["status"] == TaskStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="실행 중인 태스크는 삭제할 수 없습니다."
        )

    del _tasks[task_id]

    return {"message": f"태스크 {task_id}가 삭제되었습니다."}


@router.get("/tasks", tags=["Analysis"])
async def list_tasks():
    """
    모든 태스크 목록 조회
    """
    return {
        "tasks": [
            {
                "task_id": task["task_id"],
                "region": task["region"],
                "status": task["status"],
                "created_at": task["created_at"],
                "completed_at": task.get("completed_at"),
            }
            for task in _tasks.values()
        ]
    }
