"""
Pydantic Schemas for Geology Agent API
API 요청/응답 스키마 정의
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """태스크 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalyzeRequest(BaseModel):
    """분석 요청 스키마"""
    region: str = Field(
        default="서울",
        description="분석할 지역명 (예: 서울, Seoul)",
        examples=["서울", "Seoul"]
    )
    skip_knowledge_extraction: bool = Field(
        default=True,
        description="지식베이스 추출 스킵 여부 (기존 지식베이스 사용 시 True)"
    )
    skip_review: bool = Field(
        default=False,
        description="검토 단계 스킵 여부"
    )


class AnalyzeResponse(BaseModel):
    """분석 응답 스키마"""
    task_id: str = Field(description="태스크 고유 ID")
    status: TaskStatus = Field(description="태스크 상태")
    message: str = Field(description="상태 메시지")


class StepInfo(BaseModel):
    """단계 정보"""
    name: str
    status: str
    timestamp: Optional[datetime] = None
    error: Optional[str] = None


class StatusResponse(BaseModel):
    """상태 조회 응답 스키마"""
    task_id: str = Field(description="태스크 고유 ID")
    status: TaskStatus = Field(description="태스크 상태")
    current_step: Optional[str] = Field(default=None, description="현재 진행 중인 단계")
    progress: Optional[float] = Field(default=None, description="진행률 (0.0 ~ 1.0)")
    steps: Optional[List[StepInfo]] = Field(default=None, description="단계별 상태")
    error: Optional[str] = Field(default=None, description="에러 메시지")


class OutputFiles(BaseModel):
    """출력 파일 정보"""
    knowledge_base: Optional[str] = Field(default=None, description="지식베이스 파일 경로")
    cross_sections: List[str] = Field(default_factory=list, description="단면도 이미지 경로들")
    report: Optional[str] = Field(default=None, description="HTML 보고서 경로")
    review_report: Optional[str] = Field(default=None, description="검토 보고서 경로")


class ReviewSummary(BaseModel):
    """검토 요약"""
    total_reviewed: int = Field(default=0, description="검토된 단면 수")
    average_score: float = Field(default=0.0, description="평균 점수")
    critical_issues_count: int = Field(default=0, description="심각한 문제 수")


class ResultsResponse(BaseModel):
    """결과 조회 응답 스키마"""
    task_id: str = Field(description="태스크 고유 ID")
    status: TaskStatus = Field(description="태스크 상태")
    region: str = Field(description="분석된 지역")
    outputs: Optional[OutputFiles] = Field(default=None, description="출력 파일들")
    review_summary: Optional[ReviewSummary] = Field(default=None, description="검토 요약")
    summary: Optional[str] = Field(default=None, description="결과 요약 메시지")
    completed_at: Optional[datetime] = Field(default=None, description="완료 시간")


class RegionInfo(BaseModel):
    """지역 정보 스키마"""
    name: str = Field(description="지역명")
    name_kr: str = Field(description="한글 지역명")
    name_en: str = Field(description="영문 지역명")
    available: bool = Field(description="데이터 사용 가능 여부")
    has_pdf: bool = Field(default=False, description="PDF 도폭설명서 존재 여부")
    has_knowledge_base: bool = Field(default=False, description="지식베이스 존재 여부")


class RegionsResponse(BaseModel):
    """지역 목록 응답 스키마"""
    regions: List[RegionInfo] = Field(description="사용 가능한 지역 목록")


class ErrorResponse(BaseModel):
    """에러 응답 스키마"""
    error: str = Field(description="에러 메시지")
    detail: Optional[str] = Field(default=None, description="상세 에러 정보")


class HealthResponse(BaseModel):
    """헬스체크 응답 스키마"""
    status: str = Field(default="healthy", description="서비스 상태")
    version: str = Field(description="API 버전")
    timestamp: datetime = Field(description="현재 시간")
