"""
api/routers/health.py
Liveness and readiness probes.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    from api.state import AppState
    return HealthResponse(
        status="ok",
        pipeline_ready=AppState.pipeline is not None,
    )
