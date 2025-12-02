"""Pydantic schemas for API requests/responses"""

from app.schemas.video_job import (
    VideoProcessRequest,
    VideoProcessResponse,
    JobStatusResponse,
    GPSPoint,
    JobStatus,
    DetectionResult,
    JobResults,
    DamageCreateRequest
)

__all__ = [
    'VideoProcessRequest',
    'VideoProcessResponse',
    'JobStatusResponse',
    'GPSPoint',
    'JobStatus',
    'DetectionResult',
    'JobResults',
    'DamageCreateRequest'
]