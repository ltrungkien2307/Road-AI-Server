# app/schemas/video_job.py
"""
Pydantic schemas for video processing requests/responses
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Job processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GPSPoint(BaseModel):
    """GPS coordinate point from mobile app"""
    timestamp: float = Field(..., description="Seconds from video start")
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    speed: Optional[float] = Field(None, description="Speed in m/s")
    accuracy: Optional[float] = Field(None, description="GPS accuracy in meters")
    
    @validator('lat')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v
    
    @validator('lon')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v


class VideoProcessRequest(BaseModel):
    """Request to process a video"""
    task_id: str = Field(..., description="Task ID from mobile app")
    video_url: str = Field(..., description="Cloudinary video URL")
    gps_log: List[GPSPoint] = Field(..., description="GPS tracking log")
    company_id: str = Field(..., description="Company ID")
    road_id: Optional[str] = Field(None, description="Road ID if known")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_123",
                "video_url": "https://res.cloudinary.com/.../video.mp4",
                "gps_log": [
                    {"timestamp": 0.0, "lat": 10.8231, "lon": 106.6297},
                    {"timestamp": 1.0, "lat": 10.8232, "lon": 106.6298}
                ],
                "company_id": "company_123",
                "road_id": "road_456"
            }
        }


class VideoProcessResponse(BaseModel):
    """Response after submitting video for processing"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Human-readable message")
    estimated_time_minutes: int = Field(..., description="Estimated processing time")


class DetectionResult(BaseModel):
    """Single damage detection result"""
    damage_id: Optional[str] = Field(None, description="Created damage ID")
    class_id: int = Field(..., description="Damage class ID (0-4)")
    class_name: str = Field(..., description="Damage type name")
    confidence: float = Field(..., description="Detection confidence (0-1)")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    frame_number: int = Field(..., description="Frame number in video")
    timestamp: float = Field(..., description="Video timestamp (seconds)")
    gps_location: Optional[Dict[str, float]] = Field(None, description="GPS coordinates")
    severity: str = Field(..., description="Severity level")
    image_url: Optional[str] = Field(None, description="Detection image URL")


class JobResults(BaseModel):
    """Complete job results"""
    total_frames_processed: int
    total_detections: int
    damages_created: int
    processing_time_seconds: float
    detections: List[DetectionResult]
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics by damage type"
    )


class JobStatusResponse(BaseModel):
    """Job status check response"""
    job_id: str
    status: JobStatus
    progress: int = Field(0, description="Progress percentage (0-100)")
    message: str = Field("", description="Status message")
    created_at: datetime
    updated_at: datetime
    results: Optional[JobResults] = None
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_abc123",
                "status": "processing",
                "progress": 45,
                "message": "Processing frame 450/1000",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:35:00Z"
            }
        }


class DamageCreateRequest(BaseModel):
    """Internal schema for creating damage records"""
    company_id: str
    road_id: Optional[str]
    task_id: str
    job_id: str
    damage_type: str
    severity: str
    latitude: float
    longitude: float
    description: str
    image_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "company_id": "company_123",
                "road_id": "road_456",
                "task_id": "task_789",
                "job_id": "job_abc",
                "damage_type": "pothole",
                "severity": "high",
                "latitude": 10.8231,
                "longitude": 106.6297,
                "description": "Pothole detected with 85% confidence",
                "metadata": {
                    "confidence": 0.85,
                    "frame_number": 120,
                    "bbox": [100, 200, 300, 400]
                }
            }
        }