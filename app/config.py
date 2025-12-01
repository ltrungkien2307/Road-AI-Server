# app/config.py
"""
Configuration management for Road-AI-Server
Loads settings from environment variables
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]
    
    # Celery / Redis
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Supabase
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str
    
    # Cloudinary
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str
    
    # AI Model
    MODEL_PATH: str = "./models/yolov8_road_damage.pt"
    MODEL_CONFIDENCE_THRESHOLD: float = 0.4
    MODEL_IOU_THRESHOLD: float = 0.45
    
    # Video Processing
    FRAME_EXTRACTION_FPS: int = 1  # Extract 1 frame per second
    MAX_VIDEO_SIZE_MB: int = 500
    TEMP_DIR: str = "./temp"
    
    # GPS Matching
    GPS_PROXIMITY_THRESHOLD_METERS: float = 10.0  # Group detections within 10m
    MIN_CONFIDENCE_FOR_DAMAGE: float = 0.5
    
    # Damage Classification
    DAMAGE_CLASSES: dict = {
        0: "alligator_crack",
        1: "pothole", 
        2: "other_corruption",
        3: "longitudinal_crack",
        4: "transverse_crack"
    }
    
    # Severity mapping based on area and confidence
    SEVERITY_THRESHOLDS: dict = {
        "critical": {"area_percent": 15, "confidence": 0.8},
        "high": {"area_percent": 10, "confidence": 0.7},
        "medium": {"area_percent": 5, "confidence": 0.6},
        "low": {"area_percent": 0, "confidence": 0.4}
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure temp directory exists
os.makedirs(settings.TEMP_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_damage_type_name(class_id: int) -> str:
    """Convert class ID to damage type name"""
    return settings.DAMAGE_CLASSES.get(class_id, "unknown")


def calculate_severity(detection: dict) -> str:
    """
    Calculate damage severity based on detection area and confidence
    
    Args:
        detection: Dict with keys: confidence, bbox, image_width, image_height
    
    Returns:
        Severity level: critical, high, medium, low
    """
    confidence = detection['confidence']
    bbox = detection['bbox']
    
    # Calculate area percentage
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    image_area = detection['image_width'] * detection['image_height']
    area_percent = (bbox_area / image_area) * 100
    
    # Determine severity
    for severity, thresholds in settings.SEVERITY_THRESHOLDS.items():
        if (area_percent >= thresholds['area_percent'] and 
            confidence >= thresholds['confidence']):
            return severity
    
    return 'low'