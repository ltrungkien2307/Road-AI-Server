# app/utils/helpers.py
"""
Helper utility functions for Road-AI-Server
"""

from app.config import settings


def get_damage_type_name(class_id: int) -> str:
    """
    Convert class ID to damage type name
    
    Args:
        class_id: YOLO class ID (0-4)
    
    Returns:
        Damage type name string
    """
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
