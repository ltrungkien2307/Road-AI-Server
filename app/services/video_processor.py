# app/services/video_processor.py
"""
Video processing service - Main orchestrator
Handles video download, frame extraction, AI detection, and result storage
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from app.config import settings
from app.services.frame_extractor import FrameExtractor
from app.services.gps_mapper import GPSMapper
from app.services.storage import StorageService
from app.models.damage_detector import DamageDetector
from app.utils.cloudinary import download_video_from_cloudinary
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class VideoProcessor:
    """
    Main video processing orchestrator
    Coordinates all steps from video download to damage record creation
    """
    
    def __init__(
        self,
        job_id: str,
        task_id: str,
        company_id: str,
        road_id: Optional[str] = None
    ):
        self.job_id = job_id
        self.task_id = task_id
        self.company_id = company_id
        self.road_id = road_id
        
        # Initialize services
        self.frame_extractor = FrameExtractor()
        self.gps_mapper = GPSMapper()
        self.storage_service = StorageService()
        self.damage_detector = DamageDetector()
        
        # Create temporary directory for this job
        self.temp_dir = Path(settings.TEMP_DIR) / job_id
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VideoProcessor initialized for job {job_id}")
    
    
    def download_video(self, video_url: str) -> str:
        """
        Download video from Cloudinary
        
        Args:
            video_url: Cloudinary video URL
            
        Returns:
            Local path to downloaded video
        """
        try:
            logger.info(f"Downloading video from: {video_url}")
            
            # Generate local filename
            video_filename = f"{self.job_id}_video.mp4"
            video_path = self.temp_dir / video_filename
            
            # Download
            download_video_from_cloudinary(
                video_url=video_url,
                output_path=str(video_path)
            )
            
            # Verify file exists
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found after download: {video_path}")
            
            # Check file size
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            logger.info(f"Video downloaded: {file_size_mb:.2f} MB")
            
            if file_size_mb > settings.MAX_VIDEO_SIZE_MB:
                raise ValueError(f"Video too large: {file_size_mb:.2f}MB (max: {settings.MAX_VIDEO_SIZE_MB}MB)")
            
            return str(video_path)
            
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            raise
    
    
    def extract_frames(
        self,
        video_path: str,
        gps_log: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from video based on GPS timestamps
        
        Args:
            video_path: Local path to video
            gps_log: GPS tracking data from mobile app
            
        Returns:
            List of frame data with GPS info
        """
        try:
            logger.info(f"Extracting frames from: {video_path}")
            logger.info(f"GPS points: {len(gps_log)}")
            
            # Extract frames
            frames = self.frame_extractor.extract_frames_with_gps(
                video_path=video_path,
                gps_log=gps_log,
                output_dir=str(self.temp_dir / 'frames'),
                fps=settings.FRAME_EXTRACTION_FPS
            )
            
            logger.info(f"Extracted {len(frames)} frames")
            
            return frames
            
        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            raise
    
    
    def detect_damages(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run AI detection on a single frame
        
        Args:
            frame_data: Frame info with path and GPS data
            
        Returns:
            List of detections for this frame
        """
        try:
            # Run YOLO detection
            detections = self.damage_detector.detect(
                image_path=frame_data['frame_path'],
                confidence_threshold=settings.MODEL_CONFIDENCE_THRESHOLD
            )
            
            # Add frame metadata to each detection
            for detection in detections:
                detection.update({
                    'frame_number': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'gps_location': frame_data.get('gps_location'),
                    'frame_path': frame_data['frame_path']
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed for frame {frame_data.get('frame_number')}: {e}")
            return []
    
    
    def group_detections(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group nearby detections into unique damage instances
        
        Args:
            detections: All detections from video
            
        Returns:
            Grouped detections (1 per unique damage)
        """
        try:
            logger.info(f"Grouping {len(detections)} detections")
            
            # Filter by confidence
            filtered = [
                d for d in detections
                if d['confidence'] >= settings.MIN_CONFIDENCE_FOR_DAMAGE
            ]
            
            logger.info(f"After confidence filter: {len(filtered)} detections")
            
            # Group by GPS proximity
            grouped = self.gps_mapper.group_by_proximity(
                detections=filtered,
                threshold_meters=settings.GPS_PROXIMITY_THRESHOLD_METERS
            )
            
            logger.info(f"After grouping: {len(grouped)} unique damages")
            
            return grouped
            
        except Exception as e:
            logger.error(f"Failed to group detections: {e}")
            raise
    
    
    def create_damage_records(
        self,
        grouped_detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create damage records in Supabase
        
        Args:
            grouped_detections: Grouped damage detections
            
        Returns:
            Created damage records with IDs
        """
        try:
            logger.info(f"Creating {len(grouped_detections)} damage records")
            
            created_damages = []
            
            for detection in grouped_detections:
                try:
                    # Prepare damage data
                    damage_data = {
                        'company_id': self.company_id,
                        'road_id': self.road_id,
                        'task_id': self.task_id,
                        'job_id': self.job_id,
                        'damage_type': detection['class_name'],
                        'severity': detection['severity'],
                        'latitude': detection['gps_location']['lat'],
                        'longitude': detection['gps_location']['lon'],
                        'description': self._generate_description(detection),
                        'image_url': self._upload_detection_image(detection),
                        'metadata': {
                            'confidence': detection['confidence'],
                            'frame_number': detection['frame_number'],
                            'timestamp': detection['timestamp'],
                            'bbox': detection['bbox'],
                            'detection_count': detection.get('detection_count', 1)
                        }
                    }
                    
                    # Create in database
                    damage = self.storage_service.create_damage_sync(damage_data)
                    
                    if damage:
                        created_damages.append({
                            'damage_id': damage['id'],
                            'class_name': detection['class_name'],
                            'confidence': detection['confidence'],
                            'severity': detection['severity'],
                            'gps_location': detection['gps_location']
                        })
                        
                except Exception as e:
                    logger.error(f"Failed to create damage record: {e}")
                    continue
            
            logger.info(f"Successfully created {len(created_damages)} damage records")
            
            return created_damages
            
        except Exception as e:
            logger.error(f"Failed to create damage records: {e}")
            raise
    
    
    def _generate_description(self, detection: Dict[str, Any]) -> str:
        """Generate human-readable description for damage"""
        class_name = detection['class_name'].replace('_', ' ').title()
        confidence = int(detection['confidence'] * 100)
        severity = detection['severity'].upper()
        
        desc = f"{class_name} detected with {confidence}% confidence. "
        desc += f"Severity: {severity}. "
        
        if detection.get('detection_count', 1) > 1:
            desc += f"Detected in {detection['detection_count']} frames. "
        
        return desc
    
    
    def _upload_detection_image(self, detection: Dict[str, Any]) -> Optional[str]:
        """
        Upload detection image to Cloudinary (optional)
        For now, returns None - implement if needed
        """
        # TODO: Crop detection from frame and upload to Cloudinary
        # This would show the exact damage area
        return None
    
    
    def update_task_status(self, damages_created: List[Dict[str, Any]]):
        """Update task status with damage count"""
        try:
            damage_count = len(damages_created)
            
            self.storage_service.update_task_notes_sync(
                task_id=self.task_id,
                additional_notes=f"\n\nAI Analysis: {damage_count} damages detected and recorded."
            )
            
            logger.info(f"Task {self.task_id} updated with {damage_count} damages")
            
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
    
    
    def generate_summary(self, damages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            'total_damages': len(damages),
            'by_type': {},
            'by_severity': {}
        }
        
        for damage in damages:
            # Count by type
            damage_type = damage['class_name']
            summary['by_type'][damage_type] = summary['by_type'].get(damage_type, 0) + 1
            
            # Count by severity
            severity = damage['severity']
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
        
        return summary
    
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to cleanup: {e}")