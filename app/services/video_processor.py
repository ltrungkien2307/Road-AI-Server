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
    
    
    def detect_damages_batch(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run AI detection on multiple frames using batch processing
        More efficient than single frame detection
        
        Args:
            frames: List of frame data dicts
            
        Returns:
            List of all detections from all frames
        """
        try:
            if not frames:
                return []
            
            # Get frame paths
            frame_paths = [f['frame_path'] for f in frames]
            
            logger.info(f"Running batch detection on {len(frames)} frames")
            
            # Run batch detection
            batch_detections = self.damage_detector.detect_batch(
                image_paths=frame_paths,
                confidence_threshold=settings.MODEL_CONFIDENCE_THRESHOLD
            )
            
            # Merge results with frame metadata
            all_detections = []
            
            for frame_idx, (frame_data, detections) in enumerate(zip(frames, batch_detections)):
                for detection in detections:
                    detection.update({
                        'frame_number': frame_data['frame_number'],
                        'timestamp': frame_data['timestamp'],
                        'gps_location': frame_data.get('gps_location'),
                        'frame_path': frame_data['frame_path']
                    })
                    all_detections.append(detection)
            
            logger.info(f"Batch detection complete: {len(all_detections)} detections found")
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Batch detection failed: {e}", exc_info=True)
            # Fallback to single frame detection
            logger.info("Falling back to single frame detection")
            all_detections = []
            for frame_data in frames:
                detections = self.detect_damages(frame_data)
                all_detections.extend(detections)
            return all_detections
    
    
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
        Create damage records in Supabase with cropped images
        
        Args:
            grouped_detections: Grouped damage detections
            
        Returns:
            Created damage records with IDs
        """
        try:
            logger.info(f"Creating {len(grouped_detections)} damage records")
            
            created_damages = []
            
            for idx, detection in enumerate(grouped_detections, 1):
                try:
                    logger.info(f"Processing damage {idx}/{len(grouped_detections)}: {detection['class_name']}")
                    
                    # Upload detection image (crop from frame)
                    image_url = self._upload_detection_image(detection)
                    
                    if image_url:
                        logger.info(f"✅ Image uploaded: {image_url}")
                    else:
                        logger.warning(f"⚠️  Could not upload image for damage {idx}")
                    
                    # Prepare damage data with GPS coordinates and image
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
                        'image_url': image_url,  # Cropped image from video
                        'metadata': {
                            'confidence': detection['confidence'],
                            'frame_number': detection['frame_number'],
                            'timestamp': detection['timestamp'],
                            'bbox': detection['bbox'],
                            'detection_count': detection.get('detection_count', 1),
                            'frame_range': detection.get('frame_range', [detection['frame_number']]),
                            'confidence_range': detection.get('confidence_range', [detection['confidence'], detection['confidence']])
                        }
                    }
                    
                    # Create in database
                    damage = self.storage_service.create_damage_sync(damage_data)
                    
                    if damage:
                        logger.info(f"✅ Damage record created: {damage['id']}")
                        created_damages.append({
                            'damage_id': damage['id'],
                            'class_name': detection['class_name'],
                            'confidence': detection['confidence'],
                            'severity': detection['severity'],
                            'gps_location': detection['gps_location'],
                            'image_url': image_url
                        })
                    else:
                        logger.error(f"Failed to create damage record in database")
                        
                except Exception as e:
                    logger.error(f"Failed to create damage record {idx}: {e}", exc_info=True)
                    continue
            
            logger.info(f"Successfully created {len(created_damages)} damage records with images")
            
            return created_damages
            
        except Exception as e:
            logger.error(f"Failed to create damage records: {e}", exc_info=True)
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
        Crop detection area from frame and upload to Cloudinary
        
        Args:
            detection: Detection dict with frame_path, bbox, gps_location
            
        Returns:
            Cloudinary URL of cropped damage image, or None if failed
        """
        try:
            import cv2
            from app.utils.cloudinary import upload_image_to_cloudinary
            
            frame_path = detection.get('frame_path')
            bbox = detection.get('bbox')
            
            if not frame_path or not bbox or not len(bbox) >= 4:
                logger.warning(f"Missing frame or bbox for detection")
                return None
            
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Failed to read frame: {frame_path}")
                return None
            
            # Expand bbox by 10% for context
            x1, y1, x2, y2 = [int(x) for x in bbox[:4]]
            width = x2 - x1
            height = y2 - y1
            
            padding = int(max(width, height) * 0.1)
            
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            # Crop detection area
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size == 0:
                logger.warning("Cropped area is empty")
                return None
            
            # Save cropped image to temp file
            import tempfile
            import os
            
            temp_dir = self.temp_dir / 'crops'
            temp_dir.mkdir(exist_ok=True)
            
            crop_filename = f"damage_{uuid.uuid4()}.jpg"
            crop_path = temp_dir / crop_filename
            
            cv2.imwrite(str(crop_path), cropped)
            
            # Upload to Cloudinary
            logger.info(f"Uploading detection image: {crop_path}")
            
            image_url = upload_image_to_cloudinary(
                image_path=str(crop_path),
                folder=f"damages/{self.job_id}"
            )
            
            if image_url:
                logger.info(f"✅ Detection image uploaded: {image_url}")
            else:
                logger.warning("Failed to get URL from Cloudinary")
            
            return image_url
            
        except Exception as e:
            logger.error(f"Failed to upload detection image: {e}")
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