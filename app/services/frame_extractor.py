# app/services/frame_extractor.py
"""
Video frame extraction service
Extracts frames from video at specified FPS with GPS mapping
"""

import cv2
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

from app.config import settings
from app.services.gps_mapper import GPSMapper
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class FrameExtractor:
    """
    Extract frames from video with GPS synchronization
    """
    
    def __init__(self):
        self.gps_mapper = GPSMapper()
    
    
    def extract_frames_with_gps(
        self,
        video_path: str,
        gps_log: List[Dict[str, Any]],
        output_dir: str,
        fps: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from video and map to GPS coordinates
        
        Args:
            video_path: Path to video file
            gps_log: GPS tracking data
            output_dir: Directory to save frames
            fps: Frames per second to extract
            
        Returns:
            List of frame data with paths and GPS info
        """
        try:
            logger.info(f"Extracting frames from {video_path}")
            logger.info(f"Target FPS: {fps}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps if video_fps > 0 else 0
            
            logger.info(f"Video info:")
            logger.info(f"  FPS: {video_fps:.2f}")
            logger.info(f"  Total frames: {total_frames}")
            logger.info(f"  Duration: {duration:.2f}s")
            
            # Calculate frame interval
            frame_interval = int(video_fps / fps) if fps < video_fps else 1
            
            frames_data = []
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frame at interval
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / video_fps
                    
                    # Save frame
                    frame_filename = f"frame_{extracted_count:05d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    
                    # Get GPS coordinates for this timestamp
                    try:
                        gps_location = self.gps_mapper.interpolate_gps(
                            timestamp=timestamp,
                            gps_log=gps_log
                        )
                    except Exception as e:
                        logger.warning(f"Failed to get GPS for frame {extracted_count}: {e}")
                        gps_location = None
                    
                    # Store frame data
                    frame_data = {
                        'frame_number': extracted_count,
                        'frame_path': frame_path,
                        'timestamp': timestamp,
                        'gps_location': gps_location
                    }
                    
                    frames_data.append(frame_data)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"✅ Extracted {extracted_count} frames")
            
            return frames_data
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise
    
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            }
            
            cap.release()
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {}


# ============================================================
# app/utils/cloudinary.py
# ============================================================
"""Cloudinary utilities"""

import requests
import cloudinary
import cloudinary.uploader
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Configure Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET
)


def download_video_from_cloudinary(video_url: str, output_path: str):
    """
    Download video from Cloudinary URL
    
    Args:
        video_url: Cloudinary video URL
        output_path: Local path to save video
    """
    try:
        logger.info(f"Downloading video from Cloudinary")
        
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Write to file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Video downloaded to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        raise


def upload_image_to_cloudinary(image_path: str, folder: str = "damages") -> str:
    """
    Upload image to Cloudinary
    
    Args:
        image_path: Local image path
        folder: Cloudinary folder
        
    Returns:
        Cloudinary URL
    """
    try:
        response = cloudinary.uploader.upload(
            image_path,
            folder=folder,
            resource_type="image"
        )
        
        return response['secure_url']
        
    except Exception as e:
        logger.error(f"Failed to upload image: {e}")
        return None


# ============================================================
# app/utils/logger.py
# ============================================================
"""Logging setup"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str) -> logging.Logger:
    """
    Setup logger with custom formatting
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / 'road_ai.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger