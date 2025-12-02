# app/models/damage_detector.py
"""
AI Model wrapper for damage detection
Uses YOLOv8 trained on road damage dataset
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

from app.config import settings
from app.utils.helpers import get_damage_type_name, calculate_severity
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("⚠️  Ultralytics not installed. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False


class DamageDetector:
    """
    Damage detection model wrapper
    Loads YOLOv8 model and provides inference interface
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize damage detector
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self.device = 'cuda' if self._check_gpu() else 'cpu'
        
        self._load_model()
    
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            available = torch.cuda.is_available()
            if available:
                logger.info(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("💻 Using CPU for inference")
            return available
        except:
            return False
    
    
    def _load_model(self):
        """Load YOLO model from weights file"""
        try:
            if not YOLO_AVAILABLE:
                raise ImportError("Ultralytics YOLO not available")
            
            model_path = Path(self.model_path)
            
            if not model_path.exists():
                logger.error(f"❌ Model not found: {model_path}")
                logger.info("Please ensure your trained model is placed at:")
                logger.info(f"   {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info(f"Loading model from: {model_path}")
            
            # Load YOLO model
            self.model = YOLO(str(model_path))
            
            # Set device
            self.model.to(self.device)
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            logger.info(f"   Classes: {self.model.names}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    
    def detect(
        self,
        image_path: str,
        confidence_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Run detection on a single image
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence (0-1)
            
        Returns:
            List of detections with bounding boxes and metadata
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            conf_threshold = confidence_threshold or settings.MODEL_CONFIDENCE_THRESHOLD
            
            # Read image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            height, width = image.shape[:2]
            
            # Run inference
            results = self.model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=settings.MODEL_IOU_THRESHOLD,
                device=self.device,
                verbose=False
            )
            
            # Parse results
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Extract box data
                    xyxy = box.xyxy[0].cpu().numpy()  # Bounding box
                    conf = float(box.conf[0].cpu().numpy())  # Confidence
                    cls = int(box.cls[0].cpu().numpy())  # Class ID
                    
                    # Get class name
                    class_name = get_damage_type_name(cls)
                    
                    # Calculate severity
                    detection_data = {
                        'confidence': conf,
                        'bbox': xyxy.tolist(),
                        'image_width': width,
                        'image_height': height
                    }
                    severity = calculate_severity(detection_data)
                    
                    # Build detection dict
                    detection = {
                        'class_id': cls,
                        'class_name': class_name,
                        'confidence': round(conf, 4),
                        'bbox': [
                            float(xyxy[0]),  # x1
                            float(xyxy[1]),  # y1
                            float(xyxy[2]),  # x2
                            float(xyxy[3])   # y2
                        ],
                        'severity': severity,
                        'image_width': width,
                        'image_height': height
                    }
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return []
    
    
    def detect_batch(
        self,
        image_paths: List[str],
        confidence_threshold: float = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Run detection on multiple images (batch processing)
        More efficient for large batches
        
        Args:
            image_paths: List of image paths
            confidence_threshold: Minimum confidence
            
        Returns:
            List of detection lists (one per image)
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            conf_threshold = confidence_threshold or settings.MODEL_CONFIDENCE_THRESHOLD
            
            # Run batch inference
            results = self.model.predict(
                source=image_paths,
                conf=conf_threshold,
                iou=settings.MODEL_IOU_THRESHOLD,
                device=self.device,
                verbose=False,
                stream=True  # Generator for memory efficiency
            )
            
            all_detections = []
            
            for result in results:
                image_detections = []
                boxes = result.boxes
                
                # Get image dimensions
                height, width = result.orig_shape
                
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    class_name = get_damage_type_name(cls)
                    
                    detection_data = {
                        'confidence': conf,
                        'bbox': xyxy.tolist(),
                        'image_width': width,
                        'image_height': height
                    }
                    severity = calculate_severity(detection_data)
                    
                    detection = {
                        'class_id': cls,
                        'class_name': class_name,
                        'confidence': round(conf, 4),
                        'bbox': [float(x) for x in xyxy],
                        'severity': severity,
                        'image_width': width,
                        'image_height': height
                    }
                    
                    image_detections.append(detection)
                
                all_detections.append(image_detections)
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            return [[] for _ in image_paths]
    
    
    def visualize_detection(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        output_path: str = None
    ) -> str:
        """
        Draw bounding boxes on image
        
        Args:
            image_path: Input image path
            detections: Detection results
            output_path: Output image path (optional)
            
        Returns:
            Path to output image
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            # Define colors for each class (BGR format)
            colors = {
                'alligator_crack': (0, 255, 255),    # Yellow
                'pothole': (0, 0, 255),              # Red
                'other_corruption': (255, 0, 255),   # Magenta
                'longitudinal_crack': (255, 165, 0), # Orange
                'transverse_crack': (0, 255, 0)      # Green
            }
            
            # Draw each detection
            for det in detections:
                x1, y1, x2, y2 = [int(x) for x in det['bbox']]
                class_name = det['class_name']
                confidence = det['confidence']
                severity = det['severity']
                
                # Get color
                color = colors.get(class_name, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name} {confidence:.2f} ({severity})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Background for text
                cv2.rectangle(
                    image,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                
                # Text
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
            
            # Save output
            if output_path is None:
                output_path = image_path.replace('.jpg', '_detected.jpg')
            
            cv2.imwrite(output_path, image)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return image_path