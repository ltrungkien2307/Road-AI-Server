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