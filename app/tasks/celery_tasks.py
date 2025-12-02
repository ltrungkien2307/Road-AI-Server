# app/tasks/celery_tasks.py
"""
Celery tasks for async video processing
Main workflow orchestrator
"""

from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, task_failure
import logging
from typing import Dict, Any
import time
import httpx

from app.config import settings
from app.services.video_processor import VideoProcessor
from app.services.storage import StorageService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize Celery app
celery_app = Celery(
    'road_ai_tasks',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks
)

# Initialize services
storage_service = StorageService()


# ============================================================
# CELERY SIGNALS
# ============================================================

@task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    """Called before task execution"""
    logger.info(f"🚀 Task started: {task.name} [{task_id}]")


@task_postrun.connect
def task_postrun_handler(task_id, task, *args, **kwargs):
    """Called after task execution"""
    logger.info(f"✅ Task completed: {task.name} [{task_id}]")


@task_failure.connect
def task_failure_handler(task_id, exception, *args, **kwargs):
    """Called when task fails"""
    logger.error(f"❌ Task failed: {task_id} - {exception}")


# ============================================================
# CUSTOM TASK CLASS WITH PROGRESS TRACKING
# ============================================================

class ProgressTask(Task):
    """Custom task class with progress tracking"""
    
    def update_progress(self, job_id: str, progress: int, message: str):
        """Update job progress in database"""
        try:
            storage_service.update_job_progress_sync(
                job_id=job_id,
                progress=progress,
                message=message
            )
            logger.info(f"Progress [{job_id}]: {progress}% - {message}")
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")


# ============================================================
# MAIN VIDEO PROCESSING TASK
# ============================================================

@celery_app.task(
    bind=True,
    base=ProgressTask,
    name='process_video_task',
    max_retries=2,
    default_retry_delay=60
)
def process_video_task(self, job_id: str, request_data: Dict[str, Any]):
    """
    Main video processing task
    
    Workflow:
    1. Download video from Cloudinary
    2. Extract frames based on GPS timestamps
    3. Run AI model on each frame
    4. Map detections to GPS coordinates
    5. Group nearby detections
    6. Create damage records in database
    7. Update task status
    
    Args:
        job_id: Unique job identifier
        request_data: VideoProcessRequest data
    """
    
    start_time = time.time()
    
    try:
        logger.info(f"{'='*60}")
        logger.info(f"🚀 STARTING VIDEO PROCESSING JOB")
        logger.info(f"{'='*60}")
        logger.info(f"🚀 Job ID: {job_id}")
        logger.info(f"🚀 Task ID: {request_data['task_id']}")
        logger.info(f"🚀 Video URL: {request_data['video_url'][:80]}...")
        logger.info(f"🚀 GPS Points: {len(request_data['gps_log'])}")
        logger.info(f"🚀 Company ID: {request_data.get('company_id', 'N/A')}")
        logger.info(f"{'='*60}")
        
        # Update status to processing
        self.update_progress(job_id, 0, "Initializing video processing")
        storage_service.update_job_status_sync(
            job_id=job_id,
            status='processing',
            message='Starting video processing'
        )
        
        # Initialize video processor
        processor = VideoProcessor(
            job_id=job_id,
            task_id=request_data['task_id'],
            company_id=request_data['company_id'],
            road_id=request_data.get('road_id')
        )
        
        # Step 1: Download video (10%)
        self.update_progress(job_id, 10, "Downloading video from Cloudinary")
        video_path = processor.download_video(request_data['video_url'])
        logger.info(f"✅ Video downloaded: {video_path}")
        
        # Step 2: Extract frames (20%)
        self.update_progress(job_id, 20, "Extracting frames from video")
        frames = processor.extract_frames(
            video_path=video_path,
            gps_log=request_data['gps_log']
        )
        logger.info(f"✅ Extracted {len(frames)} frames")
        
        # Step 3: Run AI detection (20% - 80%)
        self.update_progress(job_id, 30, f"Running AI detection on {len(frames)} frames")
        
        # Use batch processing for efficiency
        all_detections = processor.detect_damages_batch(frames)
        
        logger.info(f"✅ Total detections: {len(all_detections)}")
        
        # Step 4: Group nearby detections (85%)
        self.update_progress(job_id, 85, "Grouping nearby detections")
        grouped_detections = processor.group_detections(all_detections)
        logger.info(f"✅ Grouped into {len(grouped_detections)} unique damages")
        
        # Step 5: Create damage records (90%)
        self.update_progress(job_id, 90, "Creating damage records in database")
        damages_created = processor.create_damage_records(grouped_detections)
        logger.info(f"✅ Created {len(damages_created)} damage records")
        
        # Step 6: Update task status (95%)
        self.update_progress(job_id, 95, "Updating task status")
        processor.update_task_status(damages_created)
        
        # Step 7: Cleanup (98%)
        self.update_progress(job_id, 98, "Cleaning up temporary files")
        processor.cleanup()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build results
        results = {
            'total_frames_processed': len(frames),
            'total_detections': len(all_detections),
            'damages_created': len(damages_created),
            'processing_time_seconds': round(processing_time, 2),
            'detections': [
                {
                    'damage_id': d['damage_id'],
                    'class_name': d['class_name'],
                    'confidence': d['confidence'],
                    'severity': d['severity'],
                    'gps_location': d['gps_location']
                }
                for d in damages_created
            ],
            'summary': processor.generate_summary(damages_created)
        }
        
        # Mark as completed
        self.update_progress(job_id, 100, "Processing completed successfully")
        storage_service.update_job_status_sync(
            job_id=job_id,
            status='completed',
            message='Video processing completed',
            results=results
        )
        
        logger.info(f"{'='*60}")
        logger.info(f"✅ Job completed: {job_id}")
        logger.info(f"   Task ID: {request_data['task_id']}")
        logger.info(f"   Frames: {len(frames)}")
        logger.info(f"   Detections: {len(all_detections)}")
        logger.info(f"   Damages: {len(damages_created)}")
        logger.info(f"   Time: {processing_time:.2f}s")
        logger.info(f"{'='*60}")
        
        # Callback to backend to update task status (non-blocking)
        try:
            import asyncio
            asyncio.run(notify_backend_completion(
                task_id=request_data['task_id'],
                job_id=job_id,
                damages_count=len(damages_created),
                processing_time=processing_time
            ))
        except Exception as callback_error:
            logger.error(f"⚠️ Failed to notify backend (non-blocking): {callback_error}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Job failed: {job_id}", exc_info=True)
        
        # Update job status to failed
        error_message = str(e)
        storage_service.update_job_status_sync(
            job_id=job_id,
            status='failed',
            message=f'Processing failed: {error_message}',
            error=error_message
        )
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying job {job_id} (attempt {self.request.retries + 1})")
            raise self.retry(exc=e, countdown=60)
        
        raise


# ============================================================
# ADDITIONAL HELPER TASKS
# ============================================================

@celery_app.task(name='cleanup_old_jobs')
def cleanup_old_jobs():
    """
    Periodic task to clean up old job records
    Run daily via celery beat
    """
    try:
        logger.info("🧹 Starting cleanup of old jobs")
        deleted = storage_service.cleanup_old_jobs_sync(days=30)
        logger.info(f"✅ Cleaned up {deleted} old jobs")
        return deleted
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise


@celery_app.task(name='health_check')
def health_check():
    """Simple health check task"""
    return {"status": "healthy", "timestamp": time.time()}


# ============================================================
# BACKEND CALLBACK
# ============================================================

async def notify_backend_completion(
    task_id: str,
    job_id: str,
    damages_count: int,
    processing_time: float
):
    """
    Notify backend that video processing is completed
    This will update the task status to 'completed'
    """
    callback_url = settings.BACKEND_CALLBACK_URL
    if not callback_url:
        logger.info("ℹ️ Backend callback URL not configured, skipping notification")
        return
    
    try:
        logger.info(f"📞 Notifying backend of completion: {callback_url}")
        logger.info(f"   Task ID: {task_id}")
        logger.info(f"   Job ID: {job_id}")
        logger.info(f"   Damages: {damages_count}")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{callback_url}/api/tasks/{task_id}/ai-processing-complete",
                json={
                    "job_id": job_id,
                    "task_id": task_id,
                    "status": "completed",
                    "damages_count": damages_count,
                    "processing_time_seconds": round(processing_time, 2),
                    "completed_at": time.time()
                }
            )
            
            if response.status_code == 200:
                logger.info("✅ Backend notified successfully")
            else:
                logger.warning(f"⚠️ Backend callback returned status {response.status_code}: {response.text}")
                
    except Exception as e:
        logger.error(f"❌ Failed to notify backend: {e}")
        # Don't raise - this is non-blocking