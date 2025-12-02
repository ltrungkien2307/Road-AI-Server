# app/main.py
"""
Road-AI-Server - FastAPI Main Application
Handles video processing requests and delegates to Celery workers
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

from app.config import settings
from app.schemas.video_job import (
    VideoProcessRequest,
    VideoProcessResponse,
    JobStatusResponse
)
from app.tasks.celery_tasks import process_video_task
from app.services.storage import StorageService
from app.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Initialize services
storage_service = StorageService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    logger.info("🚀 Road-AI-Server starting up...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Celery Broker: {settings.CELERY_BROKER_URL}")
    
    yield
    
    logger.info("🛑 Road-AI-Server shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Road-AI-Server",
    description="AI-powered road damage detection from video surveys",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Road-AI-Server",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }


@app.get("/health/celery")
async def celery_health():
    """Check Celery worker status"""
    try:
        from app.tasks.celery_tasks import celery_app
        
        # Inspect active workers
        inspect = celery_app.control.inspect()
        active_workers = inspect.active()
        
        if not active_workers:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "No active Celery workers"
                }
            )
        
        return {
            "status": "healthy",
            "workers": list(active_workers.keys()),
            "active_tasks": sum(len(tasks) for tasks in active_workers.values())
        }
    except Exception as e:
        logger.error(f"Celery health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# ============================================================
# VIDEO PROCESSING ENDPOINTS
# ============================================================

@app.post("/api/ai/process-video", response_model=VideoProcessResponse)
async def process_video(request: VideoProcessRequest):
    """
    Submit video for AI processing
    
    Workflow:
    1. Validate request data
    2. Create job record in database
    3. Queue Celery task
    4. Return job ID for status tracking
    """
    try:
        logger.info(f"{'='*60}")
        logger.info(f"📹 NEW VIDEO PROCESSING REQUEST")
        logger.info(f"{'='*60}")
        logger.info(f"📹 Task ID: {request.task_id}")
        logger.info(f"📹 Video URL: {request.video_url[:80]}...")
        logger.info(f"📹 GPS Points: {len(request.gps_log)}")
        logger.info(f"📹 Company ID: {request.company_id}")
        logger.info(f"📹 Road ID: {request.road_id or 'N/A'}")
        logger.info(f"{'='*60}")
        
        # Validate GPS log
        if not request.gps_log or len(request.gps_log) == 0:
            raise HTTPException(
                status_code=400,
                detail="GPS log is required and cannot be empty"
            )
        
        # Create job record in database
        job_id = await storage_service.create_job(
            task_id=request.task_id,
            video_url=request.video_url,
            gps_log=[point.dict() for point in request.gps_log],  # ✅ Convert to dicts
            company_id=request.company_id
)
        
        logger.info(f"✅ Job created with ID: {job_id}")
        
        # Convert Pydantic models to dict for Celery serialization
        # GPSPoint models need to be converted to plain dicts
        request_dict = request.dict()
        request_dict['gps_log'] = [
            {
                'timestamp': point.timestamp,
                'lat': point.lat,
                'lon': point.lon,
                'speed': point.speed,
                'accuracy': point.accuracy,
            }
            for point in request.gps_log
        ]
        
        # Queue Celery task (async processing)
        task = process_video_task.apply_async(
            args=[job_id, request_dict],
            task_id=job_id  # Use same ID for tracking
        )
        
        logger.info(f"📤 Task queued for processing: {task.id}")
        logger.info(f"📤 Video will be processed asynchronously")
        logger.info(f"{'='*60}")
        
        return VideoProcessResponse(
            job_id=job_id,
            status="queued",
            message="Video processing task queued successfully",
            estimated_time_minutes=5  # Rough estimate
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error submitting video job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit video processing job: {str(e)}"
        )


@app.get("/api/ai/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get processing status for a job
    
    Status flow:
    - queued: Task in queue
    - processing: Currently processing
    - completed: Successfully completed
    - failed: Processing failed
    """
    try:
        # Get job from database
        job = await storage_service.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get Celery task status
        from app.tasks.celery_tasks import celery_app
        task = celery_app.AsyncResult(job_id)
        
        # Build response
        response = JobStatusResponse(
            job_id=job_id,
            status=job['status'],
            progress=job.get('progress', 0),
            message=job.get('message', ''),
            created_at=job['created_at'],
            updated_at=job['updated_at']
        )
        
        # Add results if completed
        if job['status'] == 'completed' and job.get('results'):
            response.results = job['results']
        
        # Add error if failed
        if job['status'] == 'failed' and job.get('error'):
            response.error = job['error']
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )


@app.get("/api/ai/results/{task_id}")
async def get_task_results(task_id: str):
    """
    Get all detection results for a task
    Returns damages created from video processing
    """
    try:
        results = await storage_service.get_task_damages(task_id)
        
        return {
            "task_id": task_id,
            "damage_count": len(results),
            "damages": results
        }
        
    except Exception as e:
        logger.error(f"Error getting task results: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task results: {str(e)}"
        )


@app.get("/api/ai/results/{task_id}/map")
async def get_task_damages_for_map(task_id: str):
    """
    Get damages in GeoJSON format for map visualization
    Includes image URLs and metadata
    """
    try:
        from app.utils.map_utils import create_geojson_feature_collection
        
        damages = await storage_service.get_task_damages(task_id)
        
        if not damages:
            return {
                "type": "FeatureCollection",
                "features": [],
                "metadata": {
                    "total_damages": 0,
                    "damage_types": {},
                    "severity_distribution": {}
                }
            }
        
        geojson = create_geojson_feature_collection(damages)
        
        return geojson
        
    except Exception as e:
        logger.error(f"Error getting map data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get map data: {str(e)}"
        )


@app.get("/api/ai/results/{task_id}/heatmap")
async def get_task_damages_heatmap(task_id: str):
    """
    Get damages in heatmap format for map visualization
    Returns intensity-weighted points for density mapping
    """
    try:
        from app.utils.map_utils import generate_heatmap_data
        
        damages = await storage_service.get_task_damages(task_id)
        heatmap_data = generate_heatmap_data(damages)
        
        return {
            "task_id": task_id,
            "heatmap_data": heatmap_data,
            "damage_count": len(damages),
            "metadata": {
                "format": "[lat, lon, intensity]",
                "intensity_scale": {
                    "critical": 1.0,
                    "high": 0.7,
                    "medium": 0.5,
                    "low": 0.3
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting heatmap data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get heatmap data: {str(e)}"
        )


# ============================================================
# ADMIN ENDPOINTS
# ============================================================

@app.get("/api/ai/jobs")
async def list_jobs(
    status: str = None,
    limit: int = 50,
    offset: int = 0
):
    """List all processing jobs with optional filtering"""
    try:
        jobs = await storage_service.list_jobs(
            status=status,
            limit=limit,
            offset=offset
        )
        
        return {
            "jobs": jobs,
            "count": len(jobs),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list jobs: {str(e)}"
        )


@app.delete("/api/ai/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    try:
        from app.tasks.celery_tasks import celery_app
        
        # Revoke Celery task
        celery_app.control.revoke(job_id, terminate=True)
        
        # Update job status
        await storage_service.update_job_status(
            job_id=job_id,
            status='cancelled',
            message='Job cancelled by user'
        )
        
        return {
            "message": "Job cancelled successfully",
            "job_id": job_id
        }
        
    except Exception as e:
        logger.error(f"Error cancelling job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel job: {str(e)}"
        )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.ENVIRONMENT == "development" else None
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info"
    )