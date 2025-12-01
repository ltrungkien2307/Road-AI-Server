# app/services/storage.py
"""
Storage service for Supabase operations
Handles database operations for jobs and damages
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import logging

from supabase import create_client, Client
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class StorageService:
    """
    Supabase storage service
    Handles all database operations
    """
    
    def __init__(self):
        """Initialize Supabase client"""
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY
        )
        logger.info("✅ Supabase client initialized")
    
    
    # ============================================================
    # JOB MANAGEMENT
    # ============================================================
    
    async def create_job(
        self,
        task_id: str,
        video_url: str,
        gps_log: List[Dict],
        company_id: str
    ) -> str:
        """
        Create new processing job record
        
        Returns:
            Job ID (UUID)
        """
        try:
            job_data = {
                'task_id': task_id,
                'video_url': video_url,
                'gps_log': json.dumps(gps_log),
                'company_id': company_id,
                'status': 'queued',
                'progress': 0,
                'message': 'Job queued for processing',
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            response = self.supabase.table('ai_jobs').insert(job_data).execute()
            
            job_id = response.data[0]['id']
            logger.info(f"Job created: {job_id}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            raise
    
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        try:
            response = self.supabase.table('ai_jobs').select('*').eq('id', job_id).execute()
            
            if response.data:
                return response.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    
    async def update_job_status(
        self,
        job_id: str,
        status: str,
        message: str = None,
        results: Dict = None,
        error: str = None
    ):
        """Update job status"""
        try:
            update_data = {
                'status': status,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            if message:
                update_data['message'] = message
            if results:
                update_data['results'] = json.dumps(results)
            if error:
                update_data['error'] = error
            if status == 'completed':
                update_data['completed_at'] = datetime.utcnow().isoformat()
            
            self.supabase.table('ai_jobs').update(update_data).eq('id', job_id).execute()
            
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")
    
    
    async def update_job_progress(self, job_id: str, progress: int, message: str):
        """Update job progress"""
        try:
            self.supabase.table('ai_jobs').update({
                'progress': progress,
                'message': message,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', job_id).execute()
            
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")
    
    
    # Sync versions for Celery (doesn't support async)
    def create_job_sync(self, *args, **kwargs) -> str:
        return asyncio.run(self.create_job(*args, **kwargs))
    
    def update_job_status_sync(self, *args, **kwargs):
        asyncio.run(self.update_job_status(*args, **kwargs))
    
    def update_job_progress_sync(self, *args, **kwargs):
        asyncio.run(self.update_job_progress(*args, **kwargs))
    
    
    async def list_jobs(
        self,
        status: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """List jobs with optional filtering"""
        try:
            query = self.supabase.table('ai_jobs').select('*')
            
            if status:
                query = query.eq('status', status)
            
            query = query.order('created_at', desc=True).range(offset, offset + limit - 1)
            
            response = query.execute()
            return response.data
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []
    
    
    async def cleanup_old_jobs(self, days: int = 30) -> int:
        """Delete old completed jobs"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            response = self.supabase.table('ai_jobs')\
                .delete()\
                .eq('status', 'completed')\
                .lt('completed_at', cutoff_date.isoformat())\
                .execute()
            
            deleted_count = len(response.data) if response.data else 0
            logger.info(f"Cleaned up {deleted_count} old jobs")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
    
    def cleanup_old_jobs_sync(self, days: int = 30) -> int:
        return asyncio.run(self.cleanup_old_jobs(days))
    
    
    # ============================================================
    # DAMAGE MANAGEMENT
    # ============================================================
    
    async def create_damage(self, damage_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Create new damage record
        
        Args:
            damage_data: Damage information including GPS, type, severity
        """
        try:
            # Prepare data for insert
            insert_data = {
                'company_id': damage_data['company_id'],
                'road_id': damage_data.get('road_id'),
                'damage_type': damage_data['damage_type'],
                'severity': damage_data['severity'],
                'latitude': damage_data['latitude'],
                'longitude': damage_data['longitude'],
                'description': damage_data['description'],
                'image_url': damage_data.get('image_url'),
                'status': 'pending',  # Default status
                'metadata': json.dumps(damage_data.get('metadata', {})),
                'reported_at': datetime.utcnow().isoformat()
            }
            
            response = self.supabase.table('damages').insert(insert_data).execute()
            
            if response.data:
                damage = response.data[0]
                logger.info(f"Damage created: {damage['id']}")
                return damage
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create damage: {e}")
            return None
    
    
    def create_damage_sync(self, damage_data: Dict[str, Any]) -> Optional[Dict]:
        """Sync version for Celery"""
        return asyncio.run(self.create_damage(damage_data))
    
    
    async def get_task_damages(self, task_id: str) -> List[Dict]:
        """Get all damages associated with a task"""
        try:
            response = self.supabase.table('damages')\
                .select('*')\
                .contains('metadata', json.dumps({'task_id': task_id}))\
                .execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Failed to get task damages: {e}")
            return []
    
    
    # ============================================================
    # TASK UPDATES
    # ============================================================
    
    async def update_task_notes(self, task_id: str, additional_notes: str):
        """Append notes to task"""
        try:
            # Get current task
            response = self.supabase.table('tasks').select('notes').eq('id', task_id).execute()
            
            if response.data:
                current_notes = response.data[0].get('notes', '')
                new_notes = (current_notes or '') + additional_notes
                
                self.supabase.table('tasks').update({
                    'notes': new_notes,
                    'updated_at': datetime.utcnow().isoformat()
                }).eq('id', task_id).execute()
                
        except Exception as e:
            logger.error(f"Failed to update task notes: {e}")
    
    
    def update_task_notes_sync(self, task_id: str, additional_notes: str):
        """Sync version"""
        asyncio.run(self.update_task_notes(task_id, additional_notes))