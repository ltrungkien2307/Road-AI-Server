"""Celery tasks for async processing"""

from app.tasks.celery_tasks import celery_app, process_video_task

__all__ = ['celery_app', 'process_video_task']