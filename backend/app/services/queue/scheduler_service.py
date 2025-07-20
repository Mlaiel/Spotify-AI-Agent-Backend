import logging
from typing import Callable, Dict, Any, List
import threading
import time

logger = logging.getLogger("scheduler_service")

class SchedulerService:
    """
    Service de scheduler avancé : cron, interval, delayed jobs, hooks, audit, sécurité, observabilité.
    Utilisé pour planifier jobs IA, analytics, Spotify, batch, maintenance, etc.
    """
    def __init__(self):
        self.scheduled_jobs: List[Dict[str, Any]] = []
        self.hooks: List[Callable] = []
        self.running = False
    def register_hook(self, hook: Callable):
        self.hooks.append(hook)
        logger.info(f"Scheduler hook enregistré: {hook}")
    def schedule(self, job_type: str, payload: Dict[str, Any], delay: int = 0, interval: int = 0):
        job = {"type": job_type, "payload": payload, "delay": delay, "interval": interval}
        self.scheduled_jobs.append(job)
        logger.info(f"Job planifié: {job}")
        self.audit(job, "scheduled")
    def start(self, processor: Callable[Dict[str, Any], None]):
        self.running = True
        def run():
            while self.running:
                now = time.time()
                for job in list(self.scheduled_jobs):
                    if "_next" not in job:
                        job["_next"] = now + job.get("delay", 0)
                    if now >= job["_next"]:
                        processor(job)
                        for hook in self.hooks:
                            hook(job)
                        self.audit(job, "executed")
                        if job.get("interval", 0):
                            job["_next"] = now + job["interval"]
                        else:
                            self.scheduled_jobs.remove(job)
                time.sleep(1)
        threading.Thread(target=run, daemon=True).start()
    def stop(self):
        self.running = False
        logger.info("Scheduler arrêté")
    def audit(self, job: Dict[str, Any], status: str):
        logger.info(f"[AUDIT] Scheduler job {job.get('type')} status: {status} | job: {job}")
