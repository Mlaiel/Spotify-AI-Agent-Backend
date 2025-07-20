import logging
from typing import Callable, Dict, Any, List
import time

logger = logging.getLogger("job_processor")

class JobProcessor:
    """
    Service de processing de jobs avancé : async, retry, hooks, sécurité, audit, logique métier, observabilité.
    Utilisé pour exécuter les jobs de queue (IA, analytics, Spotify, batch, etc.).
    """
    def __init__(self):
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self.hooks: List[Callable] = []
    def register_handler(self, job_type: str, handler: Callable[Dict[str, Any], Any]):
        self.handlers[job_type] = handler
        logger.info(f"Handler enregistré pour {job_type}")
    def register_hook(self, hook: Callable):
        self.hooks.append(hook)
        logger.info(f"JobProcessor hook enregistré: {hook}")
    def process(self, job: Dict[str, Any], retry: int = 3):
        job_type = job.get("type")
        handler = self.handlers.get(job_type)
        if not handler:
            logger.error(f"Aucun handler pour job type: {job_type}")
            self.audit(job, "no_handler")
            return False
        for attempt in range(1, retry + 1):
            try:
                result = handler(job["payload"])
                logger.info(f"Job {job_type} traité avec succès (tentative {attempt})")
                self.audit(job, "success")
                for hook in self.hooks:
                    hook(job, result, True)
                return True
            except Exception as e:
                logger.error(f"Erreur processing job {job_type} (tentative {attempt}): {e}")
                self.audit(job, f"error_{attempt}")
                for hook in self.hooks:
                    hook(job, str(e), False)
                time.sleep(1)
        return False
    def audit(self, job: Dict[str, Any], status: str):
        logger.info(f"[AUDIT] Job {job.get('type')} status: {status} | job: {job}")
