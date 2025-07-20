"""
Module: async_logger.py
Description: Logger asynchrone industriel pour microservices, compatible FastAPI, Celery, et streaming (Kafka, Redis, etc).
"""
import logging
import asyncio
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from typing import Optional

class AsyncLogger:
    """
    Logger asynchrone pour applications Python modernes (FastAPI, Celery, etc).
    Permet le logging non-bloquant, compatible avec les architectures microservices et le streaming de logs.
    """
    def __init__(self, name: str = "async_logger", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.queue = Queue(-1)
        self.handler = QueueHandler(self.queue)
        self.logger.addHandler(self.handler)
        self.listener = QueueListener(self.queue, *self.logger.handlers)
        self.listener.start()

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, extra=kwargs)

    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, extra=kwargs)

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, extra=kwargs)

    def critical(self, msg: str, **kwargs):
        self.logger.critical(msg, extra=kwargs)

    def stop(self):
        self.listener.stop()

# Exemple d'utilisation
# logger = AsyncLogger()
# logger.info("Service started", service="api", env="prod")
