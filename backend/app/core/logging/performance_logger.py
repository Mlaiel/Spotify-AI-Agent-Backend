"""
Module: performance_logger.py
Description: Logger de performance pour le monitoring industriel (latence, throughput, métriques custom, Prometheus-ready).
"""
import logging
import time
from typing import Callable, Any, Optional

class PerformanceLogger:
    def __init__(self, name: str = "performance_logger"):
        self.logger = logging.getLogger(name)

    def log_latency(self, operation: str, start_time: float, end_time: Optional[float] = None, **kwargs):
        end_time = end_time or time.time()
        latency = end_time - start_time
        self.logger.info(f"[PERF] {operation} latency={latency:.4f}s", extra={"latency": latency, **kwargs})
        return latency

    def log_throughput(self, operation: str, count: int, duration: float, **kwargs):
        throughput = count / duration if duration > 0 else 0
        self.logger.info(f"[PERF] {operation} throughput={throughput:.2f}/s", extra={"throughput": throughput, **kwargs})
        return throughput

    def timeit(self, operation: str):
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                self.log_latency(operation, start, time.time())
                return result
            return wrapper
        return decorator

# Exemple d'utilisation
# perf_logger = PerformanceLogger()
# @perf_logger.timeit("db_query")
# def query(): ...
