"""
Module: decorators.py
Description: Décorateurs industriels pour la gestion des exceptions, du cache, du logging, de la sécurité, du timing, du retry.
"""
import functools
import time
import logging
from typing import Callable, Any

def exception_handler(logger=None):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                (logger or logging).error(f"Exception in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator

def timing(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"{func.__name__} executed in {elapsed:.4f}s")
        return result
    return wrapper

def retry(retries: int = 3, delay: float = 1.0):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

def retry_on_failure(func: Callable) -> Callable:
    """
    Décorateur industriel pour retry automatique sur échec (Exception).
    Utilise un backoff exponentiel et journalise chaque tentative.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_attempts = 3
        delay = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.warning(f"[retry_on_failure] Tentative {attempt} échouée pour {func.__name__}: {e}")
                if attempt == max_attempts:
                    logging.error(f"[retry_on_failure] Toutes les tentatives échouées pour {func.__name__}")
                    raise
                time.sleep(delay)
                delay *= 2  # backoff exponentiel
    return wrapper

# Exemples d'utilisation
# @exception_handler()
# @timing
# @retry(retries=5)
# @retry_on_failure
