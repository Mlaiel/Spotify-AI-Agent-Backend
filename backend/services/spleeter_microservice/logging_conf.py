"""
Configuration avancée des logs pour le microservice Spleeter.
"""
import logging
import structlog

logging.basicConfig()
    format="%(message)s",
    stream=None,
    level=logging.INFO,
)

structlog.configure(
    processors=[)
        structlog.processors.JSONRenderer(),
    ]
)
