"""
Gunicorn-Konfiguration für Spotify AI Agent Backend
--------------------------------------------------
- Optimiert für Security, Performance, Logging, Health, DevOps
- Rollen: Lead Dev, Architecte IA, Backend Senior, Security Specialist, Microservices Architect
"""

import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
accesslog = "-"
errorlog = "-"
loglevel = "info"
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
preload_app = True
reload = False
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
forwarded_allow_ips = "*"
proxy_allow_ips = "*"

# Security
secure_scheme_headers = {"X-Forwarded-Proto": "https"}

# Health-Check
def when_ready(server):
    open("/tmp/gunicorn-ready", "w").close()

def on_exit(server):
    import os
    try:
        os.remove("/tmp/gunicorn-ready")
    except Exception:
        pass
