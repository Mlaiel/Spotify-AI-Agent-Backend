#!/usr/bin/env python3
"""
Reindex data from one Elasticsearch index to another for zero-downtime migrations.

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices
"""

import os
import sys
import requests

ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")

if len(sys.argv) < 3:
    print("Usage: python reindex_data.py --source=<old_index> --dest=<new_index>")
    sys.exit(1)

source = None
dest = None
for arg in sys.argv[1:]:
    if arg.startswith("--source="):
        source = arg.split("=", 1)[1]
    elif arg.startswith("--dest="):
        dest = arg.split("=", 1)[1]

if not source or not dest:
    print("Both --source and --dest are required.")
    sys.exit(1)

body = {
    "source": {"index": source},
    "dest": {"index": dest}
}

resp = requests.post(f"{ES_HOST}/_reindex", json=body)
print(f"Reindex {source} -> {dest}: {resp.status_code} {resp.text}")
