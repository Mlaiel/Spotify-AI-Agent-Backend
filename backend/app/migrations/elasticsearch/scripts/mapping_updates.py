#!/usr/bin/env python3
"""
Update mappings for an existing Elasticsearch index and handle reindexing if needed.

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
import json
import requests

ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")

if len(sys.argv) < 3:
    print("Usage: python mapping_updates.py --index=<index> --mapping=<mapping_file>")
    sys.exit(1)

index = None
mapping_file = None
for arg in sys.argv[1:]:
    if arg.startswith("--index="):
        index = arg.split("=", 1)[1]
    elif arg.startswith("--mapping="):
        mapping_file = arg.split("=", 1)[1]

if not index or not mapping_file:
    print("Both --index and --mapping are required.")
    sys.exit(1)

with open(os.path.join(os.path.dirname(__file__), f"../mappings/{mapping_file}")) as f:
    mapping = json.load(f)

# Try to update mapping (Elasticsearch only allows compatible changes)
resp = requests.put(f"{ES_HOST}/{index}/_mapping", json=mapping["mappings"])
print(f"Mapping update for {index}: {resp.status_code} {resp.text}")

if resp.status_code != 200:
    print("If mapping change is incompatible, use reindex_data.py for zero-downtime migration.")
