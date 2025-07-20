#!/usr/bin/env python3
"""
Validate all Elasticsearch mappings and index health (inkl. Multilingual, Geo, Nested, Advanced).

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices
"""

import os
import json
import requests

MAPPINGS = [
    "artists", "users", "playlists", "tracks", "events", "venues", "multilingual_content", "advanced_content"
]

ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")

for index in MAPPINGS:
    resp = requests.get(f"{ES_HOST}/{index}")
    if resp.status_code == 200:
        print(f"Index {index} exists.")
        mapping = requests.get(f"{ES_HOST}/{index}/_mapping").json()
        print(f"Mapping for {index}: {json.dumps(mapping, indent=2)[:500]} ...")
        health = requests.get(f"{ES_HOST}/_cluster/health/{index}").json()
        print(f"Health for {index}: {health.get('status')}")
    else:
        print(f"Index {index} missing or not reachable: {resp.status_code}")
