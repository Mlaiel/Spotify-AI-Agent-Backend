#!/usr/bin/env python3
"""
Create all Elasticsearch indexes (inkl. Multilingual, Geo, Nested, Advanced) mit Lifecycle, Security, Audit.

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

MAPPINGS = {
    "artists": "../mappings/artists_mapping.json",
    "users": "../mappings/users_mapping.json",
    "playlists": "../mappings/playlists_mapping.json",
    "tracks": "../mappings/tracks_mapping.json",
    "events": "../mappings/events_mapping.json",
    "venues": "../mappings/venues_mapping.json",
    "multilingual_content": "../mappings/multilingual_content_mapping.json",
    "advanced_content": "../mappings/advanced_content_mapping.json"
}

ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")

for index, mapping_file in MAPPINGS.items():
    path = os.path.join(os.path.dirname(__file__), mapping_file)
    if not os.path.exists(path):
        print(f"Mapping file not found: {mapping_file}")
        continue
    with open(path) as f:
        mapping = json.load(f)
    resp = requests.put(f"{ES_HOST}/{index}", json=mapping)
    print(f"Created index {index}: {resp.status_code} {resp.text}")
