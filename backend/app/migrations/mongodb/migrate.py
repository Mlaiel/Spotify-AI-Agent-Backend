#!/usr/bin/env python3
"""
MongoDB migration runner for Spotify AI Agent

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

This script runs all migration scripts in the correct order for MongoDB.
"""

import os
import subprocess
import sys

MIGRATIONS = [
    "001_create_collections.js",
    "002_add_indexes.js",
    "003_data_migration.js",
    "004_schema_updates.js"
]

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "migrations")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/spotify_ai_agent")

for migration in MIGRATIONS:
    script_path = os.path.join(MIGRATIONS_DIR, migration)
    print(f"Running migration: {migration}")
    result = subprocess.run([
        "mongo",
        MONGO_URI,
        script_path
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error running {migration}: {result.stderr}")
        sys.exit(result.returncode)
print("All migrations applied successfully.")
