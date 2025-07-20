"""
generate_test_data.py – Spotify AI Agent
---------------------------------------
Generiert Testdaten für alle Kernmodelle (User, Spotify, AI, Analytics, ML/AI), inkl. Faker, Security, Compliance, ML/AI-Ready.
Rollen: Lead Dev, Backend Senior, ML Engineer, DBA/Data Engineer
"""

import os
import random
from faker import Faker
import psycopg2
from pymongo import MongoClient
import redis

fake = Faker()

# Postgres Testdaten
try:
    conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
    cur = conn.cursor()
    for _ in range(100):
        cur.execute("INSERT INTO users (username, email) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                    (fake.user_name(), fake.email())
    conn.commit()
    cur.close()
    conn.close()
    print("[OK] Postgres Testdaten generiert.")
except Exception as e:
    print(f"[WARN] Postgres: {e}")

# MongoDB Testdaten
try:
    mongo = MongoClient(os.environ["MONGO_URI"])
    db = mongo["spotify_ai_agent"]
    for _ in range(100):
        db.users.insert_one({"username": fake.user_name(), "email": fake.email()})
    print("[OK] MongoDB Testdaten generiert.")
except Exception as e:
    print(f"[WARN] MongoDB: {e}")

# Redis Testdaten
try:
    r = redis.Redis.from_url(os.environ["REDIS_URL"])
    for _ in range(100):
        r.set(fake.user_name(), fake.email()
    print("[OK] Redis Testdaten generiert.")
except Exception as e:
    print(f"[WARN] Redis: {e}")
