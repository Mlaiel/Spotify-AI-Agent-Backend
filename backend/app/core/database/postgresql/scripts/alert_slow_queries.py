"""
Script d’alerte requêtes lentes PostgreSQL
- Détecte et alerte sur les requêtes > 5min
"""

import os
import psycopg2
from datetime import datetime

DB_NAME = os.getenv("POSTGRES_DB", "spotify_ai_agent")
USER = os.getenv("POSTGRES_USER", "postgres")
PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
HOST = os.getenv("POSTGRES_HOST", "localhost")
PORT = os.getenv("POSTGRES_PORT", "5432")

conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
cur = conn.cursor()

cur.execute("SELECT pid, query, state, now() - query_start as duration FROM pg_stat_activity WHERE state='active' AND now() - query_start > interval '5 minutes';")
slow = cur.fetchall()

if slow:
    with open(f"alert_slow_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", "w") as f:
        for row in slow:
            f.write(str(row) + "\n")
    print(f"Alerte : {len(slow)} requêtes lentes détectées !")
else:
    print("Aucune requête lente détectée.")

cur.close()
conn.close()
