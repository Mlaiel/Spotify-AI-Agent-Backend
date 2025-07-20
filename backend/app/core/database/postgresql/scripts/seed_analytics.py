"""
Script de seed pour la table analytics (exemple IA Spotify)
- Ajoute des données analytiques simulées pour tests IA
"""

import os
import logging
from faker import Faker
from psycopg2 import connect
from random import randint, choice

DB_NAME = os.getenv("POSTGRES_DB", "spotify_ai_agent")
USER = os.getenv("POSTGRES_USER", "postgres")
PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
HOST = os.getenv("POSTGRES_HOST", "localhost")
PORT = os.getenv("POSTGRES_PORT", "5432")

fake = Faker()
conn = connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
cur = conn.cursor()

for _ in range(100):
    user_id = randint(1, 20)
    streams = randint(100, 10000)
    country = fake.country_code()
    device = choice(["mobile", "desktop", "tablet"])
    cur.execute("INSERT INTO analytics (user_id, streams, country, device) VALUES (%s, %s, %s, %s)", (user_id, streams, country, device)
    logging.info(f"Analytics ajouté: user={user_id} | streams={streams} | {country} | {device}")

conn.commit()
cur.close()
conn.close()
print("Seed analytics terminé.")
