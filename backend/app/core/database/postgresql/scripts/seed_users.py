"""
Script de seed pour la table users (exemple IA Spotify)
- Ajoute des utilisateurs de test avec rôles, emails, profils IA
"""

import os
import logging
from faker import Faker
from psycopg2 import connect

DB_NAME = os.getenv("POSTGRES_DB", "spotify_ai_agent")
USER = os.getenv("POSTGRES_USER", "postgres")
PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
HOST = os.getenv("POSTGRES_HOST", "localhost")
PORT = os.getenv("POSTGRES_PORT", "5432")

fake = Faker()

conn = connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
cur = conn.cursor()

for _ in range(20):
    name = fake.name()
    email = fake.email()
    role = fake.random_element(elements=("artist", "manager", "label", "admin")
    cur.execute("INSERT INTO users (name, email, role) VALUES (%s, %s, %s)", (name, email, role)
    logging.info(f"User ajouté: {name} | {email} | {role}")

conn.commit()
cur.close()
conn.close()
print("Seed users terminé.")
