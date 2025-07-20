"""
Script de seed pour la table music (exemple IA Spotify)
- Ajoute des morceaux, artistes, genres, durées, popularité
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

for _ in range(50):
    title = fake.sentence(nb_words=3)
    artist = fake.name()
    genre = choice(["pop", "rock", "rap", "jazz", "electro", "classique"])
    duration = randint(120, 420)
    popularity = randint(0, 100)
    cur.execute("INSERT INTO music (title, artist, genre, duration, popularity) VALUES (%s, %s, %s, %s, %s)", (title, artist, genre, duration, popularity)
    logging.info(f"Music ajouté: {title} | {artist} | {genre} | {duration}s | {popularity}")

conn.commit()
cur.close()
conn.close()
print("Seed music terminé.")
