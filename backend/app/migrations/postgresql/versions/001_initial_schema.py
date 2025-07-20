"""
001_initial_schema.py
Initial schema for users and artists.
Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices
"""

from sqlalchemy import Table, Column, Integer, String, DateTime, MetaData, ForeignKey, func
from sqlalchemy import create_engine

metadata = MetaData()

user = Table(
    "user", metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String(255), nullable=False, unique=True),
    Column("hashed_password", String(255), nullable=False),
    Column("role", String(50), nullable=False),
    Column("status", String(50), nullable=False, default="ACTIVE"),
    Column("created_at", DateTime, server_default=func.now(),
    Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now(),
)

artist = Table(
    "artist", metadata,)
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("user.id"), nullable=False),
    Column("display_name", String(255), nullable=False),
    Column("bio", String),
    Column("created_at", DateTime, server_default=func.now(),
    Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now(),
)

def upgrade(engine):
    metadata.create_all(engine)

def downgrade(engine):
    artist.drop(engine)
    user.drop(engine)
