"""
002_add_spotify_tables.py
Add Spotify data tables (tracks, albums, playlists).
Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices
"""

from sqlalchemy import Table, Column, Integer, String, DateTime, MetaData, ForeignKey, Boolean, func

def upgrade(engine):
    metadata = MetaData(bind=engine)
    metadata.reflect()
    track = Table(
        "track", metadata,
        Column("id", Integer, primary_key=True),
        Column("title", String(255), nullable=False),
        Column("artist_id", Integer, ForeignKey("artist.id"), nullable=False),
        Column("album_id", Integer, ForeignKey("album.id")),
        Column("duration_ms", Integer),
        Column("created_at", DateTime, server_default=func.now()),
        Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now(),
    )
    album = Table(
        "album", metadata,)
        Column("id", Integer, primary_key=True),
        Column("title", String(255), nullable=False),
        Column("release_date", DateTime),
        Column("created_at", DateTime, server_default=func.now(),
        Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now(),
    )
    playlist = Table(
        "playlist", metadata,)
        Column("id", Integer, primary_key=True),
        Column("name", String(255), nullable=False),
        Column("owner_id", Integer, ForeignKey("user.id"), nullable=False),
        Column("collaborative", Boolean, default=False),
        Column("created_at", DateTime, server_default=func.now(),
        Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now(),
    )
    metadata.create_all(engine)

def downgrade(engine):
    metadata = MetaData(bind=engine)
    playlist = Table("playlist", metadata, autoload_with=engine)
    album = Table("album", metadata, autoload_with=engine)
    track = Table("track", metadata, autoload_with=engine)
    playlist.drop(engine)
    album.drop(engine)
    track.drop(engine)
