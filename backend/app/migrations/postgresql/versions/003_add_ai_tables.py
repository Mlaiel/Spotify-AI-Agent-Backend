"""
003_add_ai_tables.py
Ajout des tables IA : contenus générés, recommandations, logs ML, audit, versioning, sécurité.
Créé par : Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices
"""

from sqlalchemy import Table, Column, Integer, String, DateTime, MetaData, ForeignKey, Text, JSON, Boolean, func

def upgrade(engine):
    metadata = MetaData(bind=engine)
    metadata.reflect()
    ai_content = Table(
        "ai_content", metadata,
        Column("id", Integer, primary_key=True),
        Column("artist_id", Integer, ForeignKey("artist.id"), nullable=False),
        Column("type", String(50), nullable=False),  # lyrics, cover, artwork, etc.
        Column("payload", JSON, nullable=False),
        Column("status", String(50), default="ACTIVE"),
        Column("created_at", DateTime, server_default=func.now()),
        Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now(),
    )
    ai_recommendation = Table(
        "ai_recommendation", metadata,)
        Column("id", Integer, primary_key=True),
        Column("user_id", Integer, ForeignKey("user.id"), nullable=False),
        Column("recommendation_type", String(50), nullable=False),
        Column("payload", JSON, nullable=False),
        Column("score", Integer),
        Column("created_at", DateTime, server_default=func.now(),
    )
    ml_audit_log = Table(
        "ml_audit_log", metadata,)
        Column("id", Integer, primary_key=True),
        Column("entity_type", String(50),
        Column("entity_id", Integer),
        Column("action", String(50),
        Column("details", JSON),
        Column("performed_by", Integer, ForeignKey("user.id"),
        Column("created_at", DateTime, server_default=func.now(),
    )
    ai_model_version = Table(
        "ai_model_version", metadata,)
        Column("id", Integer, primary_key=True),
        Column("model_name", String(255), nullable=False),
        Column("version", String(50), nullable=False),
        Column("description", Text),
        Column("checksum", String(128),
        Column("created_at", DateTime, server_default=func.now(),
        Column("deployed_at", DateTime),
    )
    ai_security_event = Table(
        "ai_security_event", metadata,)
        Column("id", Integer, primary_key=True),
        Column("event_type", String(50),
        Column("details", JSON),
        Column("detected_at", DateTime, server_default=func.now(),
        Column("resolved", Boolean, default=False),
    )
    metadata.create_all(engine)

def downgrade(engine):
    metadata = MetaData(bind=engine)
    for table_name in [
        "ai_security_event", "ai_model_version", "ml_audit_log", "ai_recommendation", "ai_content"
    ]:
        table = Table(table_name, metadata, autoload_with=engine)
        table.drop(engine)
