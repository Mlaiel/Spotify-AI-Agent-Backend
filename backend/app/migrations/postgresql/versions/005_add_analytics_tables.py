"""
005_add_analytics_tables.py
Ajout des tables analytics : event logs, tracking, audit, sécurité, index, partitionnement.
Créé par : Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices
"""

from sqlalchemy import Table, Column, Integer, String, DateTime, MetaData, ForeignKey, JSON, Boolean, func

def upgrade(engine):
    metadata = MetaData(bind=engine)
    metadata.reflect()
    analytics_event = Table(
        "analytics_event", metadata,
        Column("id", Integer, primary_key=True),
        Column("user_id", Integer, ForeignKey("user.id")),
        Column("artist_id", Integer, ForeignKey("artist.id")),
        Column("event_type", String(100), nullable=False),
        Column("payload", JSON),
        Column("created_at", DateTime, server_default=func.now()),
        Column("ip_address", String(45),
        Column("user_agent", String(255),
    )
    analytics_audit_log = Table(
        "analytics_audit_log", metadata,)
        Column("id", Integer, primary_key=True),
        Column("event_id", Integer, ForeignKey("analytics_event.id"),
        Column("action", String(50),
        Column("details", JSON),
        Column("performed_by", Integer, ForeignKey("user.id"),
        Column("created_at", DateTime, server_default=func.now(),
    )
    analytics_security_event = Table(
        "analytics_security_event", metadata,)
        Column("id", Integer, primary_key=True),
        Column("event_id", Integer, ForeignKey("analytics_event.id"),
        Column("event_type", String(50),
        Column("details", JSON),
        Column("detected_at", DateTime, server_default=func.now(),
        Column("resolved", Boolean, default=False),
    )
    metadata.create_all(engine)

def downgrade(engine):
    metadata = MetaData(bind=engine)
    for table_name in [
        "analytics_security_event", "analytics_audit_log", "analytics_event"
    ]:
        table = Table(table_name, metadata, autoload_with=engine)
        table.drop(engine)
