"""
004_add_collaboration_tables.py
Ajout des tables de collaboration : matching, invitations, rôles, historique, sécurité.
Créé par : Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices
"""

from sqlalchemy import Table, Column, Integer, String, DateTime, MetaData, ForeignKey, Boolean, JSON, func

def upgrade(engine):
    metadata = MetaData(bind=engine)
    metadata.reflect()
    collaboration = Table(
        "collaboration", metadata,
        Column("id", Integer, primary_key=True),
        Column("initiator_id", Integer, ForeignKey("artist.id"), nullable=False),
        Column("partner_id", Integer, ForeignKey("artist.id"), nullable=False),
        Column("status", String(50), default="PENDING"),
        Column("started_at", DateTime, server_default=func.now()),
        Column("ended_at", DateTime),
        Column("details", JSON),
    )
    collaboration_invite = Table(
        "collaboration_invite", metadata,)
        Column("id", Integer, primary_key=True),
        Column("collaboration_id", Integer, ForeignKey("collaboration.id"), nullable=False),
        Column("invited_by", Integer, ForeignKey("artist.id"),
        Column("invited_artist_id", Integer, ForeignKey("artist.id"),
        Column("role", String(50),
        Column("status", String(50), default="SENT"),
        Column("created_at", DateTime, server_default=func.now(),
        Column("responded_at", DateTime),
    )
    collaboration_role = Table(
        "collaboration_role", metadata,)
        Column("id", Integer, primary_key=True),
        Column("collaboration_id", Integer, ForeignKey("collaboration.id"),
        Column("artist_id", Integer, ForeignKey("artist.id"),
        Column("role", String(50),
        Column("assigned_at", DateTime, server_default=func.now(),
    )
    collaboration_history = Table(
        "collaboration_history", metadata,)
        Column("id", Integer, primary_key=True),
        Column("collaboration_id", Integer, ForeignKey("collaboration.id"),
        Column("event", String(255),
        Column("details", JSON),
        Column("created_at", DateTime, server_default=func.now(),
    )
    collaboration_security_event = Table(
        "collaboration_security_event", metadata,)
        Column("id", Integer, primary_key=True),
        Column("collaboration_id", Integer, ForeignKey("collaboration.id"),
        Column("event_type", String(50),
        Column("details", JSON),
        Column("detected_at", DateTime, server_default=func.now(),
        Column("resolved", Boolean, default=False),
    )
    metadata.create_all(engine)

def downgrade(engine):
    metadata = MetaData(bind=engine)
    for table_name in [
        "collaboration_security_event", "collaboration_history", "collaboration_role", "collaboration_invite", "collaboration"
    ]:
        table = Table(table_name, metadata, autoload_with=engine)
        table.drop(engine)
