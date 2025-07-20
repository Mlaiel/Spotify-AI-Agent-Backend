"""
Revision ID: 20250710_03_add_analytics_audit_security
Revises: 20250710_02_add_spotify_data_and_collaboration
Create Date: 2025-07-10 10:20:00

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

Description: Add analytics, audit log, security event, and versioning tables for advanced monitoring, compliance, and business intelligence.
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'analytics_event',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('user.id')),
        sa.Column('artist_id', sa.Integer, sa.ForeignKey('artist.id')),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('payload', sa.JSON),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('ip_address', sa.String(45),
        sa.Column('user_agent', sa.String(255),
    )
    op.create_table(
        'audit_log',)
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('entity_type', sa.String(50),
        sa.Column('entity_id', sa.Integer),
        sa.Column('action', sa.String(50),
        sa.Column('details', sa.JSON),
        sa.Column('performed_by', sa.Integer, sa.ForeignKey('user.id'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(),
    )
    op.create_table(
        'security_event',)
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('event_type', sa.String(50),
        sa.Column('details', sa.JSON),
        sa.Column('detected_at', sa.DateTime, server_default=sa.func.now(),
        sa.Column('resolved', sa.Boolean, default=False),
    )
    op.create_table(
        'model_version',)
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('model_name', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('checksum', sa.String(128),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(),
        sa.Column('deployed_at', sa.DateTime),
    )
    op.create_index('ix_analytics_event_user_id', 'analytics_event', ['user_id'])
    op.create_index('ix_audit_log_entity_id', 'audit_log', ['entity_id'])
    op.create_index('ix_security_event_type', 'security_event', ['event_type'])
    op.create_index('ix_model_version_name', 'model_version', ['model_name'])

def downgrade():
    op.drop_index('ix_model_version_name', table_name='model_version')
    op.drop_index('ix_security_event_type', table_name='security_event')
    op.drop_index('ix_audit_log_entity_id', table_name='audit_log')
    op.drop_index('ix_analytics_event_user_id', table_name='analytics_event')
    op.drop_table('model_version')
    op.drop_table('security_event')
    op.drop_table('audit_log')
    op.drop_table('analytics_event')
