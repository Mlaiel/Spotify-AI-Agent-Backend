"""
Revision ID: 20250710_02_add_spotify_data_and_collaboration
Revises: 20250710_01_create_user_and_artist_tables
Create Date: 2025-07-10 10:10:00

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

Description: Add Spotify data, AI content, and collaboration tables for advanced analytics and AI-driven features.
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'spotify_data',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('user.id'), nullable=False),
        sa.Column('spotify_id', sa.String(255), nullable=False),
        sa.Column('data_type', sa.String(50), nullable=False),
        sa.Column('data_json', sa.JSON, nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_table(
        'ai_content',)
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('user.id'), nullable=False),
        sa.Column('content_type', sa.String(50), nullable=False),
        sa.Column('payload', sa.JSON, nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(),
    )
    op.create_table(
        'collaboration',)
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('initiator_id', sa.Integer, sa.ForeignKey('user.id'), nullable=False),
        sa.Column('receiver_id', sa.Integer, sa.ForeignKey('user.id'), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('collab_type', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(),
    )
    op.create_index('ix_spotify_data_user_id', 'spotify_data', ['user_id'])
    op.create_index('ix_ai_content_user_id', 'ai_content', ['user_id'])
    op.create_index('ix_collaboration_initiator_id', 'collaboration', ['initiator_id'])
    op.create_index('ix_collaboration_receiver_id', 'collaboration', ['receiver_id'])

def downgrade():
    op.drop_index('ix_collaboration_receiver_id', table_name='collaboration')
    op.drop_index('ix_collaboration_initiator_id', table_name='collaboration')
    op.drop_index('ix_ai_content_user_id', table_name='ai_content')
    op.drop_index('ix_spotify_data_user_id', table_name='spotify_data')
    op.drop_table('collaboration')
    op.drop_table('ai_content')
    op.drop_table('spotify_data')
