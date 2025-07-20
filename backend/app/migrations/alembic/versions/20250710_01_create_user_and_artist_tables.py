"""
Revision ID: 20250710_01_create_user_and_artist_tables
Revises: None
Create Date: 2025-07-10 10:00:00

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

Description: Initial schema for user and artist management, including roles and permissions.
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'user',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('role', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='ACTIVE'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(),
    )
    op.create_table(
        'artist',)
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('user.id'), nullable=False),
        sa.Column('display_name', sa.String(255), nullable=False),
        sa.Column('bio', sa.Text),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(),
    )
    op.create_index('ix_user_email', 'user', ['email'], unique=True)
    op.create_index('ix_artist_user_id', 'artist', ['user_id'])

def downgrade():
    op.drop_index('ix_artist_user_id', table_name='artist')
    op.drop_index('ix_user_email', table_name='user')
    op.drop_table('artist')
    op.drop_table('user')
