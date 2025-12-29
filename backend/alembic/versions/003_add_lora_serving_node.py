"""Add LoRA registry and serving nodes

Revision ID: 003_add_lora_serving_node
Revises: 002_add_indexes
Create Date: 2025-01-15 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003_add_lora_serving_node'
down_revision = '002_add_indexes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # LoRA Registry table
    op.create_table(
        'lora_registry',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('ipfs_hash', sa.String(length=128), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('version', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    # Create indexes for lora_registry
    op.create_index('idx_lora_registry_name', 'lora_registry', ['name'], unique=False)
    op.create_index('idx_lora_registry_category', 'lora_registry', ['category'], unique=False)
    op.create_index('idx_lora_registry_is_active', 'lora_registry', ['is_active'], unique=False)
    
    # Serving Nodes table
    op.create_table(
        'serving_nodes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('wallet_address', sa.String(length=255), nullable=False),
        sa.Column('endpoint_url', sa.String(length=500), nullable=False),
        sa.Column('available_lora_list', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='active'),
        sa.Column('last_heartbeat', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('current_load', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('wallet_address')
    )
    
    # Create indexes for serving_nodes
    op.create_index('idx_serving_nodes_wallet', 'serving_nodes', ['wallet_address'], unique=False)
    op.create_index('idx_serving_nodes_status', 'serving_nodes', ['status'], unique=False)
    op.create_index('idx_serving_nodes_last_heartbeat', 'serving_nodes', ['last_heartbeat'], unique=False)
    op.create_index('idx_serving_nodes_status_heartbeat', 'serving_nodes', ['status', 'last_heartbeat'], unique=False)


def downgrade() -> None:
    # Drop indexes first
    op.drop_index('idx_serving_nodes_status_heartbeat', table_name='serving_nodes')
    op.drop_index('idx_serving_nodes_last_heartbeat', table_name='serving_nodes')
    op.drop_index('idx_serving_nodes_status', table_name='serving_nodes')
    op.drop_index('idx_serving_nodes_wallet', table_name='serving_nodes')
    op.drop_index('idx_lora_registry_is_active', table_name='lora_registry')
    op.drop_index('idx_lora_registry_category', table_name='lora_registry')
    op.drop_index('idx_lora_registry_name', table_name='lora_registry')
    
    # Drop tables
    op.drop_table('serving_nodes')
    op.drop_table('lora_registry')

