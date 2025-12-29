"""Add performance indexes

Revision ID: 002_add_indexes
Revises: 001_initial
Create Date: 2025-12-24 12:01:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002_add_indexes'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Users table indexes
    op.create_index('idx_users_is_miner', 'users', ['is_miner'], unique=False)
    op.create_index('idx_users_created_at', 'users', ['created_at'], unique=False)
    
    # Mining stats indexes
    op.create_index('idx_mining_stats_wallet', 'mining_stats', ['wallet_address'], unique=False)
    op.create_index('idx_mining_stats_recorded_at', 'mining_stats', ['recorded_at'], unique=False)
    op.create_index('idx_mining_stats_wallet_recorded', 'mining_stats', ['wallet_address', sa.text('recorded_at DESC')], unique=False)
    
    # Earnings history indexes
    op.create_index('idx_earnings_history_wallet', 'earnings_history', ['wallet_address'], unique=False)
    op.create_index('idx_earnings_history_recorded_at', 'earnings_history', ['recorded_at'], unique=False)
    op.create_index('idx_earnings_history_wallet_recorded', 'earnings_history', ['wallet_address', sa.text('recorded_at DESC')], unique=False)
    
    # Hashrate history indexes
    op.create_index('idx_hashrate_history_wallet', 'hashrate_history', ['wallet_address'], unique=False)
    op.create_index('idx_hashrate_history_recorded_at', 'hashrate_history', ['recorded_at'], unique=False)
    op.create_index('idx_hashrate_history_wallet_recorded', 'hashrate_history', ['wallet_address', sa.text('recorded_at DESC')], unique=False)
    
    # API Keys indexes
    op.create_index('idx_api_key_hash', 'api_keys', ['api_key_hash'], unique=False)
    op.create_index('idx_api_keys_wallet', 'api_keys', ['wallet_address'], unique=False)
    op.create_index('idx_api_keys_is_active', 'api_keys', ['is_active'], unique=False)
    op.create_index('idx_api_keys_expires_at', 'api_keys', ['expires_at'], unique=False)
    op.create_index('idx_api_keys_wallet_active', 'api_keys', ['wallet_address', 'is_active'], unique=False)
    
    # Blockchain events indexes (if not already created in initial migration)
    # These might already exist, so we use IF NOT EXISTS equivalent by checking
    try:
        op.create_index('idx_event_type', 'blockchain_events', ['event_type'], unique=False)
    except Exception:
        pass  # Index might already exist
    
    try:
        op.create_index('idx_block_height', 'blockchain_events', ['block_height'], unique=False)
    except Exception:
        pass
    
    try:
        op.create_index('idx_miner_address', 'blockchain_events', ['miner_address'], unique=False)
    except Exception:
        pass
    
    try:
        op.create_index('idx_indexed_at', 'blockchain_events', ['indexed_at'], unique=False)
    except Exception:
        pass
    
    op.create_index('idx_blockchain_events_type_height', 'blockchain_events', ['event_type', sa.text('block_height DESC')], unique=False)
    
    # Network snapshots indexes
    op.create_index('idx_snapshot_date', 'network_snapshots', ['snapshot_date'], unique=False)
    op.create_index('idx_snapshots_block_height', 'network_snapshots', ['block_height'], unique=False)


def downgrade() -> None:
    # Drop indexes in reverse order
    op.drop_index('idx_snapshots_block_height', table_name='network_snapshots')
    op.drop_index('idx_snapshot_date', table_name='network_snapshots')
    op.drop_index('idx_blockchain_events_type_height', table_name='blockchain_events')
    op.drop_index('idx_indexed_at', table_name='blockchain_events')
    op.drop_index('idx_miner_address', table_name='blockchain_events')
    op.drop_index('idx_block_height', table_name='blockchain_events')
    op.drop_index('idx_event_type', table_name='blockchain_events')
    op.drop_index('idx_api_keys_wallet_active', table_name='api_keys')
    op.drop_index('idx_api_keys_expires_at', table_name='api_keys')
    op.drop_index('idx_api_keys_is_active', table_name='api_keys')
    op.drop_index('idx_api_keys_wallet', table_name='api_keys')
    op.drop_index('idx_api_key_hash', table_name='api_keys')
    op.drop_index('idx_hashrate_history_wallet_recorded', table_name='hashrate_history')
    op.drop_index('idx_hashrate_history_recorded_at', table_name='hashrate_history')
    op.drop_index('idx_hashrate_history_wallet', table_name='hashrate_history')
    op.drop_index('idx_earnings_history_wallet_recorded', table_name='earnings_history')
    op.drop_index('idx_earnings_history_recorded_at', table_name='earnings_history')
    op.drop_index('idx_earnings_history_wallet', table_name='earnings_history')
    op.drop_index('idx_mining_stats_wallet_recorded', table_name='mining_stats')
    op.drop_index('idx_mining_stats_recorded_at', table_name='mining_stats')
    op.drop_index('idx_mining_stats_wallet', table_name='mining_stats')
    op.drop_index('idx_users_created_at', table_name='users')
    op.drop_index('idx_users_is_miner', table_name='users')

