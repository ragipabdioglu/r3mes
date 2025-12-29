"""Initial migration

Revision ID: 001_initial
Revises: 
Create Date: 2025-12-24 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Users table
    op.create_table(
        'users',
        sa.Column('wallet_address', sa.String(length=255), nullable=False),
        sa.Column('credits', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('is_miner', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('last_mining_time', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('wallet_address')
    )
    
    # Mining stats table
    op.create_table(
        'mining_stats',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('wallet_address', sa.String(length=255), nullable=False),
        sa.Column('hashrate', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('gpu_temperature', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('blocks_found', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('uptime_percentage', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('network_difficulty', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('recorded_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['wallet_address'], ['users.wallet_address'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Earnings history table
    op.create_table(
        'earnings_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('wallet_address', sa.String(length=255), nullable=False),
        sa.Column('earnings', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('recorded_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['wallet_address'], ['users.wallet_address'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Hashrate history table
    op.create_table(
        'hashrate_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('wallet_address', sa.String(length=255), nullable=False),
        sa.Column('hashrate', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('recorded_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['wallet_address'], ['users.wallet_address'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # API Keys table
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('api_key_hash', sa.String(length=64), nullable=False),
        sa.Column('wallet_address', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('last_used', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['wallet_address'], ['users.wallet_address'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('api_key_hash')
    )
    
    # Blockchain events table (for indexer)
    op.create_table(
        'blockchain_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('block_height', sa.BigInteger(), nullable=False),
        sa.Column('block_hash', sa.String(length=64), nullable=True),
        sa.Column('tx_hash', sa.String(length=64), nullable=True),
        sa.Column('miner_address', sa.String(length=255), nullable=True),
        sa.Column('validator_address', sa.String(length=255), nullable=True),
        sa.Column('pool_id', sa.BigInteger(), nullable=True),
        sa.Column('chunk_id', sa.BigInteger(), nullable=True),
        sa.Column('gradient_hash', sa.String(length=128), nullable=True),
        sa.Column('gradient_ipfs_hash', sa.String(length=128), nullable=True),
        sa.Column('amount', sa.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('event_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('indexed_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Network snapshots table (for indexer)
    op.create_table(
        'network_snapshots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('block_height', sa.BigInteger(), nullable=False),
        sa.Column('snapshot_date', sa.DateTime(), nullable=False),
        sa.Column('total_miners', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_validators', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_stake', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('total_gradients', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('total_aggregations', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('network_hashrate', sa.DECIMAL(precision=20, scale=2), nullable=False, server_default='0.0'),
        sa.Column('snapshot_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('block_height', 'snapshot_date', name='uq_network_snapshots_block_date')
    )
    
    # Indexer state table
    op.create_table(
        'indexer_state',
        sa.Column('id', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('last_indexed_height', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('last_indexed_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Initialize indexer state
    op.execute("""
        INSERT INTO indexer_state (id, last_indexed_height, last_indexed_at)
        VALUES (1, 0, CURRENT_TIMESTAMP)
        ON CONFLICT (id) DO NOTHING
    """)


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign key constraints)
    op.drop_table('indexer_state')
    op.drop_table('network_snapshots')
    op.drop_table('blockchain_events')
    op.drop_table('api_keys')
    op.drop_table('hashrate_history')
    op.drop_table('earnings_history')
    op.drop_table('mining_stats')
    op.drop_table('users')

