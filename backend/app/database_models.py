"""
SQLAlchemy ORM Models for Database Migrations

These models are used by Alembic for database schema versioning and migrations.
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, 
    ForeignKey, BigInteger, DECIMAL, JSON, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    """User model for wallet addresses and credits."""
    __tablename__ = 'users'
    
    wallet_address = Column(String(255), primary_key=True)
    credits = Column(Float, default=0.0, nullable=False)
    is_miner = Column(Boolean, default=False, nullable=False)
    last_mining_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    mining_stats = relationship("MiningStats", back_populates="user", cascade="all, delete-orphan")
    earnings_history = relationship("EarningsHistory", back_populates="user", cascade="all, delete-orphan")
    hashrate_history = relationship("HashrateHistory", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_users_is_miner', 'is_miner'),
        Index('idx_users_created_at', 'created_at'),
    )


class MiningStats(Base):
    """Mining statistics model."""
    __tablename__ = 'mining_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_address = Column(String(255), ForeignKey('users.wallet_address', ondelete='CASCADE'), nullable=False)
    hashrate = Column(Float, default=0.0, nullable=False)
    gpu_temperature = Column(Float, default=0.0, nullable=False)
    blocks_found = Column(Integer, default=0, nullable=False)
    uptime_percentage = Column(Float, default=0.0, nullable=False)
    network_difficulty = Column(Float, default=0.0, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    user = relationship("User", back_populates="mining_stats")
    
    __table_args__ = (
        Index('idx_mining_stats_wallet', 'wallet_address'),
        Index('idx_mining_stats_recorded_at', 'recorded_at'),
        Index('idx_mining_stats_wallet_recorded', 'wallet_address', 'recorded_at'),
    )


class EarningsHistory(Base):
    """Earnings history model."""
    __tablename__ = 'earnings_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_address = Column(String(255), ForeignKey('users.wallet_address', ondelete='CASCADE'), nullable=False)
    earnings = Column(Float, default=0.0, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    user = relationship("User", back_populates="earnings_history")
    
    __table_args__ = (
        Index('idx_earnings_history_wallet', 'wallet_address'),
        Index('idx_earnings_history_recorded_at', 'recorded_at'),
        Index('idx_earnings_history_wallet_recorded', 'wallet_address', 'recorded_at'),
    )


class HashrateHistory(Base):
    """Hashrate history model."""
    __tablename__ = 'hashrate_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_address = Column(String(255), ForeignKey('users.wallet_address', ondelete='CASCADE'), nullable=False)
    hashrate = Column(Float, default=0.0, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    user = relationship("User", back_populates="hashrate_history")
    
    __table_args__ = (
        Index('idx_hashrate_history_wallet', 'wallet_address'),
        Index('idx_hashrate_history_recorded_at', 'recorded_at'),
        Index('idx_hashrate_history_wallet_recorded', 'wallet_address', 'recorded_at'),
    )


class APIKey(Base):
    """API key model."""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    api_key_hash = Column(String(64), unique=True, nullable=False)
    wallet_address = Column(String(255), ForeignKey('users.wallet_address', ondelete='CASCADE'), nullable=False)
    name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    
    # Relationship
    user = relationship("User", back_populates="api_keys")
    
    __table_args__ = (
        Index('idx_api_key_hash', 'api_key_hash'),
        Index('idx_api_keys_wallet', 'wallet_address'),
        Index('idx_api_keys_is_active', 'is_active'),
        Index('idx_api_keys_expires_at', 'expires_at'),
        Index('idx_api_keys_wallet_active', 'wallet_address', 'is_active'),
    )


class BlockchainEvent(Base):
    """Blockchain events model (for indexer)."""
    __tablename__ = 'blockchain_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(100), nullable=False)
    block_height = Column(BigInteger, nullable=False)
    block_hash = Column(String(64), nullable=True)
    tx_hash = Column(String(64), nullable=True)
    miner_address = Column(String(255), nullable=True)
    validator_address = Column(String(255), nullable=True)
    pool_id = Column(BigInteger, nullable=True)
    chunk_id = Column(BigInteger, nullable=True)
    gradient_hash = Column(String(128), nullable=True)
    gradient_ipfs_hash = Column(String(128), nullable=True)
    amount = Column(DECIMAL(20, 8), nullable=True)
    event_data = Column(JSON, nullable=True)
    indexed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_event_type', 'event_type'),
        Index('idx_block_height', 'block_height'),
        Index('idx_miner_address', 'miner_address'),
        Index('idx_indexed_at', 'indexed_at'),
        Index('idx_blockchain_events_type_height', 'event_type', 'block_height'),
    )


class NetworkSnapshot(Base):
    """Network snapshots model (for indexer)."""
    __tablename__ = 'network_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    block_height = Column(BigInteger, nullable=False)
    snapshot_date = Column(DateTime, nullable=False)
    total_miners = Column(Integer, default=0, nullable=False)
    total_validators = Column(Integer, default=0, nullable=False)
    total_stake = Column(DECIMAL(20, 8), default=0.0, nullable=False)
    total_gradients = Column(BigInteger, default=0, nullable=False)
    total_aggregations = Column(BigInteger, default=0, nullable=False)
    network_hashrate = Column(DECIMAL(20, 2), default=0.0, nullable=False)
    snapshot_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        UniqueConstraint('block_height', 'snapshot_date', name='uq_network_snapshots_block_date'),
        Index('idx_snapshot_date', 'snapshot_date'),
        Index('idx_snapshots_block_height', 'block_height'),
    )


class IndexerState(Base):
    """Indexer state model (tracks last indexed block)."""
    __tablename__ = 'indexer_state'
    
    id = Column(Integer, primary_key=True, default=1)
    last_indexed_height = Column(BigInteger, default=0, nullable=False)
    last_indexed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        # Ensure only one row exists
        # This is enforced at application level, not at database level
    )


class LoRARegistry(Base):
    """LoRA registry model (stores LoRA adapter metadata)."""
    __tablename__ = 'lora_registry'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    ipfs_hash = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True)
    version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    __table_args__ = (
        Index('idx_lora_registry_name', 'name'),
        Index('idx_lora_registry_category', 'category'),
        Index('idx_lora_registry_is_active', 'is_active'),
    )


class ServingNode(Base):
    """Serving node model (tracks miners acting as serving nodes)."""
    __tablename__ = 'serving_nodes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_address = Column(String(255), unique=True, nullable=False)
    endpoint_url = Column(String(500), nullable=False)
    available_lora_list = Column(JSON, nullable=False)  # List of LoRA adapter names
    status = Column(String(50), default='active', nullable=False)  # active, idle, busy, offline
    last_heartbeat = Column(DateTime, nullable=False)
    current_load = Column(Integer, default=0, nullable=False)  # Current number of active requests
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_serving_nodes_wallet', 'wallet_address'),
        Index('idx_serving_nodes_status', 'status'),
        Index('idx_serving_nodes_last_heartbeat', 'last_heartbeat'),
        Index('idx_serving_nodes_status_heartbeat', 'status', 'last_heartbeat'),
    )

