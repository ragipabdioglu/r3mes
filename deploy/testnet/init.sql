-- R3MES Testnet Database Initialization
-- PostgreSQL 15

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wallet_address VARCHAR(64) UNIQUE NOT NULL,
    username VARCHAR(50),
    email VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true
);

-- Faucet requests table
CREATE TABLE IF NOT EXISTS faucet_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wallet_address VARCHAR(64) NOT NULL,
    amount BIGINT NOT NULL,
    tx_hash VARCHAR(64),
    status VARCHAR(20) DEFAULT 'pending',
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Mining stats table
CREATE TABLE IF NOT EXISTS mining_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wallet_address VARCHAR(64) NOT NULL,
    hashrate DECIMAL(20, 8) DEFAULT 0,
    total_earnings BIGINT DEFAULT 0,
    gradients_submitted INTEGER DEFAULT 0,
    last_submission TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Node registrations table
CREATE TABLE IF NOT EXISTS node_registrations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_address VARCHAR(64) UNIQUE NOT NULL,
    node_type VARCHAR(20) NOT NULL,
    stake BIGINT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    ip_address INET,
    port INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Training rounds table
CREATE TABLE IF NOT EXISTS training_rounds (
    id SERIAL PRIMARY KEY,
    round_number INTEGER UNIQUE NOT NULL,
    model_version VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    participants INTEGER DEFAULT 0,
    gradients_received INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Gradient submissions table
CREATE TABLE IF NOT EXISTS gradient_submissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    round_id INTEGER REFERENCES training_rounds(id),
    miner_address VARCHAR(64) NOT NULL,
    ipfs_hash VARCHAR(64) NOT NULL,
    gradient_hash VARCHAR(64),
    status VARCHAR(20) DEFAULT 'pending',
    reward BIGINT DEFAULT 0,
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    verified_at TIMESTAMP WITH TIME ZONE
);

-- Transactions table (for indexing)
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tx_hash VARCHAR(64) UNIQUE NOT NULL,
    block_height BIGINT NOT NULL,
    tx_type VARCHAR(50),
    sender VARCHAR(64),
    receiver VARCHAR(64),
    amount BIGINT,
    fee BIGINT,
    status VARCHAR(20),
    memo TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Blocks table (for indexing)
CREATE TABLE IF NOT EXISTS blocks (
    height BIGINT PRIMARY KEY,
    hash VARCHAR(64) UNIQUE NOT NULL,
    proposer VARCHAR(64),
    tx_count INTEGER DEFAULT 0,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_faucet_wallet ON faucet_requests(wallet_address);
CREATE INDEX IF NOT EXISTS idx_faucet_created ON faucet_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_mining_wallet ON mining_stats(wallet_address);
CREATE INDEX IF NOT EXISTS idx_node_type ON node_registrations(node_type);
CREATE INDEX IF NOT EXISTS idx_gradient_round ON gradient_submissions(round_id);
CREATE INDEX IF NOT EXISTS idx_gradient_miner ON gradient_submissions(miner_address);
CREATE INDEX IF NOT EXISTS idx_tx_hash ON transactions(tx_hash);
CREATE INDEX IF NOT EXISTS idx_tx_sender ON transactions(sender);
CREATE INDEX IF NOT EXISTS idx_tx_block ON transactions(block_height);
CREATE INDEX IF NOT EXISTS idx_blocks_timestamp ON blocks(timestamp);

-- Insert initial data
INSERT INTO training_rounds (round_number, model_version, status) 
VALUES (1, 'v1.0.0', 'active')
ON CONFLICT (round_number) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO r3mes;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO r3mes;
