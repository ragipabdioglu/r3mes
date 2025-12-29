"""
Deterministic Shard Assignment for R3MES Training

Implements deterministic shard assignment using:
(wallet_address + block_hash + round_id) % total_shards

This ensures:
- Same miner gets same shard within a round (stable)
- Different miners get different shards (distributed)
- Different rounds can have different assignments (flexible)
"""

import hashlib
from typing import Optional


def calculate_shard_id(
    miner_address: str,
    block_hash: str,
    training_round_id: int,
    total_shards: int = 100,
) -> int:
    """
    Calculate deterministic shard ID for a miner in a training round.
    
    Args:
        miner_address: Miner's wallet address (bech32 format)
        block_hash: Block hash from blockchain (hex string)
        training_round_id: Current training round ID
        total_shards: Total number of shards (default: 100)
    
    Returns:
        Shard ID in range [0, total_shards)
    """
    if total_shards == 0:
        raise ValueError("total_shards cannot be zero")
    
    # Create deterministic input: miner_address + block_hash + round_id
    input_str = f"{miner_address}|{block_hash}|{training_round_id}"
    
    # Hash the input
    hash_bytes = hashlib.sha256(input_str.encode()).digest()
    
    # Convert first 8 bytes to integer
    shard_id = int.from_bytes(hash_bytes[:8], byteorder='big')
    
    # Modulo to get shard ID in range [0, total_shards)
    shard_id = shard_id % total_shards
    
    return shard_id


def verify_shard_assignment(
    miner_address: str,
    block_hash: str,
    training_round_id: int,
    claimed_shard_id: int,
    total_shards: int = 100,
) -> bool:
    """
    Verify that a miner's shard assignment is correct.
    
    Args:
        miner_address: Miner's wallet address
        block_hash: Block hash from blockchain
        training_round_id: Current training round ID
        claimed_shard_id: Shard ID claimed by miner
        total_shards: Total number of shards
    
    Returns:
        True if shard assignment is correct, False otherwise
    """
    expected_shard_id = calculate_shard_id(
        miner_address,
        block_hash,
        training_round_id,
        total_shards,
    )
    
    return claimed_shard_id == expected_shard_id

