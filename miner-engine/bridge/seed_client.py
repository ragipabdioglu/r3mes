"""
Global Seed Client for R3MES Miner Engine

Retrieves the deterministic global seed from the blockchain
for bit-exact reproducible training across all miners.
"""

from typing import Optional
from bridge.blockchain_client import BlockchainClient


class SeedClient:
    """
    Client for retrieving global seed from blockchain.
    
    The global seed is derived from block hash + training round ID
    to ensure all miners use the same seed for the same training round.
    """
    
    def __init__(self, blockchain_client: BlockchainClient):
        """
        Initialize seed client.
        
        Args:
            blockchain_client: Blockchain client for querying seed
        """
        self.blockchain_client = blockchain_client
    
    def get_global_seed(self, training_round_id: int) -> Optional[int]:
        """
        Get global seed for a training round from blockchain.
        
        Args:
            training_round_id: The training round ID
        
        Returns:
            Global seed (uint64) or None if query fails
        """
        # Query blockchain for global seed using blockchain client
        return self.blockchain_client.get_global_seed(training_round_id)
    
    def derive_seed_from_block_hash(self, block_hash: str, training_round_id: int) -> int:
        """
        Derive seed from block hash and training round ID (client-side).
        
        This matches the Go implementation in seed_locking.go.
        
        Args:
            block_hash: Block hash (hex string)
            training_round_id: Training round ID
        
        Returns:
            Derived seed (uint64)
        """
        import hashlib
        import struct
        
        # Convert block hash from hex to bytes
        block_hash_bytes = bytes.fromhex(block_hash)
        
        # Combine block hash with training round ID
        seed_data = block_hash_bytes + struct.pack('>Q', training_round_id)
        
        # Hash the combined data
        hash_obj = hashlib.sha256(seed_data)
        hash_bytes = hash_obj.digest()
        
        # Convert first 8 bytes to uint64 (big-endian)
        seed = struct.unpack('>Q', hash_bytes[:8])[0]
        
        return seed

