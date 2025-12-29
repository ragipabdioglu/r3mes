"""
Cache Key Constants - Centralized cache key definitions

All cache keys should be defined here to avoid key collisions and ensure consistency.
"""

# User-related cache keys
def user_info_key(wallet_address: str) -> str:
    """Cache key for user info."""
    return f"user:info:{wallet_address}"


def user_credits_key(wallet_address: str) -> str:
    """Cache key for user credits."""
    return f"user:credits:{wallet_address}"


# Network-related cache keys
def network_stats_key() -> str:
    """Cache key for network statistics."""
    return "network:stats"


def recent_blocks_key(limit: int = 10) -> str:
    """Cache key for recent blocks."""
    return f"blocks:recent:{limit}"


def block_height_key() -> str:
    """Cache key for current block height."""
    return "block:height"


# Miner-related cache keys
def miner_stats_key(wallet_address: str) -> str:
    """Cache key for miner statistics."""
    return f"miner:stats:{wallet_address}"


def miner_earnings_key(wallet_address: str, days: int = 7) -> str:
    """Cache key for miner earnings history."""
    return f"miner:earnings:{wallet_address}:{days}"


def miner_hashrate_key(wallet_address: str, days: int = 7) -> str:
    """Cache key for miner hashrate history."""
    return f"miner:hashrate:{wallet_address}:{days}"


# API key-related cache keys
def api_key_key(api_key_hash: str) -> str:
    """Cache key for API key validation."""
    return f"api:key:{api_key_hash}"


# Cache TTL constants (in seconds)
class CacheTTL:
    """Cache TTL constants."""
    
    # User info: 5 minutes
    USER_INFO = 300
    
    # Network stats: 30 seconds
    NETWORK_STATS = 30
    
    # Blocks: 10 seconds
    BLOCKS = 10
    
    # Miner stats: 1 minute
    MINER_STATS = 60
    
    # Earnings/hashrate history: 5 minutes
    MINER_HISTORY = 300
    
    # API key: 1 hour
    API_KEY = 3600

