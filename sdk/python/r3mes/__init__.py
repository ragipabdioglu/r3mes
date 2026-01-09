"""
R3MES Python SDK

Official Python SDK for interacting with the R3MES decentralized AI training network.
"""

from .client import R3MESClient
from .miner import MinerClient
from .serving import ServingClient
from .blockchain import BlockchainClient
from .governance import GovernanceClient, StakingClient
from .wallet import Wallet
from .errors import (
    R3MESError,
    ConnectionError,
    AuthenticationError,
    InsufficientCreditsError,
    NotFoundError,
    RateLimitError,
)

__version__ = "0.2.0"
__all__ = [
    "R3MESClient",
    "MinerClient",
    "ServingClient",
    "BlockchainClient",
    "GovernanceClient",
    "StakingClient",
    "Wallet",
    "R3MESError",
    "ConnectionError",
    "AuthenticationError",
    "InsufficientCreditsError",
    "NotFoundError",
    "RateLimitError",
]
