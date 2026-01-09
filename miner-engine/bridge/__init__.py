"""
Bridge module for R3MES miner engine.

Provides blockchain communication, crypto utilities, and data transfer.
"""

from bridge.blockchain_client import BlockchainClient
from bridge.crypto import (
    sign_message,
    verify_signature,
    create_message_hash,
    derive_address_from_public_key,
    hex_to_private_key,
    generate_keypair,
    private_key_to_hex,
)
from bridge.proof_of_work import calculate_proof_of_work

__all__ = [
    'BlockchainClient',
    # Crypto utilities
    'sign_message',
    'verify_signature',
    'create_message_hash',
    'derive_address_from_public_key',
    'hex_to_private_key',
    'generate_keypair',
    'private_key_to_hex',
    # Proof of work
    'calculate_proof_of_work',
]
