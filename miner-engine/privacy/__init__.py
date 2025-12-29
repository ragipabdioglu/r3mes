"""
R3MES Privacy Module

Provides privacy-preserving computation using TEE and encryption.
"""

from .tee_privacy import (
    PrivacyEnclave,
    SimulatedEnclave,
    SGXEnclave,
    AttestationReport,
    get_privacy_enclave,
)

__all__ = [
    "PrivacyEnclave",
    "SimulatedEnclave",
    "SGXEnclave",
    "AttestationReport",
    "get_privacy_enclave",
]
