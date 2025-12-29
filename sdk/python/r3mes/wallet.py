"""
R3MES Wallet Management

Wallet creation and management utilities.
"""

import hashlib
import secrets
from typing import Tuple, Optional


class Wallet:
    """
    R3MES wallet for signing transactions.

    Example:
        # Generate new wallet
        wallet, mnemonic = Wallet.generate()
        print(f"Address: {wallet.address}")

        # Import from mnemonic
        wallet = Wallet.from_mnemonic("word1 word2 ... word24")
    """

    def __init__(self, address: str, private_key: Optional[bytes] = None):
        """
        Initialize a wallet.

        Args:
            address: Wallet address
            private_key: Optional private key bytes
        """
        self.address = address
        self._private_key = private_key

    @classmethod
    def generate(cls) -> Tuple["Wallet", str]:
        """
        Generate a new wallet with mnemonic.

        Returns:
            Tuple of (Wallet, mnemonic phrase)
        """
        # Generate random entropy
        entropy = secrets.token_bytes(32)

        # Create a simple address from entropy hash
        address_hash = hashlib.sha256(entropy).hexdigest()[:40]
        address = f"remes1{address_hash}"

        # Generate a placeholder mnemonic (in production, use BIP39)
        mnemonic_words = [
            "abandon", "ability", "able", "about", "above", "absent",
            "absorb", "abstract", "absurd", "abuse", "access", "accident",
            "account", "accuse", "achieve", "acid", "acoustic", "acquire",
            "across", "act", "action", "actor", "actress", "actual",
        ]
        mnemonic = " ".join(mnemonic_words)

        return cls(address, entropy), mnemonic

    @classmethod
    def from_mnemonic(cls, mnemonic: str) -> "Wallet":
        """
        Create a wallet from a mnemonic phrase.

        Args:
            mnemonic: BIP39 mnemonic phrase

        Returns:
            Wallet instance
        """
        # Derive key from mnemonic (simplified)
        seed = hashlib.sha256(mnemonic.encode()).digest()
        address_hash = hashlib.sha256(seed).hexdigest()[:40]
        address = f"remes1{address_hash}"

        return cls(address, seed)

    @classmethod
    def from_private_key(cls, private_key: bytes) -> "Wallet":
        """
        Create a wallet from a private key.

        Args:
            private_key: Private key bytes

        Returns:
            Wallet instance
        """
        address_hash = hashlib.sha256(private_key).hexdigest()[:40]
        address = f"remes1{address_hash}"

        return cls(address, private_key)

    async def sign(self, message: bytes) -> bytes:
        """
        Sign a message.

        Args:
            message: Message bytes to sign

        Returns:
            Signature bytes
        """
        if self._private_key is None:
            raise ValueError("Private key not available for signing")

        # Simple HMAC signature (in production, use proper ECDSA)
        import hmac
        signature = hmac.new(self._private_key, message, hashlib.sha256).digest()
        return signature

    def __repr__(self) -> str:
        return f"Wallet(address={self.address!r})"
