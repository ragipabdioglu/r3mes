"""
Tests for R3MES Wallet
"""

import pytest
from r3mes.wallet import Wallet


class TestWallet:
    """Tests for Wallet class."""

    def test_generate_wallet(self):
        """Test wallet generation."""
        wallet, mnemonic = Wallet.generate()
        
        assert wallet.address.startswith("remes1")
        assert len(wallet.address) > 10
        assert mnemonic is not None
        assert len(mnemonic.split()) == 24

    def test_from_mnemonic(self):
        """Test wallet creation from mnemonic."""
        mnemonic = "abandon ability able about above absent absorb abstract absurd abuse access accident account accuse achieve acid acoustic acquire across act action actor actress actual"
        
        wallet = Wallet.from_mnemonic(mnemonic)
        
        assert wallet.address.startswith("remes1")
        assert len(wallet.address) > 10

    def test_from_mnemonic_deterministic(self):
        """Test that same mnemonic produces same address."""
        mnemonic = "test mnemonic phrase"
        
        wallet1 = Wallet.from_mnemonic(mnemonic)
        wallet2 = Wallet.from_mnemonic(mnemonic)
        
        assert wallet1.address == wallet2.address

    def test_from_private_key(self):
        """Test wallet creation from private key."""
        private_key = b"test_private_key_32_bytes_long!!"
        
        wallet = Wallet.from_private_key(private_key)
        
        assert wallet.address.startswith("remes1")

    @pytest.mark.asyncio
    async def test_sign_message(self):
        """Test message signing."""
        wallet, _ = Wallet.generate()
        message = b"test message"
        
        signature = await wallet.sign(message)
        
        assert signature is not None
        assert len(signature) == 32  # SHA256 hash length

    @pytest.mark.asyncio
    async def test_sign_without_private_key(self):
        """Test signing without private key raises error."""
        wallet = Wallet("remes1test", private_key=None)
        
        with pytest.raises(ValueError, match="Private key not available"):
            await wallet.sign(b"test")

    def test_wallet_repr(self):
        """Test wallet string representation."""
        wallet = Wallet("remes1abc123")
        
        repr_str = repr(wallet)
        
        assert "remes1abc123" in repr_str
        assert "Wallet" in repr_str

    def test_different_mnemonics_different_addresses(self):
        """Test that different mnemonics produce different addresses."""
        wallet1 = Wallet.from_mnemonic("mnemonic one")
        wallet2 = Wallet.from_mnemonic("mnemonic two")
        
        assert wallet1.address != wallet2.address
