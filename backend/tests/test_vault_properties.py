"""
Property-Based Tests for Vault Secret Storage Integrity

Tests the correctness properties of HashiCorp Vault integration:
- Secret storage and retrieval integrity
- Secret non-exposure in logs and errors
- Secure failure behavior
"""

import pytest
import asyncio
import json
import logging
import os
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock
from hypothesis import given, strategies as st, settings, assume
from hypothesis.strategies import composite

# Import the modules we're testing
from app.vault_client import VaultClient, VaultConfig, VaultError, VaultSecretNotFound, VaultConnectionError
from app.auth_system import AuthenticationSystem
from app.config_manager import ConfigManager


# Test data generators
@composite
def secret_data(draw):
    """Generate valid secret data dictionaries."""
    keys = draw(st.lists(
        st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        min_size=1,
        max_size=10,
        unique=True
    ))
    
    values = draw(st.lists(
        st.text(min_size=1, max_size=200),
        min_size=len(keys),
        max_size=len(keys)
    ))
    
    return dict(zip(keys, values))


@composite
def secret_paths(draw):
    """Generate valid secret paths."""
    path_parts = draw(st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')) + '-_'),
        min_size=1,
        max_size=5
    ))
    return '/'.join(path_parts)


class TestVaultSecretStorageIntegrity:
    """
    Property 1: Vault Secret Storage Integrity
    For any valid secret configuration, storing it in Vault and then retrieving it 
    should return the exact same secret data without modification or exposure.
    
    **Validates: Requirements 1.1, 1.2**
    """
    
    @pytest.fixture
    def mock_vault_client(self):
        """Create a mock Vault client for testing."""
        with patch('hvac.Client') as mock_hvac:
            # Mock successful authentication
            mock_hvac.return_value.is_authenticated.return_value = True
            
            # Mock health check
            mock_hvac.return_value.sys.read_health_status.return_value = {
                'initialized': True,
                'sealed': False,
                'standby': False,
                'server_time_utc': '2024-01-01T00:00:00Z',
                'version': '1.15.4'
            }
            
            # Create VaultClient with test config
            config = VaultConfig(
                url="http://localhost:8200",
                token="test-token",
                mount_point="secret",
                path_prefix="test",
                cache_ttl=0  # Disable caching for tests
            )
            
            client = VaultClient(config)
            return client, mock_hvac.return_value
    
    @given(secret_data(), secret_paths())
    @settings(max_examples=100, deadline=5000)
    async def test_vault_secret_round_trip_integrity(self, secret_data_dict, path, mock_vault_client):
        """
        **Feature: critical-security-secrets, Property 1: Vault Secret Storage Integrity**
        
        For any valid secret configuration, storing it in Vault and then retrieving it 
        should return the exact same secret data without modification or exposure.
        """
        vault_client, mock_hvac = mock_vault_client
        
        # Mock the Vault responses
        mock_hvac.secrets.kv.v2.create_or_update_secret.return_value = None
        mock_hvac.secrets.kv.v2.read_secret_version.return_value = {
            'data': {
                'data': secret_data_dict
            }
        }
        
        # Initialize the client
        await vault_client.initialize()
        
        # Store the secret
        await vault_client.put_secret(path, secret_data_dict)
        
        # Retrieve the secret
        retrieved_data = await vault_client.get_secret(path)
        
        # Verify integrity: retrieved data should exactly match stored data
        assert retrieved_data == secret_data_dict
        assert isinstance(retrieved_data, dict)
        
        # Verify all keys and values are preserved
        for key, value in secret_data_dict.items():
            assert key in retrieved_data
            assert retrieved_data[key] == value
        
        # Verify no extra keys were added
        assert set(retrieved_data.keys()) == set(secret_data_dict.keys())
    
    @given(secret_data(), secret_paths(), st.text(min_size=1, max_size=50))
    @settings(max_examples=50, deadline=5000)
    async def test_vault_secret_key_specific_retrieval(self, secret_data_dict, path, key_name, mock_vault_client):
        """Test retrieving specific keys from secrets maintains integrity."""
        assume(key_name in secret_data_dict)
        
        vault_client, mock_hvac = mock_vault_client
        
        # Mock the Vault responses
        mock_hvac.secrets.kv.v2.create_or_update_secret.return_value = None
        mock_hvac.secrets.kv.v2.read_secret_version.return_value = {
            'data': {
                'data': secret_data_dict
            }
        }
        
        await vault_client.initialize()
        
        # Store the secret
        await vault_client.put_secret(path, secret_data_dict)
        
        # Retrieve specific key
        retrieved_value = await vault_client.get_secret(path, key_name)
        
        # Verify the specific value matches exactly
        assert retrieved_value == secret_data_dict[key_name]
    
    @given(secret_data(), secret_paths())
    @settings(max_examples=50, deadline=5000)
    async def test_vault_secret_storage_idempotency(self, secret_data_dict, path, mock_vault_client):
        """Test that storing the same secret multiple times maintains integrity."""
        vault_client, mock_hvac = mock_vault_client
        
        # Mock the Vault responses
        mock_hvac.secrets.kv.v2.create_or_update_secret.return_value = None
        mock_hvac.secrets.kv.v2.read_secret_version.return_value = {
            'data': {
                'data': secret_data_dict
            }
        }
        
        await vault_client.initialize()
        
        # Store the secret multiple times
        await vault_client.put_secret(path, secret_data_dict)
        await vault_client.put_secret(path, secret_data_dict)
        await vault_client.put_secret(path, secret_data_dict)
        
        # Retrieve the secret
        retrieved_data = await vault_client.get_secret(path)
        
        # Verify integrity is maintained after multiple stores
        assert retrieved_data == secret_data_dict


class TestVaultSecretNonExposure:
    """
    Property 2: Secret Non-Exposure
    For any system operation involving secrets, scanning all logs, error messages, 
    and responses should never reveal secret values in plain text.
    
    **Validates: Requirements 1.3, 7.5, 8.2**
    """
    
    @pytest.fixture
    def log_capture(self):
        """Capture log messages for analysis."""
        import logging
        from io import StringIO
        
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        
        # Add handler to all relevant loggers
        loggers = [
            logging.getLogger('app.vault_client'),
            logging.getLogger('app.auth_system'),
            logging.getLogger('app.config_manager'),
            logging.getLogger('hvac'),
        ]
        
        for logger in loggers:
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        
        yield log_stream
        
        # Cleanup
        for logger in loggers:
            logger.removeHandler(handler)
    
    @given(secret_data())
    @settings(max_examples=50, deadline=5000)
    async def test_secrets_not_exposed_in_logs(self, secret_data_dict, log_capture):
        """
        **Feature: critical-security-secrets, Property 2: Secret Non-Exposure**
        
        For any system operation involving secrets, scanning all logs should never 
        reveal secret values in plain text.
        """
        with patch('hvac.Client') as mock_hvac:
            # Mock successful operations
            mock_hvac.return_value.is_authenticated.return_value = True
            mock_hvac.return_value.sys.read_health_status.return_value = {
                'initialized': True, 'sealed': False
            }
            mock_hvac.return_value.secrets.kv.v2.create_or_update_secret.return_value = None
            mock_hvac.return_value.secrets.kv.v2.read_secret_version.return_value = {
                'data': {'data': secret_data_dict}
            }
            
            config = VaultConfig(url="http://localhost:8200", token="test-token", cache_ttl=0)
            vault_client = VaultClient(config)
            
            await vault_client.initialize()
            await vault_client.put_secret("test/path", secret_data_dict)
            await vault_client.get_secret("test/path")
        
        # Analyze all log messages
        log_contents = log_capture.getvalue()
        
        # Check that no secret values appear in logs
        for key, value in secret_data_dict.items():
            # Secret values should not appear in logs
            assert str(value) not in log_contents, f"Secret value '{value}' found in logs"
            
            # Even if the value is short, it shouldn't be logged
            if len(str(value)) > 3:  # Only check non-trivial values
                assert str(value) not in log_contents
    
    @given(secret_data())
    @settings(max_examples=30, deadline=5000)
    async def test_secrets_not_exposed_in_error_messages(self, secret_data_dict):
        """Test that error messages don't expose secret values."""
        with patch('hvac.Client') as mock_hvac:
            # Mock authentication failure
            mock_hvac.return_value.is_authenticated.return_value = False
            
            config = VaultConfig(url="http://localhost:8200", token="test-token")
            vault_client = VaultClient(config)
            
            # This should raise an error
            with pytest.raises(VaultConnectionError) as exc_info:
                await vault_client.initialize()
            
            error_message = str(exc_info.value)
            
            # Check that no secret values appear in error messages
            for key, value in secret_data_dict.items():
                assert str(value) not in error_message
    
    @given(secret_data())
    @settings(max_examples=30, deadline=5000)
    async def test_vault_config_not_exposed_in_repr(self, secret_data_dict):
        """Test that VaultConfig repr doesn't expose sensitive information."""
        config = VaultConfig(
            url="http://localhost:8200",
            token="super-secret-token",
            mount_point="secret"
        )
        
        config_repr = repr(config)
        config_str = str(config)
        
        # Token should not appear in string representations
        assert "super-secret-token" not in config_repr
        assert "super-secret-token" not in config_str


class TestVaultSecureFailureBehavior:
    """
    Property 3: Secure Failure Behavior
    For any Vault unavailability scenario, the system should fail without exposing 
    any fallback credentials or secret information.
    
    **Validates: Requirements 1.4, 7.1, 7.4**
    """
    
    @given(secret_data(), secret_paths())
    @settings(max_examples=50, deadline=5000)
    async def test_vault_unavailable_secure_failure(self, secret_data_dict, path):
        """
        **Feature: critical-security-secrets, Property 3: Secure Failure Behavior**
        
        For any Vault unavailability scenario, the system should fail without exposing 
        any fallback credentials or secret information.
        """
        with patch('hvac.Client') as mock_hvac:
            # Mock Vault connection failure
            mock_hvac.side_effect = Exception("Connection refused")
            
            config = VaultConfig(url="http://localhost:8200", token="test-token")
            
            # Client creation should fail
            with pytest.raises(VaultConnectionError):
                vault_client = VaultClient(config)
    
    @given(secret_data(), secret_paths())
    @settings(max_examples=30, deadline=5000)
    async def test_vault_authentication_failure_secure(self, secret_data_dict, path):
        """Test that authentication failures don't expose credentials."""
        with patch('hvac.Client') as mock_hvac:
            # Mock authentication failure
            mock_hvac.return_value.is_authenticated.return_value = False
            
            config = VaultConfig(url="http://localhost:8200", token="invalid-token")
            vault_client = VaultClient(config)
            
            # Should fail securely without exposing token
            with pytest.raises(VaultConnectionError) as exc_info:
                await vault_client.initialize()
            
            error_message = str(exc_info.value)
            assert "invalid-token" not in error_message
    
    @given(secret_paths())
    @settings(max_examples=30, deadline=5000)
    async def test_missing_secret_secure_failure(self, path):
        """Test that missing secrets fail securely without exposing system info."""
        with patch('hvac.Client') as mock_hvac:
            # Mock successful auth but missing secret
            mock_hvac.return_value.is_authenticated.return_value = True
            mock_hvac.return_value.sys.read_health_status.return_value = {
                'initialized': True, 'sealed': False
            }
            mock_hvac.return_value.secrets.kv.v2.read_secret_version.side_effect = Exception("not found")
            
            config = VaultConfig(url="http://localhost:8200", token="test-token", cache_ttl=0)
            vault_client = VaultClient(config)
            
            await vault_client.initialize()
            
            # Should raise VaultSecretNotFound without exposing internal details
            with pytest.raises(VaultSecretNotFound) as exc_info:
                await vault_client.get_secret(path)
            
            error_message = str(exc_info.value)
            # Should not expose internal Vault paths or configuration
            assert "test-token" not in error_message
            assert "localhost:8200" not in error_message


class TestAuthenticationSystemIntegration:
    """Test authentication system integration with Vault."""
    
    @given(st.text(min_size=3, max_size=50), st.text(min_size=8, max_size=100))
    @settings(max_examples=30, deadline=5000)
    async def test_auth_system_vault_integration_secure(self, username, password):
        """Test that authentication system integrates securely with Vault."""
        with patch('app.vault_client.get_vault_client') as mock_get_vault:
            # Mock Vault client
            mock_vault = AsyncMock()
            mock_vault.initialize.return_value = None
            mock_vault.get_secret.return_value = {"secret_key": "test-jwt-secret"}
            mock_get_vault.return_value = mock_vault
            
            # Mock environment
            with patch.dict(os.environ, {'VAULT_ADDR': 'http://localhost:8200'}):
                auth_system = AuthenticationSystem()
                await auth_system.initialize()
                
                # JWT secret should be loaded but not exposed
                assert auth_system._jwt_secret is not None
                assert auth_system._jwt_secret == "test-jwt-secret"
                
                # Health check should not expose secrets
                health = await auth_system.health_check()
                assert "test-jwt-secret" not in str(health)


# Integration test for the complete flow
class TestVaultIntegrationFlow:
    """Test the complete Vault integration flow."""
    
    @pytest.mark.asyncio
    async def test_complete_vault_flow_mock(self):
        """Test the complete flow with mocked Vault."""
        test_secrets = {
            "database": {
                "user": "r3mes",
                "password": "secure-db-password",
                "host": "localhost",
                "port": "5432",
                "database": "r3mes"
            },
            "jwt": {
                "secret_key": "secure-jwt-secret",
                "algorithm": "HS256"
            }
        }
        
        with patch('hvac.Client') as mock_hvac:
            # Mock successful Vault operations
            mock_hvac.return_value.is_authenticated.return_value = True
            mock_hvac.return_value.sys.read_health_status.return_value = {
                'initialized': True, 'sealed': False
            }
            
            # Mock secret retrieval
            def mock_read_secret(path, mount_point):
                secret_name = path.split('/')[-1]
                if secret_name in test_secrets:
                    return {'data': {'data': test_secrets[secret_name]}}
                raise Exception("not found")
            
            mock_hvac.return_value.secrets.kv.v2.read_secret_version.side_effect = mock_read_secret
            mock_hvac.return_value.secrets.kv.v2.create_or_update_secret.return_value = None
            
            # Test Vault client
            config = VaultConfig(url="http://localhost:8200", token="test-token", cache_ttl=0)
            vault_client = VaultClient(config)
            
            await vault_client.initialize()
            
            # Test storing and retrieving secrets
            for secret_name, secret_data in test_secrets.items():
                await vault_client.put_secret(secret_name, secret_data)
                retrieved = await vault_client.get_secret(secret_name)
                assert retrieved == secret_data
            
            # Test health check
            health = await vault_client.health_check()
            assert health["healthy"] is True
            assert health["status"] == "healthy"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])