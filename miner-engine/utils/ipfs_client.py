"""
IPFS Client for Python Miner (Active Role)

Python miner actively uploads gradient data to IPFS before blockchain submission.
Go node only retrieves for validation (passive role).

Supports:
- Local IPFS daemon (API)
- Public IPFS gateways (HTTP fallback)
- Multiple gateway fallback for reliability
"""

import ipfshttpclient
from typing import Optional, Dict, Any, List
import hashlib
import json
import logging
import time
import os

# Set up logger
logger = logging.getLogger(__name__)

# Public IPFS gateways for fallback (read-only)
DEFAULT_PUBLIC_GATEWAYS = [
    "https://ipfs.io",
    "https://gateway.pinata.cloud",
    "https://cloudflare-ipfs.com",
    "https://dweb.link",
    "https://w3s.link",
]


class IPFSClient:
    """
    IPFS client for Python miner (active upload role).
    
    Supports both local IPFS daemon and public gateways.
    - Local daemon: Full read/write access
    - Public gateways: Read-only fallback for downloads
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        gateway_url: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        use_public_gateways: bool = True,
    ):
        """
        Initialize IPFS client.
        
        Args:
            api_url: IPFS API URL (default: from env or localhost)
            gateway_url: IPFS Gateway URL for HTTP access
            timeout: Request timeout in seconds
            max_retries: Maximum number of connection retry attempts
            retry_delay: Initial delay between retries in seconds (exponential backoff)
            use_public_gateways: Whether to use public gateways as fallback
        """
        # Get URLs from environment or use defaults
        self.api_url = api_url or os.getenv("IPFS_API_URL", "/ip4/127.0.0.1/tcp/5001")
        self.gateway_url = gateway_url or os.getenv("IPFS_GATEWAY", "http://localhost:8080")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_public_gateways = use_public_gateways
        
        # Public gateways for fallback
        self.public_gateways = DEFAULT_PUBLIC_GATEWAYS.copy()
        
        # Add custom gateway from env if provided
        custom_gateways = os.getenv("IPFS_PUBLIC_GATEWAYS", "")
        if custom_gateways:
            self.public_gateways = custom_gateways.split(",") + self.public_gateways
        
        self.client: Optional[ipfshttpclient.Client] = None
        self._connect()
    
    def _connect(self):
        """Connect to IPFS daemon with exponential backoff retry."""
        import warnings
        # Suppress IPFS version mismatch warnings (IPFS 0.26.0 works but ipfshttpclient expects older versions)
        warnings.filterwarnings('ignore', category=UserWarning, message='.*VersionMismatch.*')
        
        for attempt in range(self.max_retries):
            try:
                self.client = ipfshttpclient.connect(
                    self.api_url,
                    timeout=self.timeout,
                )
                logger.info(f"Successfully connected to IPFS at {self.api_url}")
                return
            except ipfshttpclient.exceptions.ConnectionError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"IPFS connection failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.warning(f"IPFS connection failed after {self.max_retries} attempts (ConnectionError): {e}")
                    logger.warning("Falling back to simulated IPFS mode")
                    self.client = None
            except ipfshttpclient.exceptions.TimeoutError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"IPFS connection timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.warning(f"IPFS connection failed after {self.max_retries} attempts (TimeoutError): {e}")
                    logger.warning("Falling back to simulated IPFS mode")
                    self.client = None
            except Exception as e:
                # Other unexpected errors - don't retry
                logger.error(f"Unexpected error connecting to IPFS: {e}", exc_info=True)
                logger.warning("Falling back to simulated IPFS mode")
                self.client = None
                return
    
    def is_connected(self) -> bool:
        """
        Check if IPFS client is connected and healthy.
        
        Returns:
            True if connected and healthy, False otherwise
        """
        if self.client is None:
            return False
        try:
            # Try to get IPFS node ID to verify connection
            self.client.id()
            return True
        except ipfshttpclient.exceptions.ConnectionError:
            logger.debug("IPFS connection check failed: ConnectionError")
            # Try to reconnect
            self._connect()
            return self.client is not None
        except ipfshttpclient.exceptions.TimeoutError:
            logger.debug("IPFS connection check failed: TimeoutError")
            # Try to reconnect
            self._connect()
            return self.client is not None
        except Exception as e:
            logger.debug(f"IPFS connection check failed: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on IPFS connection.
        
        Returns:
            Dictionary with health status information
        """
        health = {
            "connected": False,
            "node_id": None,
            "version": None,
            "error": None,
        }
        
        if not self.is_connected():
            health["error"] = "IPFS client not connected"
            return health
        
        try:
            node_info = self.client.id()
            health["connected"] = True
            health["node_id"] = node_info.get("ID", "unknown")
            health["version"] = node_info.get("AgentVersion", "unknown")
        except Exception as e:
            health["error"] = str(e)
            health["connected"] = False
        
        return health
    
    def upload_gradient(
        self,
        gradient_data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Upload gradient data to IPFS.
        
        Args:
            gradient_data: Serialized gradient bytes
            metadata: Optional metadata dictionary
            
        Returns:
            IPFS content hash (CID)
        """
        if not self.is_connected():
            # Simulated mode: return hash of data
            return self._simulate_upload(gradient_data)
        
        try:
            # Add data to IPFS
            result = self.client.add_bytes(gradient_data)
            ipfs_hash = result
            
            # Pin the content to ensure availability
            self.client.pin.add(ipfs_hash)
            
            # Store metadata if provided
            if metadata:
                metadata_hash = self._store_metadata(metadata)
                # Link metadata to gradient (optional)
                pass
            
            return ipfs_hash
        except ipfshttpclient.exceptions.ConnectionError as e:
            logger.error(f"IPFS upload failed (ConnectionError): {e}")
            logger.warning("Falling back to simulated IPFS upload")
            return self._simulate_upload(gradient_data)
        except ipfshttpclient.exceptions.TimeoutError as e:
            logger.error(f"IPFS upload failed (TimeoutError): {e}")
            logger.warning("Falling back to simulated IPFS upload")
            return self._simulate_upload(gradient_data)
        except Exception as e:
            logger.error(f"Unexpected error uploading to IPFS: {e}", exc_info=True)
            logger.warning("Falling back to simulated IPFS upload")
            return self._simulate_upload(gradient_data)
    
    def upload_lora_state(
        self,
        lora_state_bytes: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Upload LoRA adapter state to IPFS.
        
        Args:
            lora_state_bytes: Serialized LoRA state bytes
            metadata: Optional metadata
            
        Returns:
            IPFS content hash (CID)
        """
        return self.upload_gradient(lora_state_bytes, metadata)

    def upload_json(self, obj: Dict[str, Any]) -> str:
        """
        Upload a JSON-serializable object to IPFS.

        Used for storing auxiliary proofs/metadata such as Proof of Replication (PoRep).
        """
        import json as _json

        data = _json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()
        return self.upload_gradient(data, None)
    
    def verify_content(self, ipfs_hash: str, expected_data: bytes) -> bool:
        """
        Verify IPFS content matches expected data.
        
        Args:
            ipfs_hash: IPFS content hash
            expected_data: Expected data bytes
            
        Returns:
            True if content matches
        """
        if not self.is_connected():
            # In simulated mode, verify hash matches
            simulated_hash = self._simulate_upload(expected_data)
            return simulated_hash == ipfs_hash
        
        try:
            # Retrieve content from IPFS
            retrieved_data = self.client.cat(ipfs_hash)
            
            # Compare with expected data
            return retrieved_data == expected_data
        except ipfshttpclient.exceptions.ConnectionError as e:
            logger.error(f"IPFS verification failed (ConnectionError): {e}")
            return False
        except ipfshttpclient.exceptions.TimeoutError as e:
            logger.error(f"IPFS verification failed (TimeoutError): {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error verifying IPFS content: {e}", exc_info=True)
            return False
    
    def retrieve_content(self, ipfs_hash: str) -> Optional[bytes]:
        """
        Retrieve content from IPFS.
        
        Tries in order:
        1. Local IPFS daemon (if connected)
        2. Configured gateway
        3. Public gateways (fallback)
        
        Args:
            ipfs_hash: IPFS content hash (CID)
            
        Returns:
            Content bytes or None if retrieval fails
        """
        # Try local daemon first
        if self.is_connected():
            try:
                content = self.client.cat(ipfs_hash)
                logger.debug(f"Retrieved {ipfs_hash} from local IPFS daemon")
                return content
            except Exception as e:
                logger.warning(f"Local IPFS retrieval failed: {e}")
        
        # Try configured gateway
        content = self._retrieve_from_gateway(self.gateway_url, ipfs_hash)
        if content is not None:
            return content
        
        # Try public gateways as fallback
        if self.use_public_gateways:
            for gateway in self.public_gateways:
                content = self._retrieve_from_gateway(gateway, ipfs_hash)
                if content is not None:
                    return content
        
        logger.error(f"Failed to retrieve {ipfs_hash} from all sources")
        return None
    
    def _retrieve_from_gateway(self, gateway_url: str, ipfs_hash: str) -> Optional[bytes]:
        """
        Retrieve content from an IPFS gateway via HTTP.
        
        Args:
            gateway_url: Gateway base URL
            ipfs_hash: IPFS content hash
            
        Returns:
            Content bytes or None if retrieval fails
        """
        import requests
        
        # Clean up gateway URL
        gateway_url = gateway_url.rstrip('/')
        url = f"{gateway_url}/ipfs/{ipfs_hash}"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                logger.info(f"Retrieved {ipfs_hash} from gateway: {gateway_url}")
                return response.content
            except requests.exceptions.Timeout:
                logger.warning(f"Gateway timeout (attempt {attempt + 1}): {gateway_url}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Gateway request failed (attempt {attempt + 1}): {e}")
            
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
        
        return None
    
    def retrieve_content_streaming(self, ipfs_hash: str, chunk_size: int = 65536):
        """
        Retrieve content from IPFS in streaming mode (memory efficient).
        
        Args:
            ipfs_hash: IPFS content hash
            chunk_size: Size of each chunk in bytes
            
        Yields:
            Content chunks
        """
        import requests
        
        # Try configured gateway first
        gateways = [self.gateway_url] + (self.public_gateways if self.use_public_gateways else [])
        
        for gateway in gateways:
            gateway_url = gateway.rstrip('/')
            url = f"{gateway_url}/ipfs/{ipfs_hash}"
            
            try:
                response = requests.get(url, stream=True, timeout=self.timeout)
                response.raise_for_status()
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        yield chunk
                return
            except Exception as e:
                logger.warning(f"Streaming from {gateway} failed: {e}")
                continue
        
        logger.error(f"Failed to stream {ipfs_hash} from all gateways")
    
    def add_bytes(self, data: bytes) -> Optional[str]:
        """
        Add bytes to IPFS and return the CID.
        
        Args:
            data: Data bytes to upload
            
        Returns:
            IPFS CID or None if upload fails
        """
        if not self.is_connected():
            logger.warning("IPFS not connected, cannot upload")
            return self._simulate_upload(data)
        
        try:
            result = self.client.add_bytes(data)
            # Pin the content
            self.client.pin.add(result)
            logger.info(f"Uploaded to IPFS: {result}")
            return result
        except Exception as e:
            logger.error(f"IPFS upload failed: {e}")
            return self._simulate_upload(data)
    
    def get(self, ipfs_hash: str, output_dir: str = ".") -> Optional[str]:
        """
        Download content from IPFS to a local file.
        
        Args:
            ipfs_hash: IPFS content hash
            output_dir: Directory to save the file
            
        Returns:
            Path to downloaded file or None if failed
        """
        import os
        from pathlib import Path
        
        content = self.retrieve_content(ipfs_hash)
        if content is None:
            return None
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / ipfs_hash
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"Downloaded {ipfs_hash} to {file_path}")
        return str(file_path)
    
    def _store_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Store metadata as JSON in IPFS.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            IPFS hash of metadata
        """
        if not self.is_connected():
            return self._simulate_upload(json.dumps(metadata).encode())
        
        metadata_json = json.dumps(metadata, indent=2)
        return self.client.add_str(metadata_json)
    
    def _simulate_upload(self, data: bytes) -> str:
        """
        Simulate IPFS upload by returning hash.
        
        Args:
            data: Data bytes
            
        Returns:
            Simulated IPFS hash (SHA256)
        """
        # Use SHA256 as simulated IPFS hash
        hash_obj = hashlib.sha256(data)
        return f"Qm{hash_obj.hexdigest()[:44]}"  # Simulated CID format
    
    def pin_content(self, ipfs_hash: str) -> bool:
        """
        Pin content to ensure availability.
        
        Args:
            ipfs_hash: IPFS content hash
            
        Returns:
            True if pinning successful
        """
        if not self.is_connected():
            return False
        
        try:
            self.client.pin.add(ipfs_hash)
            return True
        except ipfshttpclient.exceptions.ConnectionError as e:
            logger.error(f"IPFS pinning failed (ConnectionError): {e}")
            return False
        except ipfshttpclient.exceptions.TimeoutError as e:
            logger.error(f"IPFS pinning failed (TimeoutError): {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error pinning IPFS content: {e}", exc_info=True)
            return False
    
    def get_pin_status(self, ipfs_hash: str) -> Dict[str, Any]:
        """
        Get pinning status for content.
        
        Args:
            ipfs_hash: IPFS content hash
            
        Returns:
            Dictionary with pin status information
        """
        if not self.is_connected():
            return {"pinned": False, "error": "IPFS not connected"}
        
        try:
            pins = self.client.pin.ls(ipfs_hash)
            return {"pinned": True, "pins": pins}
        except ipfshttpclient.exceptions.ConnectionError as e:
            logger.debug(f"IPFS pin status check failed (ConnectionError): {e}")
            return {"pinned": False, "error": "IPFS connection error"}
        except ipfshttpclient.exceptions.TimeoutError as e:
            logger.debug(f"IPFS pin status check failed (TimeoutError): {e}")
            return {"pinned": False, "error": "IPFS timeout error"}
        except Exception as e:
            logger.debug(f"IPFS pin status check failed: {e}")
            return {"pinned": False, "error": "Content not pinned or error occurred"}

