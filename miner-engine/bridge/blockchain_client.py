"""
Blockchain Client for Python Miner

Connects Python miner to Go blockchain node via gRPC.
Implements message signing, authentication, and transaction submission.
"""

import grpc
from typing import Optional, Dict, Any, List
import hashlib
import time
import struct
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Import crypto utilities
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

# Import generated proto files
import sys
import os

# Add proto directory to path
proto_path = os.path.join(os.path.dirname(__file__), 'proto')
if proto_path not in sys.path:
    sys.path.insert(0, proto_path)

# Pre-register amino and gogoproto descriptors in the protobuf pool
# This is required because remes proto files depend on these, but the Python
# stubs don't create proper descriptors. We create minimal descriptors here.
try:
    from google.protobuf import descriptor_pool
    pool = descriptor_pool.Default()
    
    # Try to add minimal amino descriptor if not already present
    try:
        # Check if already registered by trying to find it
        pool.FindFileByName('amino/amino.proto')
    except KeyError:
        # Not registered, create a minimal one
        from google.protobuf import descriptor_pb2
        amino_file = descriptor_pb2.FileDescriptorProto()
        amino_file.name = 'amino/amino.proto'
        amino_file.package = 'amino'
        amino_file.syntax = 'proto3'
        pool.Add(amino_file)
    
    # Try to add minimal gogoproto descriptor if not already present
    try:
        pool.FindFileByName('gogoproto/gogo.proto')
    except KeyError:
        from google.protobuf import descriptor_pb2
        gogo_file = descriptor_pb2.FileDescriptorProto()
        gogo_file.name = 'gogoproto/gogo.proto'
        gogo_file.package = 'gogoproto'
        gogo_file.syntax = 'proto3'
        pool.Add(gogo_file)
except Exception:
    # If descriptor manipulation fails, try importing stubs
    try:
        import amino.amino_pb2  # noqa: F401
        import gogoproto.gogo_pb2  # noqa: F401
    except ImportError:
        try:
            from amino import amino_pb2  # noqa: F401
            from gogoproto import gogo_pb2  # noqa: F401
        except ImportError:
            pass  # Will fail later if actually needed

# Import proto modules
try:
    from remes.remes.v1 import tx_pb2, tx_pb2_grpc
    from remes.remes.v1 import query_pb2, query_pb2_grpc
except ImportError as e:
    # Fallback for development/testing
    print(f"Warning: Proto imports failed: {e}")
    tx_pb2 = None
    tx_pb2_grpc = None
    query_pb2 = None
    query_pb2_grpc = None


class BlockchainClient:
    """
    Blockchain client for Python miner to communicate with Go node.
    
    Architecture:
    - Python miner uploads gradient to IPFS (active role)
    - Python miner sends only IPFS hash + metadata to Go node via gRPC
    - Go node stores hash on-chain (passive IPFS role)
    """
    
    def __init__(
        self,
        node_url: Optional[str] = None,
        private_key: Optional[str] = None,
        chain_id: str = "remes-test",
        use_tls: bool = False,
        tls_cert_file: Optional[str] = None,
        tls_key_file: Optional[str] = None,
        tls_ca_file: Optional[str] = None,
        tls_server_name: Optional[str] = None,
    ):
        """
        Initialize blockchain client.
        
        Args:
            node_url: gRPC endpoint URL (default: from R3MES_NODE_GRPC_URL env var, or localhost:9090 in dev)
            private_key: Private key for message signing (hex string, Secp256k1)
            chain_id: Chain ID for transactions
            use_tls: Enable TLS mutual authentication (default: False)
            tls_cert_file: Path to client certificate file (required if use_tls=True)
            tls_key_file: Path to client private key file (required if use_tls=True)
            tls_ca_file: Path to CA certificate file (required if use_tls=True)
            tls_server_name: Server name for TLS verification (default: extracted from node_url)
        """
        # Get node URL from parameter, environment variable, or default
        # In production, R3MES_NODE_GRPC_URL must be set (no localhost fallback)
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        
        if node_url:
            self.node_url = node_url
        else:
            node_url_env = os.getenv("R3MES_NODE_GRPC_URL")
            if not node_url_env:
                if is_production:
                    raise ValueError(
                        "R3MES_NODE_GRPC_URL environment variable must be set in production. "
                        "Do not use localhost in production."
                    )
                # Development fallback
                self.node_url = "localhost:9090"
                logger.warning("R3MES_NODE_GRPC_URL not set, using localhost fallback (development only)")
            else:
                self.node_url = node_url_env
                # Validate that production doesn't use localhost
                if is_production and ("localhost" in self.node_url or "127.0.0.1" in self.node_url):
                    raise ValueError(
                        f"R3MES_NODE_GRPC_URL cannot use localhost in production: {self.node_url}"
                    )
        self.chain_id = chain_id
        self.use_tls = use_tls
        
        # Handle private key
        if private_key:
            self.private_key_bytes = hex_to_private_key(private_key)
        else:
            # Generate new keypair for testing
            self.private_key_bytes, self.public_key_bytes = generate_keypair()
            self.private_key = private_key_to_hex(self.private_key_bytes)
            print(f"Generated new keypair. Private key: {self.private_key}")
        
        # Derive miner address from public key
        if hasattr(self, 'public_key_bytes'):
            self.miner_address = derive_address_from_public_key(self.public_key_bytes)
        else:
            # If only private key provided, derive public key
            from ecdsa import SigningKey, SECP256k1
            sk = SigningKey.from_string(self.private_key_bytes, curve=SECP256k1)
            vk = sk.get_verifying_key()
            self.public_key_bytes = vk.to_string()
            self.miner_address = derive_address_from_public_key(self.public_key_bytes)
        
        self.nonce = 0
        
        # Create gRPC channel with optional TLS/mTLS
        if use_tls:
            if not tls_cert_file or not tls_key_file or not tls_ca_file:
                raise ValueError(
                    "TLS certificate files are required when use_tls=True. "
                    "Either provide certificate files or disable TLS with use_tls=False. "
                    "Run 'r3mes-miner setup' to reconfigure."
                )
            
            # Check if certificate files exist
            missing_files = []
            if not os.path.exists(tls_cert_file):
                missing_files.append(tls_cert_file)
            if not os.path.exists(tls_key_file):
                missing_files.append(tls_key_file)
            if not os.path.exists(tls_ca_file):
                missing_files.append(tls_ca_file)
            
            if missing_files:
                raise FileNotFoundError(
                    f"TLS certificate files not found:\n" +
                    "\n".join(f"  - {f}" for f in missing_files) +
                    "\n\nEither create these files or disable TLS. "
                    "Run 'r3mes-miner setup' to reconfigure without TLS."
                )
            
            try:
                # Load TLS credentials
                from grpc import ssl_channel_credentials
                
                # Read certificate files
                with open(tls_cert_file, 'rb') as f:
                    client_cert = f.read()
                with open(tls_key_file, 'rb') as f:
                    client_key = f.read()
                with open(tls_ca_file, 'rb') as f:
                    ca_cert = f.read()
                
                # Create TLS credentials with mutual authentication
                credentials = ssl_channel_credentials(
                    root_certificates=ca_cert,
                    private_key=client_key,
                    certificate_chain=client_cert,
                )
                
                # Extract server name from node_url if not provided
                if not tls_server_name:
                    # Extract hostname from node_url (e.g., "localhost:9090" -> "localhost")
                    from urllib.parse import urlparse
                    parsed = urlparse(f"//{node_url}")
                    tls_server_name = parsed.hostname or "localhost"
                
                # Create secure channel
                self.channel = grpc.secure_channel(node_url, credentials, options=[
                    ('grpc.ssl_target_name_override', tls_server_name),
                ])
            except ImportError:
                raise ImportError("TLS support requires grpcio with SSL support. Install with: pip install grpcio[secure]")
        else:
            # Create insecure channel (for development/testing)
            self.channel = grpc.insecure_channel(node_url)
        
        # Create query stub
        if query_pb2_grpc:
            self.query_stub = query_pb2_grpc.QueryStub(self.channel)
        else:
            self.query_stub = None
        
        # Create gRPC stubs (if proto files available)
        if tx_pb2_grpc is not None and query_pb2_grpc is not None:
            self.stub = tx_pb2_grpc.MsgStub(self.channel)
            self.query_stub = query_pb2_grpc.QueryStub(self.channel)
        else:
            self.stub = None
            self.query_stub = None
    
    def get_global_seed(self, training_round_id: int) -> Optional[int]:
        """
        Get global seed from blockchain for deterministic training.
        
        This retrieves the global seed derived from block hash and training round ID.
        The seed is locked to ensure all miners use the same seed for the same round.
        
        Args:
            training_round_id: Training round ID
            
        Returns:
            Global seed as integer, or None if query fails
        """
        if not self.query_stub or not query_pb2:
            print("Warning: Query stub not available, using fixed seed")
            return None
        
        try:
            # Create query request
            request = query_pb2.QueryGetGlobalSeedRequest(
                training_round_id=training_round_id
            )
            
            # Query blockchain
            response = self.query_stub.GetGlobalSeed(request)
            
            # Convert global seed from uint64 to int
            # Global seed is stored as uint64 in blockchain
            global_seed = int(response.global_seed)
            
            # Also get block hash for verification
            block_hash_hex = response.block_hash if hasattr(response, 'block_hash') else None
            
            if block_hash_hex:
                print(f"✅ Retrieved global seed from blockchain: {global_seed} (training round {training_round_id}, block hash: {block_hash_hex[:16]}...)")
            else:
                print(f"✅ Retrieved global seed from blockchain: {global_seed} (training round {training_round_id})")
            
            return global_seed
            
        except grpc.RpcError as e:
            print(f"⚠️  Failed to get global seed from blockchain: {e.code()}: {e.details()}")
            return None
        except Exception as e:
            print(f"⚠️  Error getting global seed: {e}")
            return None
    
    def get_nonce(self) -> int:
        """
        Get next nonce for transaction.
        
        Returns:
            Next nonce value
        """
        self.nonce += 1
        return self.nonce
    
    def sign_message(self, message_bytes: bytes) -> bytes:
        """
        Sign message with miner's Secp256k1 private key.
        
        Args:
            message_bytes: Message to sign
            
        Returns:
            Signature bytes (DER format)
        """
        return sign_message(message_bytes, self.private_key_bytes)
    
    def submit_gradient(
        self,
        miner_address: str,
        ipfs_hash: str,
        model_version: str,
        training_round_id: int,
        shard_id: int,
        gradient_hash: str,
        gpu_architecture: str,
        claimed_loss: Optional[str] = None,
        porep_proof_ipfs_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit gradient to blockchain.
        
        IMPORTANT: Only IPFS hash + metadata is sent, NOT gradient data.
        Python miner uploads gradient data directly to IPFS before this call.
        
        Args:
            miner_address: Miner's blockchain address
            ipfs_hash: IPFS hash of gradient data
            model_version: Model version used
            training_round_id: Training round identifier
            shard_id: Shard assignment
            gradient_hash: Deterministic gradient hash
            gpu_architecture: GPU architecture (Ada, Ampere, etc.)
            claimed_loss: Loss value claimed by miner (BitNet integer format)
                        Used for Loss-Based Spot Checking in Layer 2 verification
            
        Returns:
            Transaction response dictionary
        """
        # Get nonce
        nonce = self.get_nonce()
        
        # Create deterministic message hash for signing
        message_hash = create_message_hash(
            miner=miner_address,
            ipfs_hash=ipfs_hash,
            model_version=model_version,
            training_round_id=training_round_id,
            shard_id=shard_id,
            gradient_hash=gradient_hash,
            gpu_architecture=gpu_architecture,
            nonce=nonce,
            chain_id=self.chain_id,
        )
        
        # Sign message hash
        signature = self.sign_message(message_hash)
        
        # Calculate proof-of-work (anti-spam)
        pow_nonce = calculate_proof_of_work(message_hash, difficulty=4)
        if pow_nonce is None:
            return {
                "success": False,
                "error": "Failed to calculate proof-of-work",
            }
        
        # Create and send gRPC message
        if self.stub is None or tx_pb2 is None:
            # Fallback to simulated mode
            return {
                "success": True,
                "stored_gradient_id": 1,  # Simulated
                "tx_hash": "simulated_tx_hash",
                "message": "Gradient submitted (simulated - proto files not available)",
            }
        
        msg = tx_pb2.MsgSubmitGradient(
            miner=miner_address,
            ipfs_hash=ipfs_hash,
            model_version=model_version,
            training_round_id=training_round_id,
            shard_id=shard_id,
            gradient_hash=gradient_hash,
            gpu_architecture=gpu_architecture,
            nonce=nonce,
            signature=signature,
            proof_of_work_nonce=pow_nonce,
            claimed_loss=claimed_loss or "",  # Loss-Based Spot Checking: miner's claimed loss
            porep_proof_ipfs_hash=porep_proof_ipfs_hash or "",
        )
        
        try:
            response = self.stub.SubmitGradient(msg)
            # Transaction hash is now returned in the response (added in Go handler)
            tx_hash = getattr(response, 'tx_hash', '')
            if not tx_hash:
                # Fallback: if tx_hash is empty, use "pending" for backward compatibility
                tx_hash = "pending"
            
            return {
                "success": True,
                "stored_gradient_id": response.stored_gradient_id,
                "tx_hash": tx_hash,  # Transaction hash from Go handler
                "message": "Gradient submitted successfully",
            }
        except grpc.RpcError as e:
            return {
                "success": False,
                "error": str(e),
                "code": e.code(),
            }
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get current global model parameters from blockchain.
        
        Returns:
            Dictionary with model IPFS hash and version
        """
        # Query model parameters
        if self.query_stub is None or query_pb2 is None:
            # Fallback to simulated mode
            return {
                "model_ipfs_hash": "",
                "model_version": "v1.0.0",
                "last_updated_height": 0,
            }
        
        request = query_pb2.QueryGetModelParamsRequest()
        try:
            response = self.query_stub.GetModelParams(request)
            return {
                "model_ipfs_hash": response.model_ipfs_hash,
                "model_version": response.model_version,
                "last_updated_height": response.last_updated_height,
            }
        except grpc.RpcError as e:
            # Return default if query fails
            return {
                "model_ipfs_hash": "",
                "model_version": "v1.0.0",
                "last_updated_height": 0,
                "error": str(e),
            }
    
    def get_block_hash(self) -> str:
        """
        Get current block hash from blockchain via gRPC query.
        
        Returns:
            Block hash as hex string (or simulated if query fails)
        """
        # Try to get block hash from GetGlobalSeed query (training_round_id=0 for current)
        if self.query_stub is not None and query_pb2 is not None:
            try:
                # Use training_round_id=0 to get current block hash
                request = query_pb2.QueryGetGlobalSeedRequest(training_round_id=0)
                response = self.query_stub.GetGlobalSeed(request)
                if response and response.block_hash:
                    return response.block_hash
            except grpc.RpcError as e:
                error_code = e.code()
                error_details = e.details() if hasattr(e, 'details') else str(e)
                logger.warning(f"Failed to get block hash via gRPC (RpcError): {error_code} - {error_details}")
            except Exception as e:
                logger.warning(f"Failed to get block hash via gRPC: {e}", exc_info=True)
        
        # Retry mechanism with exponential backoff (async sleep, non-blocking)
        # Use asyncio.run to execute async retry in sync context
        import asyncio
        import concurrent.futures
        
        max_retries = 3
        retry_delay = 1.0
        last_error = None
        
        # Helper function to make request in thread pool (non-blocking)
        def _make_request(rpc_url: str) -> Dict[str, Any]:
            """Make blocking HTTP request in thread pool."""
            import requests
            response = requests.post(
                f"{rpc_url}/block",
                json={"jsonrpc": "2.0", "id": 1, "method": "block", "params": {}},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        
        # Try async retry with non-blocking sleep
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, use sync fallback
                return self._get_block_hash_sync_fallback()
            
            # Run async retry with thread pool for blocking requests
            async def _async_retry():
                nonlocal last_error
                rpc_url = self.node_url.replace(":9090", ":26657")
                if not rpc_url.startswith("http"):
                    rpc_url = f"http://{rpc_url}"
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for attempt in range(max_retries):
                        try:
                            # Run blocking request in thread pool
                            data = await loop.run_in_executor(executor, _make_request, rpc_url)
                            
                            if "result" in data and "block" in data["result"]:
                                block_hash = data["result"]["block"]["header"]["hash"]
                                logger.info(f"Retrieved block hash from Tendermint RPC: {block_hash[:16]}...")
                                return block_hash
                            else:
                                raise ValueError("Invalid response format from Tendermint RPC")
                                
                        except Exception as e:
                            last_error = e
                            if attempt < max_retries - 1:
                                delay = retry_delay * (2 ** attempt)
                                logger.warning(f"Failed to get block hash (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                                await asyncio.sleep(delay)  # Non-blocking sleep
                            else:
                                logger.error(f"Failed to get block hash after {max_retries} attempts: {e}")
                
                raise RuntimeError(
                    f"Failed to get block hash after {max_retries} retries. Last error: {last_error}. "
                    "Ensure blockchain node is running and accessible."
                )
            
            return asyncio.run(_async_retry())
        except RuntimeError:
            # No event loop, use sync fallback
            return self._get_block_hash_sync_fallback()
    
    def _get_block_hash_sync_fallback(self) -> str:
        """
        Sync fallback for getting block hash (used when async is not available).
        
        Returns:
            Block hash as hex string
            
        Raises:
            RuntimeError: If all retries fail
        """
        max_retries = 3
        retry_delay = 1.0
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Try Tendermint RPC as fallback
                import requests
                rpc_url = self.node_url.replace(":9090", ":26657")  # Convert gRPC to RPC
                if not rpc_url.startswith("http"):
                    rpc_url = f"http://{rpc_url}"
                
                response = requests.post(
                    f"{rpc_url}/block",
                    json={"jsonrpc": "2.0", "id": 1, "method": "block", "params": {}},
                    timeout=5
                )
                response.raise_for_status()
                data = response.json()
                
                if "result" in data and "block" in data["result"]:
                    block_hash = data["result"]["block"]["header"]["hash"]
                    logger.info(f"Retrieved block hash from Tendermint RPC: {block_hash[:16]}...")
                    return block_hash
                else:
                    raise ValueError("Invalid response format from Tendermint RPC")
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Failed to get block hash (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                    import time
                    time.sleep(delay)  # Sync fallback - blocking but necessary if async not available
                else:
                    logger.error(f"Failed to get block hash after {max_retries} attempts: {e}")
        
        # No fallback - raise error in production
        raise RuntimeError(
            f"Failed to get block hash after {max_retries} retries. Last error: {last_error}. "
            "Ensure blockchain node is running and accessible."
        )
    
    def get_miner_address(self) -> str:
        """
        Get miner's blockchain address.
        
        Returns:
            Miner address (hex string)
        """
        return self.miner_address
    
    def get_available_chunks(self, pool_id: int, limit: int = 100) -> List[Any]:
        """
        Get available chunks from a task pool.
        
        Args:
            pool_id: Task pool ID
            limit: Maximum number of chunks to return
            
        Returns:
            List of available task chunks (dicts with chunk_id, data_hash, shard_id)
        """
        if self.query_stub is None or query_pb2 is None:
            logger.warning("Query stub not available, returning empty list")
            return []
        
        try:
            # Query available chunks using QueryAvailableChunks endpoint
            request = query_pb2.QueryAvailableChunksRequest(
                pool_id=pool_id,
                limit=limit
            )
            response = self.query_stub.QueryAvailableChunks(request)
            
            # Convert response chunks to list of dicts
            chunks = []
            for chunk in response.chunks:
                chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "data_hash": chunk.data_hash,
                    "shard_id": chunk.shard_id,
                })
            
            logger.info(f"Retrieved {len(chunks)} available chunks from pool {pool_id} (total available: {response.total_available})")
            return chunks
            
        except grpc.RpcError as e:
            logger.error(f"Failed to get available chunks: {e.code()}: {e.details()}")
            return []
        except AttributeError as e:
            # QueryAvailableChunks may not be available in proto yet (needs regeneration)
            logger.warning(f"QueryAvailableChunks not available in proto (may need regeneration): {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting available chunks: {e}", exc_info=True)
            return []
    
    def send_claim_task(self, miner: str, pool_id: int, chunk_id: int) -> Dict[str, Any]:
        """
        Send ClaimTask transaction to blockchain.
        
        Args:
            miner: Miner address
            pool_id: Task pool ID
            chunk_id: Chunk ID to claim
            
        Returns:
            Dictionary with success status and transaction hash (or error message)
        """
        if self.stub is None or tx_pb2 is None:
            logger.warning("Transaction stub not available")
            return {
                "success": False,
                "error": "Transaction stub not available (proto files not loaded)",
            }
        
        try:
            # Create and send ClaimTask message
            msg = tx_pb2.MsgClaimTask(
                miner=miner,
                pool_id=pool_id,
                chunk_id=chunk_id,
            )
            
            response = self.stub.ClaimTask(msg)
            tx_hash = getattr(response, 'tx_hash', '')
            success = getattr(response, 'success', False)
            
            if not tx_hash:
                # Fallback: if tx_hash is empty, use "pending" for backward compatibility
                tx_hash = "pending"
            
            if not success:
                return {
                    "success": False,
                    "error": "ClaimTask transaction failed",
                    "tx_hash": tx_hash,
                }
            
            logger.info(f"Task claimed successfully: Pool {pool_id}, Chunk {chunk_id}, TX: {tx_hash}")
            return {
                "success": True,
                "tx_hash": tx_hash,
                "message": "Task claimed successfully",
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to claim task: {e.code()}: {e.details()}")
            return {
                "success": False,
                "error": str(e.details()) if hasattr(e, 'details') else str(e),
                "code": str(e.code()),
            }
        except Exception as e:
            logger.error(f"Error claiming task: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
    
    def send_complete_task(
        self, 
        miner: str, 
        pool_id: int, 
        chunk_id: int, 
        gradient_hash: str,
        gradient_ipfs_hash: str = "",
        miner_gpu: str = ""
    ) -> Dict[str, Any]:
        """
        Send CompleteTask transaction to blockchain.
        
        Args:
            miner: Miner address
            pool_id: Task pool ID
            chunk_id: Chunk ID to complete
            gradient_hash: Deterministic hash of gradient result
            gradient_ipfs_hash: IPFS hash of gradient data (optional)
            miner_gpu: GPU architecture used (optional, e.g., "Ada", "Ampere", "Blackwell")
            
        Returns:
            Dictionary with success status and transaction hash (or error message)
        """
        if self.stub is None or tx_pb2 is None:
            logger.warning("Transaction stub not available")
            return {
                "success": False,
                "error": "Transaction stub not available (proto files not loaded)",
            }
        
        try:
            # Create and send CompleteTask message
            msg = tx_pb2.MsgCompleteTask(
                miner=miner,
                pool_id=pool_id,
                chunk_id=chunk_id,
                gradient_hash=gradient_hash,
                gradient_ipfs_hash=gradient_ipfs_hash,
                miner_gpu=miner_gpu,
            )
            
            response = self.stub.CompleteTask(msg)
            tx_hash = getattr(response, 'tx_hash', '')
            success = getattr(response, 'success', False)
            
            if not tx_hash:
                # Fallback: if tx_hash is empty, use "pending" for backward compatibility
                tx_hash = "pending"
            
            if not success:
                return {
                    "success": False,
                    "error": "CompleteTask transaction failed",
                    "tx_hash": tx_hash,
                }
            
            logger.info(f"Task completed successfully: Pool {pool_id}, Chunk {chunk_id}, TX: {tx_hash}")
            return {
                "success": True,
                "tx_hash": tx_hash,
                "message": "Task completed successfully",
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to complete task: {e.code()}: {e.details()}")
            return {
                "success": False,
                "error": str(e.details()) if hasattr(e, 'details') else str(e),
                "code": str(e.code()),
            }
        except Exception as e:
            logger.error(f"Error completing task: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
    
    def get_miner_score(self, miner_address: str) -> Dict[str, Any]:
        """
        Get miner reputation score and statistics.
        
        Args:
            miner_address: Miner's blockchain address
            
        Returns:
            Dictionary with miner score information
        """
        # Query miner score
        if self.query_stub is None or query_pb2 is None:
            # Fallback to simulated mode
            return {
                "miner": miner_address,
                "trust_score": "0.5",
                "reputation_tier": "new",
                "total_submissions": 0,
                "successful_submissions": 0,
                "slashing_events": 0,
            }
        
        request = query_pb2.QueryGetMinerScoreRequest(miner=miner_address)
        try:
            response = self.query_stub.GetMinerScore(request)
            return {
                "miner": response.miner,
                "trust_score": response.trust_score,
                "reputation_tier": response.reputation_tier,
                "total_submissions": response.total_submissions,
                "successful_submissions": response.successful_submissions,
                "slashing_events": response.slashing_events,
            }
        except grpc.RpcError as e:
            # Return default if query fails
            return {
                "miner": miner_address,
                "trust_score": "0.5",
                "reputation_tier": "new",
                "total_submissions": 0,
                "successful_submissions": 0,
                "slashing_events": 0,
                "error": str(e),
            }
    
    def get_active_pool(self) -> Optional[int]:
        """
        Get the currently active task pool ID from blockchain.
        
        Returns:
            Active pool ID if available, None otherwise
        """
        if self.query_stub is None or query_pb2 is None:
            # Fallback: return None if query stub not available
            logger.warning("Query stub not available, cannot get active pool")
            return None
        
        try:
            request = query_pb2.QueryActivePoolRequest()
            response = self.query_stub.QueryActivePool(request)
            if response.has_active_pool:
                return response.pool_id
            return None
        except grpc.RpcError as e:
            logger.error(f"Failed to query active pool: {e.code()}: {e.details()}")
            return None
        except AttributeError:
            # QueryActivePool may not be available in proto yet (needs regeneration)
            logger.warning("QueryActivePool not available in proto (may need regeneration)")
            return None
        except Exception as e:
            logger.error(f"Error querying active pool: {e}", exc_info=True)
            return None
    
    def register_node(
        self,
        node_address: str,
        node_type: int = 1,  # Default: MINER (1)
        stake: str = "0",
        roles: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Register a node (miner/validator) with the blockchain.
        
        Args:
            node_address: Node's blockchain address
            node_type: Primary node type (1=MINER, 2=VALIDATOR, etc.)
            stake: Staked amount (as string, e.g., "1000000")
            roles: List of role IDs this node can perform (optional)
            
        Returns:
            Dictionary with success status and registration_id (or error message)
        """
        if self.stub is None or tx_pb2 is None:
            logger.warning("Transaction stub not available")
            return {
                "success": False,
                "error": "Transaction stub not available (proto files not loaded)",
            }
        
        try:
            # Create RegisterNode message
            # Note: MsgRegisterNode requires ResourceSpec and other fields
            # For now, we'll create a minimal message with default/empty values
            # In production, ResourceSpec should be properly populated
            
            # Check if ResourceSpec is available in proto
            if hasattr(tx_pb2, 'ResourceSpec'):
                # Create empty ResourceSpec (fields will be set by blockchain defaults)
                resources = tx_pb2.ResourceSpec()
            else:
                resources = None
            
            # Prepare roles list (default to [node_type] if not provided)
            if roles is None:
                roles = [node_type]
            
            # Create message
            # Note: Some fields may need to be set based on actual proto structure
            msg_kwargs = {
                "node_address": node_address,
                "node_type": node_type,
                "stake": stake,
            }
            
            # Add resources if available
            if resources is not None:
                msg_kwargs["resources"] = resources
            
            # Add roles if available in proto
            if hasattr(tx_pb2.MsgRegisterNode, 'roles'):
                msg_kwargs["roles"] = roles
            
            msg = tx_pb2.MsgRegisterNode(**msg_kwargs)
            
            response = self.stub.RegisterNode(msg)
            registration_id = getattr(response, 'registration_id', 0)
            
            logger.info(f"Node registered successfully: {node_address}, Registration ID: {registration_id}")
            return {
                "success": True,
                "registration_id": registration_id,
                "message": "Node registered successfully",
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to register node: {e.code()}: {e.details()}")
            return {
                "success": False,
                "error": str(e.details()) if hasattr(e, 'details') else str(e),
                "code": str(e.code()),
            }
        except Exception as e:
            logger.error(f"Error registering node: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
    
    def update_serving_node_status(
        self,
        serving_node: str,
        is_available: bool,
        model_version: Optional[str] = None,
        model_ipfs_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update serving node status on blockchain.
        
        Args:
            serving_node: Serving node address
            is_available: Whether node is available for requests
            model_version: Model version (optional)
            model_ipfs_hash: IPFS hash of model (optional)
            
        Returns:
            Dictionary with success status and transaction hash
        """
        if self.stub is None or tx_pb2 is None:
            logger.warning("Transaction stub not available")
            return {
                "success": False,
                "error": "Transaction stub not available (proto files not loaded)",
            }
        
        try:
            # Create MsgUpdateServingNodeStatus message
            msg_kwargs = {
                "serving_node": serving_node,
                "is_available": is_available,
            }
            
            if model_version is not None:
                msg_kwargs["model_version"] = model_version
            if model_ipfs_hash is not None:
                msg_kwargs["model_ipfs_hash"] = model_ipfs_hash
            
            msg = tx_pb2.MsgUpdateServingNodeStatus(**msg_kwargs)
            response = self.stub.UpdateServingNodeStatus(msg)
            
            tx_hash = getattr(response, 'tx_hash', '')
            success = getattr(response, 'success', False)
            
            if not tx_hash:
                tx_hash = "pending"
            
            logger.info(f"Serving node status updated: {serving_node}, available={is_available}, TX: {tx_hash}")
            return {
                "success": success,
                "tx_hash": tx_hash,
                "message": "Serving node status updated successfully",
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to update serving node status: {e.code()}: {e.details()}")
            return {
                "success": False,
                "error": str(e.details()) if hasattr(e, 'details') else str(e),
                "code": str(e.code()),
            }
        except AttributeError as e:
            # MsgUpdateServingNodeStatus may not be available in proto yet
            logger.warning(f"UpdateServingNodeStatus not available in proto (may need regeneration): {e}")
            return {
                "success": False,
                "error": "UpdateServingNodeStatus not implemented in proto",
            }
        except Exception as e:
            logger.error(f"Error updating serving node status: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
    
    def submit_inference_result(
        self,
        serving_node: str,
        request_id: str,
        result_ipfs_hash: str,
        latency_ms: int,
    ) -> Dict[str, Any]:
        """
        Submit inference result to blockchain.
        
        Args:
            serving_node: Serving node address
            request_id: Inference request ID
            result_ipfs_hash: IPFS hash of inference result
            latency_ms: Inference latency in milliseconds
            
        Returns:
            Dictionary with success status and transaction hash
        """
        if self.stub is None or tx_pb2 is None:
            logger.warning("Transaction stub not available")
            return {
                "success": False,
                "error": "Transaction stub not available (proto files not loaded)",
            }
        
        try:
            # Create MsgSubmitInferenceResult message
            msg = tx_pb2.MsgSubmitInferenceResult(
                serving_node=serving_node,
                request_id=request_id,
                result_ipfs_hash=result_ipfs_hash,
                latency_ms=latency_ms,
            )
            
            response = self.stub.SubmitInferenceResult(msg)
            tx_hash = getattr(response, 'tx_hash', '')
            success = getattr(response, 'success', False)
            
            if not tx_hash:
                tx_hash = "pending"
            
            logger.info(f"Inference result submitted: request_id={request_id}, TX: {tx_hash}")
            return {
                "success": success,
                "tx_hash": tx_hash,
                "message": "Inference result submitted successfully",
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to submit inference result: {e.code()}: {e.details()}")
            return {
                "success": False,
                "error": str(e.details()) if hasattr(e, 'details') else str(e),
                "code": str(e.code()),
            }
        except AttributeError as e:
            logger.warning(f"SubmitInferenceResult not available in proto (may need regeneration): {e}")
            return {
                "success": False,
                "error": "SubmitInferenceResult not implemented in proto",
            }
        except Exception as e:
            logger.error(f"Error submitting inference result: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
    
    def commit_aggregation(
        self,
        proposer: str,
        gradient_ids: List[int],
        training_round_id: int,
        commitment_hash: str,
    ) -> Dict[str, Any]:
        """
        Commit aggregation to blockchain.
        
        Args:
            proposer: Proposer address
            gradient_ids: List of gradient IDs included in aggregation
            training_round_id: Training round ID
            commitment_hash: Commitment hash (hash of aggregation result)
            
        Returns:
            Dictionary with success status and commitment_id
        """
        if self.stub is None or tx_pb2 is None:
            logger.warning("Transaction stub not available")
            return {
                "success": False,
                "error": "Transaction stub not available (proto files not loaded)",
            }
        
        try:
            # Create MsgCommitAggregation message
            msg = tx_pb2.MsgCommitAggregation(
                proposer=proposer,
                gradient_ids=gradient_ids,
                training_round_id=training_round_id,
                commitment_hash=commitment_hash,
            )
            
            response = self.stub.CommitAggregation(msg)
            commitment_id = getattr(response, 'commitment_id', 0)
            success = getattr(response, 'success', False)
            
            logger.info(f"Aggregation committed: commitment_id={commitment_id}, TX: pending")
            return {
                "success": success,
                "commitment_id": commitment_id,
                "message": "Aggregation committed successfully",
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to commit aggregation: {e.code()}: {e.details()}")
            return {
                "success": False,
                "error": str(e.details()) if hasattr(e, 'details') else str(e),
                "code": str(e.code()),
            }
        except AttributeError as e:
            logger.warning(f"CommitAggregation not available in proto (may need regeneration): {e}")
            return {
                "success": False,
                "error": "CommitAggregation not implemented in proto",
            }
        except Exception as e:
            logger.error(f"Error committing aggregation: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
    
    def reveal_aggregation(
        self,
        proposer: str,
        commitment_id: int,
        aggregated_hash: str,
        merkle_root: str,
        training_round_id: int,
    ) -> Dict[str, Any]:
        """
        Reveal committed aggregation.
        
        Args:
            proposer: Proposer address
            commitment_id: Commitment ID from commit step
            aggregated_hash: IPFS hash of aggregated gradient
            merkle_root: Merkle root of included gradients
            training_round_id: Training round ID
            
        Returns:
            Dictionary with success status
        """
        if self.stub is None or tx_pb2 is None:
            logger.warning("Transaction stub not available")
            return {
                "success": False,
                "error": "Transaction stub not available (proto files not loaded)",
            }
        
        try:
            # Create MsgRevealAggregation message
            msg = tx_pb2.MsgRevealAggregation(
                proposer=proposer,
                commitment_id=commitment_id,
                aggregated_hash=aggregated_hash,
                merkle_root=merkle_root,
                training_round_id=training_round_id,
            )
            
            response = self.stub.RevealAggregation(msg)
            success = getattr(response, 'success', False)
            
            logger.info(f"Aggregation revealed: commitment_id={commitment_id}, TX: pending")
            return {
                "success": success,
                "message": "Aggregation revealed successfully",
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to reveal aggregation: {e.code()}: {e.details()}")
            return {
                "success": False,
                "error": str(e.details()) if hasattr(e, 'details') else str(e),
                "code": str(e.code()),
            }
        except AttributeError as e:
            logger.warning(f"RevealAggregation not available in proto (may need regeneration): {e}")
            return {
                "success": False,
                "error": "RevealAggregation not implemented in proto",
            }
        except Exception as e:
            logger.error(f"Error revealing aggregation: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
    
    def submit_aggregation(
        self,
        proposer: str,
        gradient_ids: List[int],
        aggregated_hash: str,
        merkle_root: str,
        training_round_id: int,
    ) -> Dict[str, Any]:
        """
        Submit aggregation to blockchain.
        
        Args:
            proposer: Proposer address
            gradient_ids: List of gradient IDs included in aggregation
            aggregated_hash: IPFS hash of aggregated gradient
            merkle_root: Merkle root of included gradients
            training_round_id: Training round ID
            
        Returns:
            Dictionary with success status and aggregation_id
        """
        if self.stub is None or tx_pb2 is None:
            logger.warning("Transaction stub not available")
            return {
                "success": False,
                "error": "Transaction stub not available (proto files not loaded)",
            }
        
        try:
            # Create MsgSubmitAggregation message
            msg = tx_pb2.MsgSubmitAggregation(
                proposer=proposer,
                gradient_ids=gradient_ids,
                aggregated_hash=aggregated_hash,
                merkle_root=merkle_root,
                training_round_id=training_round_id,
            )
            
            response = self.stub.SubmitAggregation(msg)
            aggregation_id = getattr(response, 'aggregation_id', 0)
            success = getattr(response, 'success', False)
            
            logger.info(f"Aggregation submitted: aggregation_id={aggregation_id}, TX: pending")
            return {
                "success": success,
                "aggregation_id": aggregation_id,
                "message": "Aggregation submitted successfully",
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to submit aggregation: {e.code()}: {e.details()}")
            return {
                "success": False,
                "error": str(e.details()) if hasattr(e, 'details') else str(e),
                "code": str(e.code()),
            }
        except AttributeError as e:
            logger.warning(f"SubmitAggregation not available in proto (may need regeneration): {e}")
            return {
                "success": False,
                "error": "SubmitAggregation not implemented in proto",
            }
        except Exception as e:
            logger.error(f"Error submitting aggregation: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
    
    def close(self):
        """Close gRPC channel."""
        if self.channel:
            self.channel.close()

