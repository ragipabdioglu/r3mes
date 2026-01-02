#!/usr/bin/env python3
"""
R3MES Proposer Aggregator

Production-ready aggregator that:
1. Queries pending gradients from blockchain
2. Downloads gradients from IPFS
3. Aggregates gradients (weighted average)
4. Commits aggregation via MsgCommitAggregation
5. Reveals aggregation via MsgRevealAggregation
6. Submits aggregation via MsgSubmitAggregation
"""

import logging
from typing import List, Dict, Any, Optional
import hashlib
import time

from r3mes.utils.logger import setup_logger
from r3mes.utils.ipfs_manager import IPFSClient
from bridge.blockchain_client import BlockchainClient
from bridge.crypto import derive_address_from_public_key, hex_to_private_key


class ProposerAggregator:
    """Proposer aggregator class."""
    
    def __init__(
        self,
        private_key: str,
        blockchain_url: str = "localhost:9090",
        chain_id: str = "remes-test",
        log_level: str = "INFO",
        use_json_logs: bool = False,
    ):
        """
        Initialize proposer aggregator.
        
        Args:
            private_key: Private key for blockchain transactions
            blockchain_url: gRPC endpoint URL
            chain_id: Chain ID
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            use_json_logs: Whether to use JSON-formatted logs
        """
        # Production localhost validation
        import os
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        if is_production:
            # Extract hostname from blockchain_url (e.g., "localhost:9090" -> "localhost")
            blockchain_host = blockchain_url.split(":")[0] if ":" in blockchain_url else blockchain_url
            if blockchain_host.lower() in ("localhost", "127.0.0.1", "::1") or blockchain_host.startswith("127."):
                raise ValueError(
                    f"blockchain_url cannot use localhost in production: {blockchain_url}. "
                    "Please set blockchain_url to a production gRPC endpoint or use R3MES_NODE_GRPC_URL environment variable."
                )
        
        # Setup logger
        log_level_map = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
        }
        self.logger = setup_logger(
            "r3mes.proposer",
            level=log_level_map.get(log_level.upper(), logging.INFO),
            use_json=use_json_logs,
        )
        
        self.private_key = private_key
        self.chain_id = chain_id
        
        # Blockchain client
        self.blockchain_client = BlockchainClient(
            node_url=blockchain_url,
            chain_id=chain_id,
            private_key=private_key,
        )
        self.proposer_address = derive_address_from_public_key(
            hex_to_private_key(private_key).public_key()
        ) if private_key else None
        
        # IPFS client
        self.ipfs_client = IPFSClient()
        
        self.logger.info("Proposer aggregator initialized")
    
    def query_pending_gradients(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query pending gradients from blockchain.
        
        Args:
            limit: Maximum number of gradients to query
            
        Returns:
            List of pending gradient records with fields: id, gradient_ipfs_hash, status, etc.
        """
        try:
            self.logger.info(f"Querying pending gradients from blockchain (limit: {limit})")
            
            # Use blockchain client to query stored gradients
            if not hasattr(self, 'blockchain_client') or self.blockchain_client is None:
                self.logger.warning("Blockchain client not initialized, cannot query gradients")
                return []
            
            # Query stored gradients with status="pending" using gRPC
            # This uses ListStoredGradient query with pagination
            try:
                from bridge.blockchain_client import BlockchainClient
                from bridge.proto.remes.remes.v1 import query_pb2
                from bridge.proto.cosmos.base.query.v1beta1 import pagination_pb2
                
                if not hasattr(self.blockchain_client, 'query_stub') or self.blockchain_client.query_stub is None:
                    self.logger.warning("Query stub not available, cannot query gradients")
                    return []
                
                # Create query request with pagination
                request = query_pb2.QueryListStoredGradientRequest(
                    pagination=pagination_pb2.PageRequest(
                        limit=limit,
                        offset=0,
                    )
                )
                
                # Query stored gradients
                response = self.blockchain_client.query_stub.ListStoredGradient(request)
                
                # Filter for pending gradients and convert to dict
                pending_gradients = []
                if hasattr(response, 'stored_gradients') and response.stored_gradients:
                    for gradient in response.stored_gradients:
                        # Check if gradient status is "pending"
                        status = getattr(gradient, 'status', '')
                        if status.lower() == 'pending':
                            gradient_dict = {
                                'id': getattr(gradient, 'id', 0),
                                'gradient_ipfs_hash': getattr(gradient, 'gradient_ipfs_hash', ''),
                                'status': status,
                                'miner_address': getattr(gradient, 'miner_address', ''),
                                'submission_time': getattr(gradient, 'submission_time', None),
                            }
                            pending_gradients.append(gradient_dict)
                
                self.logger.info(f"Found {len(pending_gradients)} pending gradients")
                return pending_gradients
                
            except ImportError as e:
                self.logger.error(f"Failed to import proto files: {e}")
                return []
            except Exception as e:
                self.logger.error(f"Error querying gradients from blockchain: {e}", exc_info=True)
                return []
                
        except Exception as e:
            self.logger.error(f"Error querying pending gradients: {e}", exc_info=True)
            return []
    
    def download_gradient(self, gradient_ipfs_hash: str) -> Optional[bytes]:
        """
        Download gradient from IPFS.
        
        Args:
            gradient_ipfs_hash: IPFS hash of gradient
            
        Returns:
            Gradient data as bytes, or None if failed
        """
        try:
            self.logger.debug(f"Downloading gradient from IPFS: {gradient_ipfs_hash}")
            
            # Download from IPFS
            gradient_path = self.ipfs_client.get(gradient_ipfs_hash, output_dir="gradients")
            if not gradient_path:
                self.logger.error(f"Failed to download gradient: {gradient_ipfs_hash}")
                return None
            
            # Read gradient data
            with open(gradient_path, 'rb') as f:
                gradient_data = f.read()
            
            return gradient_data
        except Exception as e:
            self.logger.error(f"Error downloading gradient: {e}", exc_info=True)
            return None
    
    def aggregate_gradients(self, gradients: List[bytes]) -> Optional[bytes]:
        """
        Aggregate gradients using weighted average.
        
        Args:
            gradients: List of gradient data (bytes)
            
        Returns:
            Aggregated gradient data, or None if failed
        """
        try:
            if not gradients:
                self.logger.error("No gradients to aggregate")
                return None
            
            self.logger.info(f"Aggregating {len(gradients)} gradients")
            
            # Import LoRA serializer for deserialization
            from core.serialization import LoRASerializer, LoRAAggregator
            
            serializer = LoRASerializer()
            aggregator = LoRAAggregator()
            
            # Deserialize all gradients
            deserialized_gradients = []
            for i, gradient_bytes in enumerate(gradients):
                try:
                    gradients_dict, metadata = serializer.deserialize_gradients(gradient_bytes)
                    deserialized_gradients.append(gradients_dict)
                    self.logger.debug(f"Deserialized gradient {i+1}/{len(gradients)}")
                except Exception as e:
                    self.logger.error(f"Failed to deserialize gradient {i+1}: {e}", exc_info=True)
                    continue
            
            if not deserialized_gradients:
                self.logger.error("No gradients successfully deserialized")
                return None
            
            # Aggregate gradients using weighted average
            # All gradients have equal weight (1.0) by default
            weights = [1.0] * len(deserialized_gradients)
            
            aggregated_gradients = aggregator.aggregate_lora_states(
                deserialized_gradients,
                weights=weights,
            )
            
            # Serialize aggregated result
            aggregated_bytes = serializer.serialize_gradients(
                aggregated_gradients,
                metadata={
                    'aggregation_time': time.time(),
                    'num_gradients': len(deserialized_gradients),
                }
            )
            
            self.logger.info(f"Aggregation completed: {len(aggregated_bytes) / (1024 * 1024):.4f} MB")
            return aggregated_bytes
            
        except Exception as e:
            self.logger.error(f"Error aggregating gradients: {e}", exc_info=True)
            return None
    
    def commit_aggregation(
        self,
        gradient_ids: List[int],
        training_round_id: int,
        commitment_hash: str
    ) -> Optional[int]:
        """
        Commit aggregation to blockchain.
        
        Args:
            gradient_ids: List of gradient IDs to aggregate
            training_round_id: Training round ID
            commitment_hash: Commitment hash (hash of aggregation result)
            
        Returns:
            Commitment ID, or None if failed
        """
        try:
            self.logger.info(f"Committing aggregation: {len(gradient_ids)} gradients, round={training_round_id}")
            
            # Commit aggregation on blockchain
            if self.blockchain_client and self.proposer_address:
                result = self.blockchain_client.commit_aggregation(
                    proposer=self.proposer_address,
                    gradient_ids=gradient_ids,
                    training_round_id=training_round_id,
                    commitment_hash=commitment_hash,
                )
                if result.get("success", False):
                    commitment_id = result.get("commitment_id", 0)
                    self.logger.info(f"Aggregation committed: commitment_id={commitment_id}")
                    return commitment_id
                else:
                    self.logger.error(f"Failed to commit aggregation: {result.get('error', 'Unknown error')}")
                    return None
            else:
                self.logger.warning("Blockchain client or proposer address not available, skipping commit")
                return None
            
        except Exception as e:
            self.logger.error(f"Error committing aggregation: {e}", exc_info=True)
            return None
    
    def reveal_aggregation(
        self,
        commitment_id: int,
        aggregated_hash: str,
        merkle_root: str,
        training_round_id: int
    ) -> bool:
        """
        Reveal committed aggregation.
        
        Args:
            commitment_id: Commitment ID from commit step
            aggregated_hash: IPFS hash of aggregated gradient
            merkle_root: Merkle root of included gradients
            training_round_id: Training round ID
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Revealing aggregation: commitment_id={commitment_id}")
            
            # Reveal aggregation on blockchain
            if self.blockchain_client and self.proposer_address:
                result = self.blockchain_client.reveal_aggregation(
                    proposer=self.proposer_address,
                    commitment_id=commitment_id,
                    aggregated_hash=aggregated_hash,
                    merkle_root=merkle_root,
                    training_round_id=training_round_id,
                )
                if result.get("success", False):
                    self.logger.info(f"Aggregation revealed: commitment_id={commitment_id}")
                    return True
                else:
                    self.logger.error(f"Failed to reveal aggregation: {result.get('error', 'Unknown error')}")
                    return False
            else:
                self.logger.warning("Blockchain client or proposer address not available, skipping reveal")
                return False
            
        except Exception as e:
            self.logger.error(f"Error revealing aggregation: {e}", exc_info=True)
            return False
    
    def submit_aggregation(
        self,
        gradient_ids: List[int],
        aggregated_hash: str,
        merkle_root: str,
        training_round_id: int
    ) -> Optional[int]:
        """
        Submit aggregation to blockchain.
        
        Args:
            gradient_ids: List of gradient IDs included in aggregation
            aggregated_hash: IPFS hash of aggregated gradient
            merkle_root: Merkle root of included gradients
            training_round_id: Training round ID
            
        Returns:
            Aggregation ID, or None if failed
        """
        try:
            self.logger.info(f"Submitting aggregation: {len(gradient_ids)} gradients, round={training_round_id}")
            
            # Submit aggregation on blockchain
            if self.blockchain_client and self.proposer_address:
                result = self.blockchain_client.submit_aggregation(
                    proposer=self.proposer_address,
                    gradient_ids=gradient_ids,
                    aggregated_hash=aggregated_hash,
                    merkle_root=merkle_root,
                    training_round_id=training_round_id,
                )
                if result.get("success", False):
                    aggregation_id = result.get("aggregation_id", 0)
                    self.logger.info(f"Aggregation submitted: aggregation_id={aggregation_id}")
                    return aggregation_id
                else:
                    self.logger.error(f"Failed to submit aggregation: {result.get('error', 'Unknown error')}")
                    return None
            else:
                self.logger.warning("Blockchain client or proposer address not available, skipping submission")
                return None
            
        except Exception as e:
            self.logger.error(f"Error submitting aggregation: {e}", exc_info=True)
            return None
    
    def aggregate_and_submit(self, gradient_ids: List[int], training_round_id: int) -> bool:
        """
        Complete aggregation workflow: download, aggregate, commit, reveal, submit.
        
        Args:
            gradient_ids: List of gradient IDs to aggregate
            training_round_id: Training round ID
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Starting aggregation workflow: {len(gradient_ids)} gradients")
            
            # 1. Download gradients from IPFS
            gradients_data = []
            gradient_hashes = []
            
            for grad_id in gradient_ids:
                # Get gradient IPFS hash from blockchain query
                gradient_info = self.blockchain_client.get_stored_gradient_by_id(grad_id)
                if not gradient_info:
                    self.logger.warning(f"Failed to get gradient info for ID {grad_id}")
                    continue
                
                gradient_ipfs_hash = gradient_info.get('gradient_ipfs_hash', '')
                if not gradient_ipfs_hash:
                    self.logger.warning(f"No IPFS hash for gradient ID {grad_id}")
                    continue
                
                # Download gradient data
                gradient_data = self.download_gradient(gradient_ipfs_hash)
                if gradient_data:
                    gradients_data.append(gradient_data)
                    gradient_hashes.append(gradient_ipfs_hash)
                    self.logger.debug(f"Downloaded gradient {grad_id}: {len(gradient_data)} bytes")
                else:
                    self.logger.warning(f"Failed to download gradient {grad_id} from IPFS: {gradient_ipfs_hash}")
            
            if not gradients_data:
                self.logger.error("No gradients downloaded")
                return False
            
            # 2. Aggregate gradients
            aggregated_data = self.aggregate_gradients(gradients_data)
            if not aggregated_data:
                self.logger.error("Failed to aggregate gradients")
                return False
            
            # 3. Upload aggregated result to IPFS
            aggregated_hash = self.ipfs_client.add_bytes(aggregated_data)
            if not aggregated_hash:
                self.logger.error("Failed to upload aggregated gradient to IPFS")
                return False
            
            # 4. Compute Merkle root (simplified)
            merkle_root = hashlib.sha256(b''.join(gradients_data)).hexdigest()
            
            # 5. Commit aggregation
            commitment_hash = hashlib.sha256(aggregated_data).hexdigest()
            commitment_id = self.commit_aggregation(gradient_ids, training_round_id, commitment_hash)
            
            if commitment_id:
                # 6. Reveal aggregation
                if not self.reveal_aggregation(commitment_id, aggregated_hash, merkle_root, training_round_id):
                    self.logger.error("Failed to reveal aggregation")
                    return False
            
            # 7. Submit aggregation (or submit directly if commit-reveal not used)
            aggregation_id = self.submit_aggregation(gradient_ids, aggregated_hash, merkle_root, training_round_id)
            
            if aggregation_id:
                self.logger.info(f"Aggregation submitted successfully: aggregation_id={aggregation_id}")
                return True
            else:
                self.logger.error("Failed to submit aggregation")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in aggregation workflow: {e}", exc_info=True)
            return False


def main():
    """Main entry point for proposer aggregator."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="R3MES Proposer Aggregator")
    parser.add_argument("--private-key", required=True, help="Private key for blockchain transactions")
    parser.add_argument("--blockchain-url", default="localhost:9090", help="Blockchain gRPC URL")
    parser.add_argument("--chain-id", default="remes-test", help="Chain ID")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--json-logs", action="store_true", help="Use JSON-formatted logs")
    parser.add_argument("--training-round-id", type=int, default=1, help="Training round ID to aggregate")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of gradients to query")
    
    args = parser.parse_args()
    
    try:
        # Create proposer aggregator
        aggregator = ProposerAggregator(
            private_key=args.private_key,
            blockchain_url=args.blockchain_url,
            chain_id=args.chain_id,
            log_level=args.log_level,
            use_json_logs=args.json_logs,
        )
        
        # Query pending gradients
        pending_gradients = aggregator.query_pending_gradients(limit=args.limit)
        if not pending_gradients:
            print("No pending gradients found")
            return
        
        print(f"Found {len(pending_gradients)} pending gradients")
        
        # Extract gradient IDs
        gradient_ids = [grad['id'] for grad in pending_gradients]
        
        # Run aggregation workflow
        success = aggregator.aggregate_and_submit(gradient_ids, args.training_round_id)
        
        if success:
            print("Aggregation completed successfully")
        else:
            print("Aggregation failed")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()