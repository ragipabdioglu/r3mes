#!/usr/bin/env python3
"""
Blockchain Query Extensions for Dataset and Adapter Registry

Provides query methods for:
1. Approved datasets from blockchain
2. Active dataset information
3. Approved adapters from blockchain
4. Model version information
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class BlockchainQueryExtensions:
    """
    Mixin class providing dataset and adapter query methods.
    
    Add these methods to BlockchainClient via inheritance or composition.
    """
    
    def query_approved_datasets(self) -> Dict[str, Any]:
        """
        Query all approved datasets from blockchain.
        
        Returns:
            Dictionary with success status and list of datasets
        """
        # Check if query stub is available
        if not hasattr(self, 'query_stub') or self.query_stub is None:
            logger.warning("Query stub not available, returning empty list")
            return {
                "success": True,
                "datasets": [],
                "message": "Query stub not available (simulated mode)",
            }
        
        try:
            # Import query proto
            from remes.remes.v1 import query_pb2
            
            # Query approved datasets
            request = query_pb2.QueryApprovedDatasetsRequest()
            response = self.query_stub.QueryApprovedDatasets(request)
            
            datasets = []
            for dataset in response.datasets:
                datasets.append({
                    "dataset_id": str(dataset.dataset_id),
                    "dataset_ipfs_hash": dataset.dataset_ipfs_hash,
                    "name": dataset.metadata.name if dataset.metadata else "",
                    "version": dataset.metadata.version if dataset.metadata else "1.0.0",
                    "size_bytes": dataset.metadata.size_bytes if dataset.metadata else 0,
                    "checksum": dataset.metadata.checksum if dataset.metadata else "",
                    "category": dataset.metadata.category if dataset.metadata else "",
                    "description": dataset.metadata.description if dataset.metadata else "",
                    "approved_at": dataset.approved_at,
                    "approval_tx_hash": dataset.approval_tx_hash,
                })
            
            return {
                "success": True,
                "datasets": datasets,
            }
            
        except AttributeError as e:
            logger.warning(f"QueryApprovedDatasets not available in proto: {e}")
            return {
                "success": True,
                "datasets": [],
                "message": "QueryApprovedDatasets not implemented in proto",
            }
        except Exception as e:
            logger.error(f"Failed to query approved datasets: {e}")
            return {
                "success": False,
                "datasets": [],
                "error": str(e),
            }
    
    def query_active_dataset(self) -> Dict[str, Any]:
        """
        Query the currently active dataset from blockchain.
        
        Returns:
            Dictionary with success status and active dataset info
        """
        if not hasattr(self, 'query_stub') or self.query_stub is None:
            logger.warning("Query stub not available")
            return {
                "success": False,
                "error": "Query stub not available",
            }
        
        try:
            from remes.remes.v1 import query_pb2
            
            request = query_pb2.QueryActiveDatasetRequest()
            response = self.query_stub.QueryActiveDataset(request)
            
            return {
                "success": True,
                "dataset_id": str(response.dataset_id),
                "ipfs_hash": response.dataset_ipfs_hash,
                "name": response.metadata.name if response.metadata else "",
                "version": response.metadata.version if response.metadata else "1.0.0",
            }
            
        except AttributeError as e:
            logger.warning(f"QueryActiveDataset not available in proto: {e}")
            return {
                "success": False,
                "error": "QueryActiveDataset not implemented in proto",
            }
        except Exception as e:
            logger.error(f"Failed to query active dataset: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    
    def query_approved_adapters(self) -> Dict[str, Any]:
        """
        Query all approved DoRA/LoRA adapters from blockchain.
        
        Returns:
            Dictionary with success status and list of adapters
        """
        if not hasattr(self, 'query_stub') or self.query_stub is None:
            logger.warning("Query stub not available, returning empty list")
            return {
                "success": True,
                "adapters": [],
                "message": "Query stub not available (simulated mode)",
            }
        
        try:
            from remes.remes.v1 import query_pb2
            
            request = query_pb2.QueryApprovedAdaptersRequest()
            response = self.query_stub.QueryApprovedAdapters(request)
            
            adapters = []
            for adapter in response.adapters:
                adapters.append({
                    "adapter_id": str(adapter.adapter_id),
                    "name": adapter.name,
                    "adapter_type": adapter.adapter_type,
                    "version": adapter.version,
                    "ipfs_hash": adapter.ipfs_hash,
                    "checksum": adapter.checksum,
                    "size_bytes": adapter.size_bytes,
                    "compatible_model_versions": list(adapter.compatible_model_versions),
                    "min_model_version": adapter.min_model_version,
                    "max_model_version": adapter.max_model_version,
                    "domain": adapter.domain,
                    "description": adapter.description,
                    "lora_rank": adapter.lora_rank,
                    "lora_alpha": adapter.lora_alpha,
                    "target_modules": list(adapter.target_modules),
                    "approved_at": adapter.approved_at,
                    "approval_tx_hash": adapter.approval_tx_hash,
                    "proposer": adapter.proposer,
                })
            
            return {
                "success": True,
                "adapters": adapters,
            }
            
        except AttributeError as e:
            logger.warning(f"QueryApprovedAdapters not available in proto: {e}")
            return {
                "success": True,
                "adapters": [],
                "message": "QueryApprovedAdapters not implemented in proto",
            }
        except Exception as e:
            logger.error(f"Failed to query approved adapters: {e}")
            return {
                "success": False,
                "adapters": [],
                "error": str(e),
            }
    
    def query_model_versions(self) -> Dict[str, Any]:
        """
        Query available model versions from blockchain.
        
        Returns:
            Dictionary with success status and model version info
        """
        if not hasattr(self, 'query_stub') or self.query_stub is None:
            logger.warning("Query stub not available")
            return {
                "success": False,
                "error": "Query stub not available",
            }
        
        try:
            from remes.remes.v1 import query_pb2
            
            request = query_pb2.QueryModelVersionsRequest()
            response = self.query_stub.QueryModelVersions(request)
            
            versions = []
            for version in response.versions:
                versions.append({
                    "version_number": version.version_number,
                    "model_hash": version.model_hash,
                    "ipfs_path": version.ipfs_path,
                    "architecture": version.architecture,
                    "is_active": version.is_active,
                    "created_at": version.created_at,
                })
            
            return {
                "success": True,
                "versions": versions,
                "active_version": response.active_version,
            }
            
        except AttributeError as e:
            logger.warning(f"QueryModelVersions not available in proto: {e}")
            return {
                "success": False,
                "error": "QueryModelVersions not implemented in proto",
            }
        except Exception as e:
            logger.error(f"Failed to query model versions: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def query_global_model_state(self) -> Dict[str, Any]:
        """
        Query global model state from blockchain.
        
        Returns:
            Dictionary with model hash, version, and update info
        """
        if not hasattr(self, 'query_stub') or self.query_stub is None:
            return {
                "success": False,
                "error": "Query stub not available",
            }
        
        try:
            from remes.remes.v1 import query_pb2
            
            request = query_pb2.QueryGlobalModelStateRequest()
            response = self.query_stub.QueryGlobalModelState(request)
            
            return {
                "success": True,
                "model_hash": response.model_hash,
                "model_version": response.model_version,
                "update_height": response.update_height,
                "ipfs_hash": response.ipfs_hash,
            }
            
        except AttributeError as e:
            logger.warning(f"QueryGlobalModelState not available in proto: {e}")
            # Fallback to GetModelParams
            return self.get_model_params() if hasattr(self, 'get_model_params') else {
                "success": False,
                "error": "QueryGlobalModelState not implemented",
            }
        except Exception as e:
            logger.error(f"Failed to query global model state: {e}")
            return {
                "success": False,
                "error": str(e),
            }


def extend_blockchain_client(client_class):
    """
    Decorator to extend BlockchainClient with query methods.
    
    Usage:
        @extend_blockchain_client
        class BlockchainClient:
            ...
    """
    # Add all methods from BlockchainQueryExtensions to the client class
    for name in dir(BlockchainQueryExtensions):
        if not name.startswith('_'):
            method = getattr(BlockchainQueryExtensions, name)
            if callable(method):
                setattr(client_class, name, method)
    
    return client_class
