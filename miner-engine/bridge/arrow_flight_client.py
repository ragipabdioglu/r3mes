"""
Apache Arrow Flight Client for Zero-Copy Tensor Transfer

Replaces gRPC for gradient data transfer to enable zero-copy memory sharing.
gRPC still used for metadata/control messages, Arrow Flight for tensor data.
"""

import pyarrow as pa
import pyarrow.flight as flight
from typing import List, Optional, Dict, Any
import torch
import numpy as np


class ArrowFlightClient:
    """Apache Arrow Flight client for zero-copy tensor transfer."""
    
    def __init__(self, host: str = "localhost", port: int = 8815):
        """
        Initialize Arrow Flight client.
        
        Args:
            host: Flight server host
            port: Flight server port
        """
        self.location = flight.Location.for_grpc_tcp(host, port)
        self.client = None
        self._connected = False
        
        # Try to connect (optional - fallback to gRPC if unavailable)
        try:
            self.client = flight.FlightClient(self.location)
            self._connected = True
        except Exception as e:
            print(f"⚠️  Arrow Flight not available: {e}")
            print("   Falling back to gRPC for tensor transfer")
            self._connected = False
    
    def is_connected(self) -> bool:
        """Check if Arrow Flight client is connected."""
        return self._connected
    
    def upload_gradients(self, gradients: List[torch.Tensor], metadata: Dict[str, Any]) -> Optional[str]:
        """
        Upload gradients using Arrow Flight (zero-copy).
        
        Args:
            gradients: List of gradient tensors
            metadata: Metadata dict (miner, model_version, etc.)
        
        Returns:
            Flight descriptor path (used as reference), or None if not connected
        """
        if not self._connected or self.client is None:
            return None
        
        try:
            # Convert PyTorch tensors to Arrow arrays (zero-copy)
            arrays = []
            for grad in gradients:
                # Convert to numpy (zero-copy if possible)
                np_array = grad.detach().cpu().numpy()
                # Create Arrow array (zero-copy from numpy)
                arrow_array = pa.array(np_array.flatten(), type=pa.float32())
                arrays.append(arrow_array)
            
            # Create Arrow table
            table = pa.Table.from_arrays(arrays, names=[f"grad_{i}" for i in range(len(arrays))])
            
            # Create Flight descriptor
            descriptor = flight.FlightDescriptor.for_path(
                f"gradients/{metadata.get('miner', 'unknown')}/{metadata.get('training_round_id', 0)}"
            )
            
            # Upload (zero-copy)
            writer, _ = self.client.do_put(descriptor, table.schema)
            writer.write_table(table)
            writer.close()
            
            return descriptor.path[0].decode()
        except Exception as e:
            print(f"⚠️  Arrow Flight upload failed: {e}")
            print("   Falling back to gRPC")
            return None
    
    def download_gradients(self, path: str) -> Optional[List[torch.Tensor]]:
        """
        Download gradients using Arrow Flight (zero-copy).
        
        Args:
            path: Flight descriptor path
        
        Returns:
            List of gradient tensors, or None if not connected
        """
        if not self._connected or self.client is None:
            return None
        
        try:
            descriptor = flight.FlightDescriptor.for_path(path)
            flight_info = self.client.get_flight_info(descriptor)
            
            # Read (zero-copy)
            reader = self.client.do_get(flight_info.endpoints[0].ticket)
            table = reader.read_all()
            
            # Convert back to PyTorch tensors (zero-copy where possible)
            gradients = []
            for col in table.columns:
                np_array = col.to_numpy()
                tensor = torch.from_numpy(np_array)
                gradients.append(tensor)
            
            return gradients
        except Exception as e:
            print(f"⚠️  Arrow Flight download failed: {e}")
            return None

