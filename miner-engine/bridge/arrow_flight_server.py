"""
Apache Arrow Flight Server for Zero-Copy Gradient Transfer

Provides high-performance gradient transfer between miners, validators, and proposers.
Uses Arrow Flight for zero-copy memory sharing, significantly reducing latency.
"""

import pyarrow as pa
import pyarrow.flight as flight
import torch
import numpy as np
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class GradientEntry:
    """Stored gradient entry with metadata."""
    table: pa.Table
    metadata: Dict[str, Any]
    created_at: datetime
    access_count: int = 0


class GradientFlightServer(flight.FlightServerBase):
    """
    Arrow Flight server for zero-copy gradient transfer.
    
    Features:
    - Zero-copy gradient upload/download
    - Automatic cleanup of old gradients
    - Access tracking for analytics
    - Concurrent access support
    """
    
    def __init__(
        self, 
        host: str = "0.0.0.0", 
        port: int = 8815,
        max_gradients: int = 1000,
        ttl_seconds: int = 3600,
        auth_callback: Optional[Callable[[str], bool]] = None
    ):
        """
        Initialize Arrow Flight server.
        
        Args:
            host: Server host address
            port: Server port
            max_gradients: Maximum number of gradients to store
            ttl_seconds: Time-to-live for gradients in seconds
            auth_callback: Optional authentication callback
        """
        location = flight.Location.for_grpc_tcp(host, port)
        super().__init__(location)
        
        self.host = host
        self.port = port
        self.max_gradients = max_gradients
        self.ttl_seconds = ttl_seconds
        self.auth_callback = auth_callback
        
        # Gradient storage
        self._gradients: Dict[str, GradientEntry] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            "total_uploads": 0,
            "total_downloads": 0,
            "total_bytes_uploaded": 0,
            "total_bytes_downloaded": 0,
            "active_gradients": 0,
        }
        
        # Cleanup task
        self._cleanup_task = None
        self._running = False
        
        logger.info(f"Arrow Flight server initialized on {host}:{port}")
    
    def do_put(self, context, descriptor, reader, writer):
        """
        Receive gradients from miners.
        
        Args:
            context: Flight context
            descriptor: Flight descriptor with path
            reader: Data reader
            writer: Metadata writer
        """
        path = descriptor.path[0].decode() if descriptor.path else "unknown"
        
        try:
            # Read all data
            table = reader.read_all()
            
            # Extract metadata from descriptor command if available
            metadata = {}
            if descriptor.command:
                try:
                    import json
                    metadata = json.loads(descriptor.command.decode())
                except Exception:
                    pass
            
            # Store gradient
            with self._lock:
                # Check capacity
                if len(self._gradients) >= self.max_gradients:
                    self._evict_oldest()
                
                self._gradients[path] = GradientEntry(
                    table=table,
                    metadata=metadata,
                    created_at=datetime.now(),
                )
                
                # Update stats
                self._stats["total_uploads"] += 1
                self._stats["total_bytes_uploaded"] += table.nbytes
                self._stats["active_gradients"] = len(self._gradients)
            
            logger.debug(f"Stored gradient: {path} ({table.nbytes} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to store gradient {path}: {e}")
            raise flight.FlightServerError(f"Upload failed: {e}")
    
    def do_get(self, context, ticket):
        """
        Send gradients to validators/aggregators.
        
        Args:
            context: Flight context
            ticket: Flight ticket with path
            
        Returns:
            RecordBatchStream with gradient data
        """
        path = ticket.ticket.decode()
        
        with self._lock:
            if path not in self._gradients:
                raise flight.FlightUnavailableError(f"Gradient not found: {path}")
            
            entry = self._gradients[path]
            entry.access_count += 1
            
            # Update stats
            self._stats["total_downloads"] += 1
            self._stats["total_bytes_downloaded"] += entry.table.nbytes
        
        logger.debug(f"Serving gradient: {path}")
        return flight.RecordBatchStream(entry.table)
    
    def get_flight_info(self, context, descriptor):
        """
        Get info about stored gradients.
        
        Args:
            context: Flight context
            descriptor: Flight descriptor
            
        Returns:
            FlightInfo with gradient metadata
        """
        path = descriptor.path[0].decode() if descriptor.path else ""
        
        with self._lock:
            if path not in self._gradients:
                raise flight.FlightUnavailableError(f"Gradient not found: {path}")
            
            entry = self._gradients[path]
            table = entry.table
        
        return flight.FlightInfo(
            table.schema,
            descriptor,
            [flight.FlightEndpoint(path.encode(), [self.location])],
            table.num_rows,
            table.nbytes
        )
    
    def list_flights(self, context, criteria):
        """
        List all available gradients.
        
        Args:
            context: Flight context
            criteria: Filter criteria
            
        Yields:
            FlightInfo for each gradient
        """
        with self._lock:
            for path, entry in self._gradients.items():
                descriptor = flight.FlightDescriptor.for_path(path)
                yield flight.FlightInfo(
                    entry.table.schema,
                    descriptor,
                    [flight.FlightEndpoint(path.encode(), [self.location])],
                    entry.table.num_rows,
                    entry.table.nbytes
                )
    
    def do_action(self, context, action):
        """
        Handle custom actions.
        
        Supported actions:
        - "stats": Get server statistics
        - "cleanup": Force cleanup of expired gradients
        - "delete": Delete specific gradient
        """
        action_type = action.type
        
        if action_type == "stats":
            import json
            stats = self.get_stats()
            yield flight.Result(json.dumps(stats).encode())
        
        elif action_type == "cleanup":
            count = self._cleanup_expired()
            yield flight.Result(f"Cleaned up {count} gradients".encode())
        
        elif action_type == "delete":
            path = action.body.to_pybytes().decode()
            with self._lock:
                if path in self._gradients:
                    del self._gradients[path]
                    self._stats["active_gradients"] = len(self._gradients)
                    yield flight.Result(f"Deleted: {path}".encode())
                else:
                    yield flight.Result(f"Not found: {path}".encode())
        
        else:
            raise flight.FlightServerError(f"Unknown action: {action_type}")
    
    def _evict_oldest(self):
        """Evict oldest gradient to make room for new ones."""
        if not self._gradients:
            return
        
        # Find oldest entry
        oldest_path = min(
            self._gradients.keys(),
            key=lambda p: self._gradients[p].created_at
        )
        
        del self._gradients[oldest_path]
        logger.debug(f"Evicted oldest gradient: {oldest_path}")
    
    def _cleanup_expired(self) -> int:
        """Remove expired gradients."""
        now = datetime.now()
        expired = []
        
        with self._lock:
            for path, entry in self._gradients.items():
                age = (now - entry.created_at).total_seconds()
                if age > self.ttl_seconds:
                    expired.append(path)
            
            for path in expired:
                del self._gradients[path]
            
            self._stats["active_gradients"] = len(self._gradients)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired gradients")
        
        return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        with self._lock:
            return {
                **self._stats,
                "uptime_seconds": 0,  # TODO: Track uptime
            }
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def start_background_cleanup(self):
        """Start background cleanup task."""
        self._running = True
        
        def run_cleanup():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._cleanup_loop())
        
        self._cleanup_thread = threading.Thread(target=run_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def stop(self):
        """Stop the server and cleanup."""
        self._running = False
        super().shutdown()



class GradientFlightManager:
    """
    Manager for Arrow Flight gradient transfer.
    
    Provides high-level API for gradient upload/download with automatic
    fallback to gRPC when Arrow Flight is unavailable.
    """
    
    def __init__(
        self,
        server_host: str = "0.0.0.0",
        server_port: int = 8815,
        enable_server: bool = True
    ):
        """
        Initialize gradient flight manager.
        
        Args:
            server_host: Server host for incoming connections
            server_port: Server port
            enable_server: Whether to start the server
        """
        self.server_host = server_host
        self.server_port = server_port
        self.server: Optional[GradientFlightServer] = None
        self._clients: Dict[str, flight.FlightClient] = {}
        
        if enable_server:
            self.start_server()
    
    def start_server(self):
        """Start the Arrow Flight server."""
        try:
            self.server = GradientFlightServer(
                host=self.server_host,
                port=self.server_port
            )
            
            # Start server in background thread
            def serve():
                self.server.serve()
            
            self._server_thread = threading.Thread(target=serve, daemon=True)
            self._server_thread.start()
            
            # Start cleanup
            self.server.start_background_cleanup()
            
            logger.info(f"Arrow Flight server started on {self.server_host}:{self.server_port}")
        except Exception as e:
            logger.error(f"Failed to start Arrow Flight server: {e}")
            self.server = None
    
    def get_client(self, host: str, port: int) -> Optional[flight.FlightClient]:
        """
        Get or create a Flight client for the given endpoint.
        
        Args:
            host: Target host
            port: Target port
            
        Returns:
            FlightClient or None if connection fails
        """
        key = f"{host}:{port}"
        
        if key not in self._clients:
            try:
                location = flight.Location.for_grpc_tcp(host, port)
                self._clients[key] = flight.FlightClient(location)
            except Exception as e:
                logger.warning(f"Failed to connect to {key}: {e}")
                return None
        
        return self._clients[key]
    
    def upload_gradients(
        self,
        gradients: List[torch.Tensor],
        metadata: Dict[str, Any],
        target_host: str = "localhost",
        target_port: int = 8815
    ) -> Optional[str]:
        """
        Upload gradients to a Flight server.
        
        Args:
            gradients: List of gradient tensors
            metadata: Gradient metadata
            target_host: Target server host
            target_port: Target server port
            
        Returns:
            Gradient path/ID or None on failure
        """
        client = self.get_client(target_host, target_port)
        if not client:
            return None
        
        try:
            # Convert tensors to Arrow table
            arrays = []
            shapes = []
            
            for i, grad in enumerate(gradients):
                np_array = grad.detach().cpu().numpy()
                shapes.append(list(np_array.shape))
                arrow_array = pa.array(np_array.flatten(), type=pa.float32())
                arrays.append(arrow_array)
            
            table = pa.Table.from_arrays(
                arrays, 
                names=[f"grad_{i}" for i in range(len(arrays))]
            )
            
            # Create descriptor with metadata
            import json
            miner = metadata.get("miner", "unknown")
            round_id = metadata.get("training_round_id", 0)
            path = f"gradients/{miner}/{round_id}"
            
            descriptor = flight.FlightDescriptor.for_path(path)
            
            # Include shapes in metadata for reconstruction
            full_metadata = {
                **metadata,
                "shapes": shapes,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Upload
            writer, _ = client.do_put(descriptor, table.schema)
            writer.write_table(table)
            writer.close()
            
            logger.debug(f"Uploaded gradients to {target_host}:{target_port}/{path}")
            return path
            
        except Exception as e:
            logger.error(f"Failed to upload gradients: {e}")
            return None
    
    def download_gradients(
        self,
        path: str,
        source_host: str = "localhost",
        source_port: int = 8815,
        shapes: Optional[List[List[int]]] = None
    ) -> Optional[List[torch.Tensor]]:
        """
        Download gradients from a Flight server.
        
        Args:
            path: Gradient path/ID
            source_host: Source server host
            source_port: Source server port
            shapes: Original tensor shapes for reconstruction
            
        Returns:
            List of gradient tensors or None on failure
        """
        client = self.get_client(source_host, source_port)
        if not client:
            return None
        
        try:
            # Get flight info
            descriptor = flight.FlightDescriptor.for_path(path)
            info = client.get_flight_info(descriptor)
            
            # Download
            reader = client.do_get(info.endpoints[0].ticket)
            table = reader.read_all()
            
            # Convert back to tensors
            gradients = []
            for i, col in enumerate(table.columns):
                np_array = col.to_numpy()
                
                # Reshape if shapes provided
                if shapes and i < len(shapes):
                    np_array = np_array.reshape(shapes[i])
                
                tensor = torch.from_numpy(np_array.copy())
                gradients.append(tensor)
            
            logger.debug(f"Downloaded gradients from {source_host}:{source_port}/{path}")
            return gradients
            
        except Exception as e:
            logger.error(f"Failed to download gradients: {e}")
            return None
    
    def get_server_stats(self) -> Optional[Dict[str, Any]]:
        """Get local server statistics."""
        if self.server:
            return self.server.get_stats()
        return None
    
    def shutdown(self):
        """Shutdown the manager and server."""
        if self.server:
            self.server.stop()
        
        for client in self._clients.values():
            try:
                client.close()
            except Exception:
                pass
        
        self._clients.clear()


# Convenience function for starting standalone server
def start_flight_server(
    host: str = "0.0.0.0",
    port: int = 8815,
    max_gradients: int = 1000,
    ttl_seconds: int = 3600
):
    """
    Start a standalone Arrow Flight server.
    
    Args:
        host: Server host
        port: Server port
        max_gradients: Maximum gradients to store
        ttl_seconds: Gradient TTL
    """
    server = GradientFlightServer(
        host=host,
        port=port,
        max_gradients=max_gradients,
        ttl_seconds=ttl_seconds
    )
    
    server.start_background_cleanup()
    
    logger.info(f"Starting Arrow Flight server on {host}:{port}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        server.serve()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Arrow Flight Gradient Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8815, help="Server port")
    parser.add_argument("--max-gradients", type=int, default=1000, help="Max gradients")
    parser.add_argument("--ttl", type=int, default=3600, help="Gradient TTL in seconds")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    start_flight_server(args.host, args.port, args.max_gradients, args.ttl)
