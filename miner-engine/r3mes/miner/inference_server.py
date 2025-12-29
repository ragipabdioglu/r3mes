"""
Inference Server for Serving Node

HTTP server that handles chat inference requests from the backend.
"""

import http.server
import socketserver
import json
import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class InferenceHTTPHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for inference endpoint."""
    
    def __init__(self, *args, miner_engine=None, **kwargs):
        """Initialize handler with miner engine reference."""
        self.miner_engine = miner_engine
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        """Handle POST requests for chat inference."""
        if self.path == "/chat":
            self._handle_chat()
        else:
            self._send_error(404, "Not Found")
    
    def do_GET(self):
        """Handle GET requests for health check."""
        if self.path == "/health":
            self._handle_health()
        else:
            self._send_error(404, "Not Found")
    
    def _handle_chat(self):
        """Handle /chat endpoint for inference."""
        if not self.miner_engine:
            self._send_error(503, "Miner engine not initialized")
            return
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_error(400, "Request body is required")
                return
            
            body = self.rfile.read(content_length)
            request_data = json.loads(body.decode('utf-8'))
            
            message = request_data.get('message')
            wallet_address = self.headers.get('X-Wallet-Address') or request_data.get('wallet_address')
            
            if not message:
                self._send_error(400, "Message is required")
                return
            
            # Get adapter name from request (backend sends lora_name)
            adapter_name = request_data.get('lora_name') or request_data.get('adapter_name')
            
            # Run inference
            logger.info(f"Processing inference request: message length={len(message)}, adapter={adapter_name}")
            
            # Check if miner engine has inference capability
            if not hasattr(self.miner_engine, 'generate_inference'):
                self._send_error(503, "Inference not available on this miner")
                return
            
            # Generate response (streaming)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            # Stream inference response
            try:
                for token in self.miner_engine.generate_inference(message, adapter_name=adapter_name):
                    if token:
                        # Send token as chunk
                        chunk = token.encode('utf-8')
                        chunk_size = hex(len(chunk))[2:].encode('utf-8')
                        self.wfile.write(chunk_size + b'\r\n')
                        self.wfile.write(chunk + b'\r\n')
                        self.wfile.flush()
                
                # Send final empty chunk
                self.wfile.write(b'0\r\n\r\n')
            except Exception as e:
                logger.error(f"Error during inference streaming: {e}", exc_info=True)
                # Try to send error chunk
                try:
                    error_msg = f"\n[Error: {str(e)}]"
                    chunk = error_msg.encode('utf-8')
                    chunk_size = hex(len(chunk))[2:].encode('utf-8')
                    self.wfile.write(chunk_size + b'\r\n')
                    self.wfile.write(chunk + b'\r\n')
                    self.wfile.write(b'0\r\n\r\n')
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # Client disconnected, nothing we can do
                    pass
                
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON in request body")
        except Exception as e:
            logger.error(f"Error handling chat request: {e}", exc_info=True)
            self._send_error(500, f"Internal Server Error: {str(e)}")
    
    def _handle_health(self):
        """Handle /health endpoint."""
        health = {
            "status": "ok" if self.miner_engine else "not_initialized",
            "available_loras": []
        }
        
        if self.miner_engine and hasattr(self.miner_engine, 'available_lora_list'):
            health["available_loras"] = getattr(self.miner_engine, 'available_lora_list', [])
        
        self._send_json(health)
    
    def _send_json(self, data: dict):
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
    
    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        error_response = {"error": message}
        self.wfile.write(json.dumps(error_response).encode("utf-8"))
    
    def log_message(self, format, *args):
        """Override to use logger instead of stderr."""
        logger.debug(f"{self.address_string()} - {format % args}")


def create_inference_handler(miner_engine):
    """Create handler class with miner engine bound."""
    def handler(*args, **kwargs):
        return InferenceHTTPHandler(*args, miner_engine=miner_engine, **kwargs)
    return handler


def start_inference_server(
    port: Optional[int] = None,
    host: Optional[str] = None,
    miner_engine=None
):
    """
    Start HTTP server for inference endpoint.
    
    Args:
        port: Port to listen on (default: from R3MES_SERVING_NODE_PORT env var or 8081)
        host: Host to bind to (default: from R3MES_SERVING_NODE_HOST env var or 0.0.0.0)
        miner_engine: MinerEngine instance for inference
    """
    # Get port from parameter, environment variable, or default
    if port is None:
        port = int(os.getenv("R3MES_SERVING_NODE_PORT", "8081"))
    
    # Get host from parameter, environment variable, or default
    if host is None:
        host = os.getenv("R3MES_SERVING_NODE_HOST", "0.0.0.0")
    if miner_engine is None:
        logger.error("Miner engine is required for inference server")
        return None
    
    try:
        # Allow address reuse to prevent "Address already in use" errors
        socketserver.TCPServer.allow_reuse_address = True
        
        # Create handler with miner engine
        handler = create_inference_handler(miner_engine)
        
        with socketserver.TCPServer((host, port), handler) as httpd:
            logger.info(f"Inference HTTP server started on {host}:{port}")
            logger.info(f"Chat endpoint: http://{host}:{port}/chat")
            logger.info(f"Health endpoint: http://{host}:{port}/health")
            print(f"✅ Inference HTTP server started on http://{host}:{port}/chat")
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            logger.warning(f"Port {port} already in use, inference server may already be running")
            print(f"⚠️  Port {port} already in use")
        else:
            logger.error(f"Failed to start inference server: {e}")
            raise
    except Exception as e:
        logger.error(f"Error starting inference server: {e}", exc_info=True)
        raise


def start_inference_server_thread(
    port: Optional[int] = None,
    host: Optional[str] = None,
    miner_engine=None,
    daemon: bool = True
) -> threading.Thread:
    """
    Start inference server in a separate thread.
    
    Args:
        port: Port to listen on (default: 8081)
        host: Host to bind to (default: 0.0.0.0)
        miner_engine: MinerEngine instance for inference
        daemon: Whether thread should be daemon (default: True)
        
    Returns:
        Thread object
    """
    server_thread = threading.Thread(
        target=start_inference_server,
        args=(port, host, miner_engine),
        daemon=daemon
    )
    server_thread.start()
    return server_thread


if __name__ == "__main__":
    # For testing
    logging.basicConfig(level=logging.INFO)
    # Would need a mock miner engine for testing
    # start_inference_server(port=8081, miner_engine=None)

