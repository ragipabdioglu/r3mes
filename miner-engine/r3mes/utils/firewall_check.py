"""
Firewall Port Check Utility

Checks if required P2P ports are accessible for blockchain and IPFS.
Windows Firewall may block these ports silently.
"""

import os
import socket
import logging
from typing import List, Tuple, Optional
import platform

logger = logging.getLogger(__name__)

# Required ports for R3MES mining
REQUIRED_PORTS = [
    (26656, "Blockchain P2P (Tendermint)"),
    (4001, "IPFS P2P"),
]

# Optional ports (local only, but should be checked)
OPTIONAL_PORTS = [
    (5001, "IPFS API (local only)"),
    (1317, "Cosmos SDK REST API (local only)"),
    (9090, "gRPC (local only)"),
]


def check_port_listening(port: int, host: str = "0.0.0.0") -> bool:
    """
    Check if a port is listening on the specified host.
    
    Args:
        port: Port number to check
        host: Host to bind to (default: 0.0.0.0 for all interfaces)
    
    Returns:
        True if port is listening, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(1)
            return True
    except OSError as e:
        # Port is already in use or permission denied
        if e.errno == 98 or e.errno == 48:  # Address already in use (Linux/Windows)
            return True  # Port is in use, so it's "listening"
        elif e.errno == 13:  # Permission denied
            logger.warning(f"Permission denied for port {port} (may need admin rights)")
            return False
        else:
            logger.debug(f"Port {port} check failed: {e}")
            return False
    except Exception as e:
        logger.debug(f"Unexpected error checking port {port}: {e}")
        return False


def check_port_external_connectivity(port: int, timeout: float = 2.0) -> bool:
    """
    Check if a port is accessible from external connections.
    
    This is a simplified check - in production, you'd want to use
    an external service to verify actual connectivity.
    
    Args:
        port: Port number to check
        timeout: Connection timeout in seconds
    
    Returns:
        True if port appears accessible, False otherwise
    """
    # For P2P ports, we can't easily test external connectivity without
    # an external service. This is a placeholder for future implementation.
    # For now, we just check if the port can be bound (not already in use).
    return check_port_listening(port)


def check_firewall_ports(required_ports: Optional[List[Tuple[int, str]]] = None) -> Tuple[bool, List[str]]:
    """
    Check if required ports are available and provide firewall warnings.
    
    Args:
        required_ports: List of (port, description) tuples to check.
                       If None, uses default REQUIRED_PORTS.
    
    Returns:
        Tuple of (all_ok, warnings)
        - all_ok: True if all required ports are available
        - warnings: List of warning messages
    """
    if required_ports is None:
        required_ports = REQUIRED_PORTS
    
    warnings = []
    all_ok = True
    
    # Detect OS
    system = platform.system()
    
    for port, description in required_ports:
        is_listening = check_port_listening(port)
        
        if not is_listening:
            # Port is not listening - may be blocked by firewall
            if system == "Windows":
                warning = (
                    f"⚠️  Port {port} ({description}) may be blocked by Windows Firewall.\n"
                    f"   When the miner starts, Windows Firewall may prompt for access.\n"
                    f"   Please click 'Allow Access' to enable P2P connectivity."
                )
            else:
                warning = (
                    f"⚠️  Port {port} ({description}) is not accessible.\n"
                    f"   Check your firewall rules and ensure the port is open."
                )
            warnings.append(warning)
            all_ok = False
            logger.warning(f"Port {port} ({description}) check failed")
        else:
            logger.debug(f"Port {port} ({description}) is available")
    
    return all_ok, warnings


def print_firewall_warnings(warnings: List[str]):
    """Print firewall warnings to console."""
    if warnings:
        print("\n" + "=" * 60)
        print("FIREWALL WARNING")
        print("=" * 60)
        for warning in warnings:
            print(warning)
        print("=" * 60)
        print()


def check_firewall_and_warn(required_ports: Optional[List[Tuple[int, str]]] = None) -> bool:
    """
    Check firewall ports and print warnings if issues found.
    
    Args:
        required_ports: List of (port, description) tuples to check.
    
    Returns:
        True if all ports OK, False if warnings issued
    """
    all_ok, warnings = check_firewall_ports(required_ports)
    
    if warnings:
        print_firewall_warnings(warnings)
    
    return all_ok

