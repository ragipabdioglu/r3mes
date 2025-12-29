#!/usr/bin/env python3
"""
IPFS Public Gateway Test

Tests if IPFS-pinned files are accessible via public gateway (e.g., ipfs.io)
to ensure global DHT propagation and open ports.

Usage:
    python scripts/test_ipfs_gateway.py <CID>
    python scripts/test_ipfs_gateway.py --pin-test  # Pin a test file and verify
"""

import argparse
import sys
import os
import time
import requests
from pathlib import Path
from typing import Optional

# Try to import IPFSClient (optional, script can work without it)
IPFSClient = None
try:
    # Add miner-engine to path for IPFS client
    sys.path.insert(0, str(Path(__file__).parent.parent / "miner-engine"))
    from utils.ipfs_client import IPFSClient
except ImportError:
    # IPFSClient is optional for this script - we only need it for --pin-test
    pass


# Public IPFS gateways to test
PUBLIC_GATEWAYS = [
    "https://ipfs.io/ipfs/",
    "https://gateway.pinata.cloud/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
    "https://dweb.link/ipfs/",
]


def test_gateway_access(cid: str, gateway: str, timeout: int = 10) -> tuple[bool, Optional[str]]:
    """
    Test if a CID is accessible via a public gateway.
    
    Args:
        cid: IPFS Content ID
        gateway: Gateway base URL (without CID)
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (is_accessible, error_message)
    """
    url = f"{gateway}{cid}"
    
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        
        if response.status_code == 200:
            return True, None
        elif response.status_code == 404:
            return False, "CID not found (may not be propagated to this gateway yet)"
        else:
            return False, f"HTTP {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, f"Timeout after {timeout}s"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except Exception as e:
        return False, str(e)


def test_all_gateways(cid: str, timeout: int = 10) -> dict[str, tuple[bool, Optional[str]]]:
    """
    Test CID access across all public gateways.
    
    Args:
        cid: IPFS Content ID
        timeout: Request timeout per gateway
    
    Returns:
        Dictionary mapping gateway name to (is_accessible, error_message)
    """
    results = {}
    
    print(f"Testing CID: {cid}")
    print("=" * 60)
    
    for gateway in PUBLIC_GATEWAYS:
        gateway_name = gateway.replace("https://", "").replace("/ipfs/", "")
        print(f"\nTesting gateway: {gateway_name}...", end=" ", flush=True)
        
        is_accessible, error = test_gateway_access(cid, gateway, timeout)
        results[gateway_name] = (is_accessible, error)
        
        if is_accessible:
            print("✅ Accessible")
        else:
            print(f"❌ Not accessible: {error}")
    
    return results


def pin_test_file() -> Optional[str]:
    """
    Pin a test file to IPFS and return its CID.
    
    Returns:
        CID if successful, None otherwise
    """
    print("Creating test file...")
    
    # Create a small test file
    test_content = f"R3MES IPFS Gateway Test - {time.time()}\nThis file is used to test IPFS public gateway connectivity."
    test_file_path = Path("/tmp/r3mes_ipfs_test.txt")
    
    try:
        with open(test_file_path, 'w') as f:
            f.write(test_content)
        
        print(f"Test file created: {test_file_path}")
        
        # Pin to IPFS
        print("Pinning to IPFS...")
        
        if IPFSClient is None:
            print("❌ Error: IPFSClient not available. Install miner-engine dependencies to use --pin-test")
            print("   Run: cd miner-engine && pip install -e .")
            test_file_path.unlink()
            return None
        
        ipfs_client = IPFSClient()
        
        if not ipfs_client.is_connected():
            print("❌ Error: IPFS client not connected")
            test_file_path.unlink()
            return None
        
        result = ipfs_client.add_file(str(test_file_path))
        
        if result and "Hash" in result:
            cid = result["Hash"]
            print(f"✅ File pinned with CID: {cid}")
            
            # Pin to ensure it's available
            ipfs_client.pin(cid)
            print(f"✅ CID {cid} pinned")
            
            # Clean up test file
            test_file_path.unlink()
            
            return cid
        else:
            print("❌ Error: Failed to get CID from IPFS")
            test_file_path.unlink()
            return None
            
    except Exception as e:
        print(f"❌ Error pinning test file: {e}")
        if test_file_path.exists():
            test_file_path.unlink()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test IPFS public gateway connectivity")
    parser.add_argument(
        "cid",
        nargs="?",
        help="IPFS Content ID (CID) to test"
    )
    parser.add_argument(
        "--pin-test",
        action="store_true",
        help="Pin a test file and verify gateway access"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout per gateway in seconds (default: 10)"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Wait N seconds after pinning before testing (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Get CID
    cid = args.cid
    
    if args.pin_test:
        cid = pin_test_file()
        if not cid:
            print("❌ Failed to pin test file")
            sys.exit(1)
        
        if args.wait > 0:
            print(f"\nWaiting {args.wait} seconds for DHT propagation...")
            time.sleep(args.wait)
    elif not cid:
        parser.print_help()
        sys.exit(1)
    
    # Validate CID format
    if not (cid.startswith("Qm") or cid.startswith(("bafy", "bafk"))):
        print(f"⚠️  Warning: CID format may be invalid: {cid}")
        print("   Expected CIDv0 (Qm...) or CIDv1 (bafy.../bafk...)")
    
    # Test all gateways
    results = test_all_gateways(cid, timeout=args.timeout)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    accessible_count = sum(1 for is_ok, _ in results.values() if is_ok)
    total_count = len(results)
    
    print(f"Accessible gateways: {accessible_count}/{total_count}")
    
    if accessible_count == 0:
        print("\n❌ CRITICAL: CID is not accessible via any public gateway")
        print("\nPossible causes:")
        print("  1. IPFS P2P port (4001) is blocked by firewall")
        print("  2. CID has not propagated to DHT yet (wait and try again)")
        print("  3. IPFS node is not connected to public network")
        print("  4. CID is invalid or file was not actually pinned")
        print("\nRecommendations:")
        print("  - Check firewall rules for port 4001")
        print("  - Verify IPFS node is running and connected")
        print("  - Wait a few minutes and test again")
        sys.exit(1)
    elif accessible_count < total_count:
        print(f"\n⚠️  WARNING: CID is only accessible via {accessible_count}/{total_count} gateways")
        print("  This may indicate partial DHT propagation or gateway issues")
        print("  Wait a few minutes and test again")
    else:
        print("\n✅ SUCCESS: CID is accessible via all public gateways")
        print("  Your IPFS node is properly configured for global access")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

