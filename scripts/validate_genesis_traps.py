#!/usr/bin/env python3
"""
Validate Genesis Trap Jobs

Validates the generated trap jobs JSON file to ensure:
1. All required fields are present
2. Hash formats are valid
3. IPFS hash formats are valid
4. Data integrity checks

Usage:
    python scripts/validate_genesis_traps.py genesis_traps.json
"""

import argparse
import json
import sys
from pathlib import Path
import re


def validate_ipfs_hash(hash_str: str) -> bool:
    """Validate IPFS hash format."""
    # CIDv0: Qm... (44 characters)
    # CIDv1: bafy... or bafk... (59+ characters)
    if not hash_str:
        return False
    
    if hash_str.startswith("Qm") and len(hash_str) == 46:
        return True
    elif hash_str.startswith(("bafy", "bafk")) and len(hash_str) >= 59:
        return True
    
    return False


def validate_hex_hash(hash_str: str, expected_length: int = 64) -> bool:
    """Validate hex hash format (SHA256)."""
    if not hash_str:
        return False
    
    # SHA256 produces 64 hex characters (256 bits)
    if len(hash_str) != expected_length:
        return False
    
    # Check if all characters are hex
    if not re.match(r'^[0-9a-fA-F]+$', hash_str):
        return False
    
    return True


def validate_vault_entry(entry: dict, index: int) -> tuple[bool, list[str]]:
    """
    Validate a single GenesisVaultEntry.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    required_fields = [
        "entry_id",
        "data_hash",
        "expected_gradient_hash",
        "expected_fingerprint",
        "gpu_architecture",
        "created_height",
        "usage_count",
        "last_used_height",
    ]
    
    # Check required fields
    for field in required_fields:
        if field not in entry:
            errors.append(f"Missing required field: {field}")
    
    # Validate entry_id
    if "entry_id" in entry:
        if not isinstance(entry["entry_id"], int):
            errors.append("entry_id must be an integer")
        elif entry["entry_id"] <= 0:
            errors.append("entry_id must be > 0")
    
    # Validate IPFS hash (data_hash)
    if "data_hash" in entry:
        if not validate_ipfs_hash(entry["data_hash"]):
            errors.append(f"Invalid IPFS hash format for data_hash: {entry['data_hash']}")
    
    # Validate gradient hash (SHA256 hex)
    if "expected_gradient_hash" in entry:
        if not validate_hex_hash(entry["expected_gradient_hash"]):
            errors.append(f"Invalid gradient hash format (expected 64 hex chars): {entry['expected_gradient_hash']}")
    
    # Validate fingerprint JSON
    if "expected_fingerprint" in entry:
        fingerprint_str = entry["expected_fingerprint"]
        if not fingerprint_str:
            errors.append("expected_fingerprint cannot be empty")
        else:
            try:
                fingerprint = json.loads(fingerprint_str)
                if "top_k" not in fingerprint:
                    errors.append("expected_fingerprint missing 'top_k' field")
                if "indices" not in fingerprint:
                    errors.append("expected_fingerprint missing 'indices' field")
                if "values" not in fingerprint:
                    errors.append("expected_fingerprint missing 'values' field")
                if "shape" not in fingerprint:
                    errors.append("expected_fingerprint missing 'shape' field")
                
                # Validate indices and values have same length
                if "indices" in fingerprint and "values" in fingerprint:
                    if len(fingerprint["indices"]) != len(fingerprint["values"]):
                        errors.append(f"expected_fingerprint indices and values must have same length (got {len(fingerprint['indices'])} vs {len(fingerprint['values'])})")
            except json.JSONDecodeError as e:
                errors.append(f"expected_fingerprint is not valid JSON: {e}")
    
    # Validate heights
    if "created_height" in entry:
        if not isinstance(entry["created_height"], int):
            errors.append("created_height must be an integer")
        elif entry["created_height"] < 0:
            errors.append("created_height must be >= 0")
    
    if "usage_count" in entry:
        if not isinstance(entry["usage_count"], int):
            errors.append("usage_count must be an integer")
        elif entry["usage_count"] < 0:
            errors.append("usage_count must be >= 0")
    
    if "last_used_height" in entry:
        if not isinstance(entry["last_used_height"], int):
            errors.append("last_used_height must be an integer")
        elif entry["last_used_height"] < 0:
            errors.append("last_used_height must be >= 0")
    
    # Validate encrypted
    if "encrypted" in entry:
        if not isinstance(entry["encrypted"], bool):
            errors.append("encrypted must be a boolean")
    
    # Validate metadata fields (optional)
    if "dataset_seed" in entry:
        if not isinstance(entry["dataset_seed"], int):
            errors.append("dataset_seed must be an integer")
    
    if "training_loss" in entry:
        if not isinstance(entry["training_loss"], (int, float)):
            errors.append("training_loss must be a number")
    
    return len(errors) == 0, errors


def validate_genesis_traps(json_path: str) -> bool:
    """
    Validate the entire genesis traps JSON file.
    
    Returns:
        True if valid, False otherwise
    """
    path = Path(json_path)
    
    if not path.exists():
        print(f"❌ Error: File not found: {json_path}")
        return False
    
    print(f"Validating: {json_path}")
    print("=" * 60)
    
    # Load JSON
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Check top-level structure (support both old "trap_jobs" and new "genesis_vault_entries")
    vault_entries = None
    if "genesis_vault_entries" in data:
        vault_entries = data["genesis_vault_entries"]
    elif "trap_jobs" in data:
        # Backward compatibility with old format
        print("⚠️  Warning: Using deprecated 'trap_jobs' field. Use 'genesis_vault_entries' instead.")
        vault_entries = data["trap_jobs"]
    else:
        print("❌ Error: Missing 'genesis_vault_entries' or 'trap_jobs' field")
        return False
    
    if not isinstance(vault_entries, list):
        print("❌ Error: 'genesis_vault_entries' must be a list")
        return False
    
    if len(vault_entries) == 0:
        print("⚠️  Warning: No vault entries found")
        return True
    
    print(f"Found {len(vault_entries)} genesis vault entries")
    print("=" * 60)
    
    # Validate each vault entry
    all_valid = True
    total_errors = 0
    
    for i, entry in enumerate(vault_entries):
        is_valid, errors = validate_vault_entry(entry, i)
        
        if not is_valid:
            all_valid = False
            total_errors += len(errors)
            print(f"\n❌ Vault Entry {i+1}: entry_id={entry.get('entry_id', 'unknown')}")
            for error in errors:
                print(f"   - {error}")
    
    # Check for duplicate entry_id
    entry_ids = [ve.get("entry_id") for ve in vault_entries if "entry_id" in ve]
    duplicates = [eid for eid in set(entry_ids) if entry_ids.count(eid) > 1]
    
    if duplicates:
        all_valid = False
        total_errors += len(duplicates)
        print(f"\n❌ Duplicate entry_id found: {duplicates}")
    
    # Summary
    print("=" * 60)
    if all_valid and not duplicates:
        print(f"✅ All {len(vault_entries)} vault entries are valid!")
        return True
    else:
        print(f"❌ Validation failed: {total_errors} error(s) found")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate genesis vault entries JSON file")
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to genesis vault entries JSON file"
    )
    
    args = parser.parse_args()
    
    success = validate_genesis_traps(args.json_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

