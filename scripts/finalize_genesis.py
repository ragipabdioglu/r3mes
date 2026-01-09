#!/usr/bin/env python3
"""
Finalize Genesis JSON

Integrates generated trap jobs, model config, and network parameters into final genesis.json.
This script:
1. Runs generate_genesis_traps.py to create trap job data
2. Integrates vault entries into genesis template
3. Sets model config (IPFS hash, version)
4. Sets network parameters
5. Outputs final genesis.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List


def run_script(script_path: Path, args: List[str] = None) -> bool:
    """Run a Python script and return success status."""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running script: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return False


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json_file(file_path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Save JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def integrate_genesis_vault_entries(
    genesis: Dict[str, Any],
    vault_entries_path: Path
) -> None:
    """Integrate genesis vault entries from JSON file."""
    if not vault_entries_path.exists():
        print(f"⚠️  Warning: Vault entries file not found: {vault_entries_path}")
        print("   Genesis vault will be empty")
        return
    
    # Load vault entries
    vault_entries = load_json_file(vault_entries_path)
    
    if not isinstance(vault_entries, list):
        print(f"❌ Error: Vault entries file must contain a JSON array")
        sys.exit(1)
    
    print(f"Found {len(vault_entries)} genesis vault entries")
    
    # Integrate into genesis
    if "app_state" not in genesis:
        genesis["app_state"] = {}
    
    if "remes" not in genesis["app_state"]:
        genesis["app_state"]["remes"] = {}
    
    # Remove placeholder entry if exists
    genesis_vault_list = genesis["app_state"]["remes"].get("genesis_vault_list", [])
    genesis_vault_list = [entry for entry in genesis_vault_list if not entry.get("_comment")]
    
    # Add actual entries
    genesis["app_state"]["remes"]["genesis_vault_list"] = vault_entries
    
    print(f"✅ Integrated {len(vault_entries)} vault entries into genesis")


def set_model_config(
    genesis: Dict[str, Any],
    model_hash: str = None,
    model_version: str = None
) -> None:
    """Set model configuration in genesis."""
    if "app_state" not in genesis:
        genesis["app_state"] = {}
    
    if "remes" not in genesis["app_state"]:
        genesis["app_state"]["remes"] = {}
    
    remes_state = genesis["app_state"]["remes"]
    
    if model_hash:
        remes_state["model_hash"] = model_hash
        print(f"✅ Set model_hash: {model_hash}")
    
    if model_version:
        remes_state["model_version"] = model_version
        print(f"✅ Set model_version: {model_version}")


def set_network_parameters(
    genesis: Dict[str, Any],
    chain_id: str = None,
    genesis_time: str = None
) -> None:
    """Set network parameters in genesis."""
    if chain_id:
        genesis["chain_id"] = chain_id
        print(f"✅ Set chain_id: {chain_id}")
    
    if genesis_time:
        genesis["genesis_time"] = genesis_time
        print(f"✅ Set genesis_time: {genesis_time}")


def clean_template_comments(genesis: Dict[str, Any]) -> None:
    """Remove template comment fields from genesis."""
    def clean_dict(obj: Any) -> Any:
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                if not key.startswith("_comment"):
                    cleaned[key] = clean_dict(value)
            return cleaned
        elif isinstance(obj, list):
            return [clean_dict(item) for item in obj]
        else:
            return obj
    
    # Clean top-level
    for key in list(genesis.keys()):
        if key.startswith("_comment"):
            del genesis[key]
    
    # Clean nested
    genesis = clean_dict(genesis)
    
    # Clean app_state.remes
    if "app_state" in genesis and "remes" in genesis["app_state"]:
        remes_state = genesis["app_state"]["remes"]
        for key in list(remes_state.keys()):
            if key.startswith("_comment"):
                del remes_state[key]
    
    print("✅ Cleaned template comments")


def main():
    parser = argparse.ArgumentParser(description="Finalize genesis.json with trap jobs and config")
    parser.add_argument(
        "--template",
        type=str,
        default="remes/config/genesis.template.json",
        help="Path to genesis template file (default: remes/config/genesis.template.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="remes/config/genesis.json",
        help="Output genesis.json path (default: remes/config/genesis.json)"
    )
    parser.add_argument(
        "--vault-entries",
        type=str,
        default="genesis_vault_entries.json",
        help="Path to generated vault entries JSON file (default: genesis_vault_entries.json)"
    )
    parser.add_argument(
        "--generate-traps",
        action="store_true",
        help="Generate trap jobs before finalizing (runs generate_genesis_traps.py)"
    )
    parser.add_argument(
        "--num-traps",
        type=int,
        default=50,
        help="Number of trap jobs to generate (default: 50)"
    )
    parser.add_argument(
        "--model-hash",
        type=str,
        help="IPFS hash of initial model (optional, will be empty if not provided)"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.0.0",
        help="Model version (default: v1.0.0)"
    )
    parser.add_argument(
        "--chain-id",
        type=str,
        default="remes-mainnet-1",
        help="Chain ID (default: remes-mainnet-1)"
    )
    parser.add_argument(
        "--genesis-time",
        type=str,
        help="Genesis time in RFC3339 format (default: current time)"
    )
    parser.add_argument(
        "--keep-comments",
        action="store_true",
        help="Keep template comments in output (default: removed)"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    template_path = project_root / args.template
    output_path = project_root / args.output
    vault_entries_path = project_root / args.vault_entries
    
    print("=" * 60)
    print("R3MES Genesis Finalization")
    print("=" * 60)
    print(f"Template: {template_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    print()
    
    # Check template exists
    if not template_path.exists():
        print(f"❌ Error: Template file not found: {template_path}")
        sys.exit(1)
    
    # Step 1: Generate trap jobs (if requested)
    if args.generate_traps:
        print("Step 1: Generating trap jobs...")
        generate_script = script_dir / "generate_genesis_traps.py"
        
        if not generate_script.exists():
            print(f"❌ Error: Generate script not found: {generate_script}")
            sys.exit(1)
        
        generate_args = [
            "--count", str(args.num_traps),
            "--output", str(vault_entries_path)
        ]
        
        if not run_script(generate_script, generate_args):
            print("❌ Failed to generate trap jobs")
            sys.exit(1)
        
        print()
    else:
        print("Step 1: Skipping trap job generation (use --generate-traps to enable)")
        print()
    
    # Step 2: Load template
    print("Step 2: Loading genesis template...")
    genesis = load_json_file(template_path)
    print("✅ Template loaded")
    print()
    
    # Step 3: Integrate vault entries
    print("Step 3: Integrating genesis vault entries...")
    integrate_genesis_vault_entries(genesis, vault_entries_path)
    print()
    
    # Step 4: Set model config
    print("Step 4: Setting model configuration...")
    set_model_config(genesis, args.model_hash, args.model_version)
    print()
    
    # Step 5: Set network parameters
    print("Step 5: Setting network parameters...")
    if not args.genesis_time:
        from datetime import datetime, timezone
        args.genesis_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    set_network_parameters(genesis, args.chain_id, args.genesis_time)
    print()
    
    # Step 6: Clean comments (unless keep-comments is set)
    if not args.keep_comments:
        print("Step 6: Cleaning template comments...")
        clean_template_comments(genesis)
        print()
    
    # Step 7: Save final genesis
    print("Step 7: Saving final genesis.json...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json_file(output_path, genesis)
    print(f"✅ Saved to: {output_path}")
    print()
    
    # Summary
    print("=" * 60)
    print("✅ Genesis finalization completed!")
    print("=" * 60)
    print(f"Output: {output_path}")
    
    # Count vault entries
    vault_count = len(genesis.get("app_state", {}).get("remes", {}).get("genesis_vault_list", []))
    print(f"Vault entries: {vault_count}")
    print(f"Chain ID: {genesis.get('chain_id', 'N/A')}")
    print(f"Genesis time: {genesis.get('genesis_time', 'N/A')}")
    print()
    print("Next steps:")
    print(f"1. Validate: python scripts/validate_genesis.py {output_path}")
    print("2. Use with: remesd init --chain-id remes-mainnet-1 --genesis genesis.json")
    print("=" * 60)


if __name__ == "__main__":
    main()

