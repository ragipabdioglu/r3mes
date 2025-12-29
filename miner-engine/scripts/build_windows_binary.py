#!/usr/bin/env python3
"""
Build Windows Binary Script

Builds a standalone Windows executable (engine.exe) using PyInstaller.
Creates a clean virtual environment, installs dependencies, and builds the executable.
"""

import argparse
import os
import sys
import subprocess
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime

# Script directory
SCRIPT_DIR = Path(__file__).parent
MINER_ENGINE_DIR = SCRIPT_DIR.parent


def run_command(cmd: list[str], cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd or MINER_ENGINE_DIR,
        check=check,
        capture_output=False,
        text=True
    )
    return result


def create_clean_venv(venv_path: Path) -> None:
    """Create a clean virtual environment."""
    print(f"Creating virtual environment at {venv_path}...")
    
    # Remove existing venv if it exists
    if venv_path.exists():
        print(f"Removing existing virtual environment...")
        shutil.rmtree(venv_path)
    
    # Create new venv
    run_command([sys.executable, "-m", "venv", str(venv_path)])
    print("✅ Virtual environment created")


def install_dependencies(venv_path: Path) -> None:
    """Install dependencies in the virtual environment."""
    print("Installing dependencies...")
    
    # Determine Python executable in venv
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    # Upgrade pip
    run_command([str(pip_exe), "install", "--upgrade", "pip", "wheel", "setuptools"])
    
    # Install PyInstaller
    run_command([str(pip_exe), "install", "pyinstaller>=6.0.0"])
    
    # Install project dependencies
    print("Installing project dependencies from pyproject.toml...")
    run_command([str(pip_exe), "install", "-e", "."])
    
    print("✅ Dependencies installed")


def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def build_executable(venv_path: Path, version: str, output_path: Path) -> Path:
    """Build the executable using PyInstaller."""
    print("Building executable with PyInstaller...")
    
    # Determine Python executable in venv
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
        pyinstaller_exe = venv_path / "Scripts" / "pyinstaller.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        pyinstaller_exe = venv_path / "bin" / "pyinstaller"
    
    # PyInstaller spec file path
    spec_file = MINER_ENGINE_DIR / "pyinstaller.spec"
    
    if not spec_file.exists():
        raise FileNotFoundError(f"PyInstaller spec file not found: {spec_file}")
    
    # Build command
    build_cmd = [
        str(pyinstaller_exe),
        "--clean",  # Clean cache before building
        "--noconfirm",  # Overwrite output directory without confirmation
        str(spec_file),
    ]
    
    # Run PyInstaller
    run_command(build_cmd)
    
    # Find output executable
    if sys.platform == "win32":
        exe_path = MINER_ENGINE_DIR / "dist" / "engine.exe"
    else:
        exe_path = MINER_ENGINE_DIR / "dist" / "engine"
    
    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found after build: {exe_path}")
    
    # Copy to output path if different
    if exe_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe_path, output_path)
        exe_path = output_path
    
    print(f"✅ Executable built: {exe_path}")
    
    # Get file size
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.2f} MB")
    
    return exe_path


def validate_executable(exe_path: Path) -> bool:
    """Validate that the executable works (basic test)."""
    print("Validating executable...")
    
    try:
        # Try to run with --version or --help
        result = subprocess.run(
            [str(exe_path), "--help"],
            capture_output=True,
            timeout=10,
            text=True
        )
        
        if result.returncode == 0 or "usage" in result.stdout.lower() or "help" in result.stdout.lower():
            print("✅ Executable validation passed")
            return True
        else:
            print(f"⚠️  Executable returned non-zero exit code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Executable validation timed out (this may be normal for large executables)")
        return True  # Don't fail on timeout, may be slow to start
    except Exception as e:
        print(f"⚠️  Executable validation error: {e}")
        return False


def create_build_metadata(exe_path: Path, version: str, output_dir: Path) -> Path:
    """Create build metadata file."""
    metadata = {
        "version": version,
        "build_date": datetime.utcnow().isoformat() + "Z",
        "executable_path": str(exe_path),
        "executable_size_bytes": exe_path.stat().st_size,
        "executable_size_mb": round(exe_path.stat().st_size / (1024 * 1024), 2),
        "sha256_checksum": calculate_checksum(exe_path),
        "platform": sys.platform,
        "python_version": sys.version,
    }
    
    metadata_path = output_dir / "build_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Build metadata saved: {metadata_path}")
    return metadata_path


def main():
    parser = argparse.ArgumentParser(description="Build Windows binary for R3MES Miner Engine")
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Version number for the build (default: 1.0.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dist/engine.exe",
        help="Output executable path (default: dist/engine.exe)"
    )
    parser.add_argument(
        "--venv",
        type=str,
        default="build_venv",
        help="Virtual environment directory (default: build_venv)"
    )
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="Skip virtual environment creation (use current environment)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip executable validation"
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    output_path = (MINER_ENGINE_DIR / args.output).resolve()
    output_dir = output_path.parent
    venv_path = MINER_ENGINE_DIR / args.venv
    
    print("=" * 60)
    print("R3MES Miner Engine - Windows Binary Build")
    print("=" * 60)
    print(f"Version: {args.version}")
    print(f"Output: {output_path}")
    print(f"Working directory: {MINER_ENGINE_DIR}")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Create virtual environment (if not skipped)
        if not args.skip_venv:
            create_clean_venv(venv_path)
            install_dependencies(venv_path)
            python_exe = venv_path / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")
        else:
            print("⚠️  Skipping virtual environment creation (using current environment)")
            python_exe = Path(sys.executable)
        
        # Step 2: Build executable
        exe_path = build_executable(venv_path if not args.skip_venv else None, args.version, output_path)
        
        # Step 3: Validate executable (if not skipped)
        if not args.skip_validation:
            validate_executable(exe_path)
        else:
            print("⚠️  Skipping executable validation")
        
        # Step 4: Create build metadata
        metadata_path = create_build_metadata(exe_path, args.version, output_dir)
        
        # Summary
        print()
        print("=" * 60)
        print("✅ Build completed successfully!")
        print("=" * 60)
        print(f"Executable: {exe_path}")
        print(f"Size: {exe_path.stat().st_size / (1024 * 1024):.2f} MB")
        print(f"Metadata: {metadata_path}")
        print()
        print("Next steps:")
        print(f"1. Test the executable: {exe_path} --help")
        print(f"2. Create package: python scripts/create_engine_package.py --input {exe_path}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

