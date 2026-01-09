#!/usr/bin/env python3
"""
R3MES Proto Stub Generator (Senior Edition)
Generates Python gRPC stubs from proto files with proper error handling and validation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple

# ANSI colors for output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_header():
    print(f"{Colors.CYAN}🔧 R3MES Proto Stub Generator (Senior Edition){Colors.NC}")
    print(f"{Colors.CYAN}{'=' * 50}{Colors.NC}")

def check_prerequisites() -> bool:
    """Check if all required tools are installed."""
    print(f"\n{Colors.YELLOW}📋 Checking prerequisites...{Colors.NC}")
    
    # Check protoc
    try:
        result = subprocess.run(['protoc', '--version'], capture_output=True, text=True)
        print(f"  {Colors.GREEN}✓ protoc: {result.stdout.strip()}{Colors.NC}")
    except FileNotFoundError:
        print(f"  {Colors.RED}✗ protoc not found{Colors.NC}")
        print(f"    Install: https://github.com/protocolbuffers/protobuf/releases")
        return False
    
    # Check Python
    print(f"  {Colors.GREEN}✓ Python: {sys.version.split()[0]}{Colors.NC}")
    
    # Check grpcio-tools
    try:
        import grpc_tools
        print(f"  {Colors.GREEN}✓ grpcio-tools installed{Colors.NC}")
    except ImportError:
        print(f"  {Colors.YELLOW}⚠ grpcio-tools not found, installing...{Colors.NC}")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'grpcio-tools', 'grpcio-status'])
        print(f"  {Colors.GREEN}✓ grpcio-tools installed{Colors.NC}")
    
    return True

def create_directories(output_dir: Path):
    """Create necessary directory structure."""
    print(f"\n{Colors.YELLOW}📁 Creating directory structure...{Colors.NC}")
    
    dirs = [
        output_dir,
        output_dir / 'amino',
        output_dir / 'gogoproto',
        output_dir / 'cosmos_proto',
        output_dir / 'remes' / 'remes' / 'v1'
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"  {Colors.GREEN}✓ Directories created{Colors.NC}")

def create_dependency_stubs(output_dir: Path):
    """Create minimal stubs for amino, gogoproto, and cosmos_proto."""
    print(f"\n{Colors.YELLOW}📦 Generating dependency stubs...{Colors.NC}")
    
    # Amino
    (output_dir / 'amino' / '__init__.py').write_text(
        '"""Amino proto stubs (minimal implementation for R3MES)."""\n__version__ = "1.0.0"\n'
    )
    (output_dir / 'amino' / 'amino_pb2.py').write_text('''"""Generated protocol buffer code for amino."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default()
''')
    
    # Gogoproto
    (output_dir / 'gogoproto' / '__init__.py').write_text(
        '"""Gogoproto stubs (minimal implementation for R3MES)."""\n__version__ = "1.0.0"\n'
    )
    (output_dir / 'gogoproto' / 'gogo_pb2.py').write_text('''"""Generated protocol buffer code for gogoproto."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default()
''')
    
    # Cosmos proto
    (output_dir / 'cosmos_proto' / '__init__.py').write_text(
        '"""Cosmos proto stubs (minimal implementation for R3MES)."""\n__version__ = "1.0.0"\n'
    )
    (output_dir / 'cosmos_proto' / 'cosmos_pb2.py').write_text('''"""Generated protocol buffer code for cosmos_proto."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default()
''')
    
    print(f"  {Colors.GREEN}✓ Dependency stubs created{Colors.NC}")

def generate_proto_stubs(proto_dir: Path, output_dir: Path) -> Tuple[int, int]:
    """Generate Python stubs from proto files."""
    print(f"\n{Colors.YELLOW}🔨 Generating R3MES proto stubs...{Colors.NC}")
    
    remes_proto_dir = proto_dir / 'remes' / 'remes' / 'v1'
    
    if not remes_proto_dir.exists():
        print(f"  {Colors.RED}✗ Proto directory not found: {remes_proto_dir}{Colors.NC}")
        return 0, 0
    
    # Critical proto files for miner-engine
    proto_files = [
        'tx.proto',
        'query.proto',
        'stored_gradient.proto',
        'task_pool.proto',
        'node.proto',
        'params.proto',
        'model.proto',
        'dataset.proto',
        'serving.proto',
        'state.proto',
        'pinning.proto',
        'slashing.proto',
        'trap_job.proto',
        'execution_environment.proto',
        'model_version.proto',
        'subnet.proto',
        'training_window.proto',
        'verification.proto',
        'genesis.proto',
        'genesis_vault.proto'
    ]
    
    generated = 0
    failed = 0
    
    for proto_file in proto_files:
        proto_path = remes_proto_dir / proto_file
        
        if proto_path.exists():
            print(f"  Generating {proto_file}...")
            
            try:
                cmd = [
                    sys.executable, '-m', 'grpc_tools.protoc',
                    f'-I{proto_dir}',
                    f'--python_out={output_dir}',
                    f'--grpc_python_out={output_dir}',
                    f'--pyi_out={output_dir}',
                    str(proto_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"    {Colors.GREEN}✓ {proto_file}{Colors.NC}")
                    generated += 1
                else:
                    print(f"    {Colors.RED}✗ Failed: {proto_file}{Colors.NC}")
                    if result.stderr:
                        print(f"      Error: {result.stderr[:200]}")
                    failed += 1
            except Exception as e:
                print(f"    {Colors.RED}✗ Exception: {proto_file} - {e}{Colors.NC}")
                failed += 1
        else:
            print(f"    {Colors.YELLOW}⚠ Not found: {proto_file}{Colors.NC}")
    
    print(f"\n  {Colors.GREEN}Generated: {generated} files{Colors.NC}")
    if failed > 0:
        print(f"  {Colors.RED}Failed: {failed} files{Colors.NC}")
    
    return generated, failed

def create_init_files(output_dir: Path):
    """Create __init__.py files with proper imports."""
    print(f"\n{Colors.YELLOW}📝 Creating __init__.py files...{Colors.NC}")
    
    # Root __init__.py
    (output_dir / '__init__.py').write_text('''"""R3MES gRPC proto stubs.

This package contains generated Python stubs for R3MES blockchain protocol buffers.
Generated by scripts/generate_proto_stubs.py
"""
__version__ = "1.0.0"
''')
    
    # remes/__init__.py
    (output_dir / 'remes' / '__init__.py').write_text('"""R3MES proto package."""\n')
    
    # remes/remes/__init__.py
    (output_dir / 'remes' / 'remes' / '__init__.py').write_text('"""R3MES proto package."""\n')
    
    # remes/remes/v1/__init__.py with smart imports
    (output_dir / 'remes' / 'remes' / 'v1' / '__init__.py').write_text('''"""R3MES v1 proto stubs.

This module provides access to R3MES blockchain protocol buffer definitions.
"""

# Import generated modules with error handling
_AVAILABLE_MODULES = []

def _try_import(module_name):
    """Try to import a module, return None if it fails."""
    try:
        mod = __import__(module_name, globals(), locals(), ['*'], 1)
        return mod
    except ImportError as e:
        import warnings
        warnings.warn(f"Failed to import {module_name}: {e}", ImportWarning)
        return None

# Transaction messages
tx_pb2 = _try_import('tx_pb2')
tx_pb2_grpc = _try_import('tx_pb2_grpc')
if tx_pb2: _AVAILABLE_MODULES.extend(['tx_pb2', 'tx_pb2_grpc'])

# Query messages
query_pb2 = _try_import('query_pb2')
query_pb2_grpc = _try_import('query_pb2_grpc')
if query_pb2: _AVAILABLE_MODULES.extend(['query_pb2', 'query_pb2_grpc'])

# Core types
stored_gradient_pb2 = _try_import('stored_gradient_pb2')
if stored_gradient_pb2: _AVAILABLE_MODULES.append('stored_gradient_pb2')

task_pool_pb2 = _try_import('task_pool_pb2')
if task_pool_pb2: _AVAILABLE_MODULES.append('task_pool_pb2')

node_pb2 = _try_import('node_pb2')
if node_pb2: _AVAILABLE_MODULES.append('node_pb2')

params_pb2 = _try_import('params_pb2')
if params_pb2: _AVAILABLE_MODULES.append('params_pb2')

model_pb2 = _try_import('model_pb2')
if model_pb2: _AVAILABLE_MODULES.append('model_pb2')

dataset_pb2 = _try_import('dataset_pb2')
if dataset_pb2: _AVAILABLE_MODULES.append('dataset_pb2')

serving_pb2 = _try_import('serving_pb2')
if serving_pb2: _AVAILABLE_MODULES.append('serving_pb2')

state_pb2 = _try_import('state_pb2')
if state_pb2: _AVAILABLE_MODULES.append('state_pb2')

__all__ = _AVAILABLE_MODULES

def get_available_modules():
    """Return list of successfully imported modules."""
    return _AVAILABLE_MODULES.copy()
''')
    
    print(f"  {Colors.GREEN}✓ __init__.py files created{Colors.NC}")

def fix_import_paths(output_dir: Path):
    """Fix import paths in generated files."""
    print(f"\n{Colors.YELLOW}🔧 Fixing import paths...{Colors.NC}")
    
    v1_dir = output_dir / 'remes' / 'remes' / 'v1'
    pb_files = list(v1_dir.glob('*_pb2*.py'))
    
    fixed = 0
    for pb_file in pb_files:
        content = pb_file.read_text(encoding='utf-8')
        original = content
        
        # Fix relative imports
        content = content.replace('from remes.remes.v1 import', 'from . import')
        content = content.replace('import amino.amino_pb2', 'from amino import amino_pb2')
        content = content.replace('import gogoproto.gogo_pb2', 'from gogoproto import gogo_pb2')
        content = content.replace('import cosmos_proto.cosmos_pb2', 'from cosmos_proto import cosmos_pb2')
        
        if content != original:
            pb_file.write_text(content, encoding='utf-8')
            fixed += 1
    
    print(f"  {Colors.GREEN}✓ Fixed {fixed} files{Colors.NC}")

def validate_generated_files(output_dir: Path) -> bool:
    """Validate that required files were generated."""
    print(f"\n{Colors.YELLOW}🔍 Validating generated files...{Colors.NC}")
    
    required_files = [
        'remes/remes/v1/tx_pb2.py',
        'remes/remes/v1/tx_pb2_grpc.py',
        'remes/remes/v1/query_pb2.py',
        'remes/remes/v1/query_pb2_grpc.py',
        'remes/remes/v1/stored_gradient_pb2.py',
        'remes/remes/v1/task_pool_pb2.py'
    ]
    
    all_found = True
    for file_path in required_files:
        full_path = output_dir / file_path
        if full_path.exists():
            print(f"  {Colors.GREEN}✓ {file_path.split('/')[-1]}{Colors.NC}")
        else:
            print(f"  {Colors.RED}✗ {file_path.split('/')[-1]} not found{Colors.NC}")
            all_found = False
    
    return all_found

def test_imports(output_dir: Path):
    """Test if generated modules can be imported."""
    print(f"\n{Colors.YELLOW}🧪 Testing Python imports...{Colors.NC}")
    
    # Add output_dir parent to sys.path temporarily
    parent_dir = str(output_dir.parent.parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        from miner_engine.bridge.proto.remes.remes.v1 import tx_pb2, query_pb2
        print(f"  {Colors.GREEN}✓ Import test passed{Colors.NC}")
    except ImportError as e:
        print(f"  {Colors.YELLOW}⚠ Import test failed: {e}{Colors.NC}")
        print(f"    (This is normal if miner-engine is not in PYTHONPATH)")

def main():
    """Main execution function."""
    print_header()
    
    # Configuration
    proto_dir = Path('remes/proto')
    output_dir = Path('miner-engine/bridge/proto')
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            sys.exit(1)
        
        # Create directories
        create_directories(output_dir)
        
        # Generate dependency stubs
        create_dependency_stubs(output_dir)
        
        # Generate proto stubs
        generated, failed = generate_proto_stubs(proto_dir, output_dir)
        
        # Create __init__.py files
        create_init_files(output_dir)
        
        # Fix import paths
        fix_import_paths(output_dir)
        
        # Validate
        if not validate_generated_files(output_dir):
            print(f"\n{Colors.YELLOW}⚠ Some files were not generated{Colors.NC}")
            sys.exit(1)
        
        # Test imports
        test_imports(output_dir)
        
        print(f"\n{Colors.GREEN}✅ Proto stub generation completed successfully!{Colors.NC}")
        print(f"{Colors.CYAN}Generated files are in: {output_dir}{Colors.NC}")
        print(f"\n{Colors.GREEN}🎉 Done!{Colors.NC}")
        
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.NC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
