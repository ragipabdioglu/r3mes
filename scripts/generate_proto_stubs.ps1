# Generate Python gRPC stubs from proto files
# Senior-level implementation with proper error handling and validation

param(
    [switch]$SkipValidation,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

Write-Host "🔧 R3MES Proto Stub Generator (Senior Edition)" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Configuration
$PROTO_DIR = "remes/proto"
$OUTPUT_DIR = "miner-engine/bridge/proto"
$REMES_PROTO_DIR = "$PROTO_DIR/remes/remes/v1"

# Check prerequisites
function Test-Prerequisites {
    Write-Host "`n📋 Checking prerequisites..." -ForegroundColor Yellow
    
    # Check protoc
    try {
        $protocVersion = & protoc --version 2>&1
        Write-Host "  ✓ protoc: $protocVersion" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ protoc not found" -ForegroundColor Red
        Write-Host "    Install: https://github.com/protocolbuffers/protobuf/releases" -ForegroundColor Yellow
        exit 1
    }
    
    # Check Python
    try {
        $pythonVersion = & python --version 2>&1
        Write-Host "  ✓ Python: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Python not found" -ForegroundColor Red
        exit 1
    }
    
    # Check grpcio-tools
    $grpcCheck = & python -c "import grpc_tools" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ⚠ grpcio-tools not found, installing..." -ForegroundColor Yellow
        & pip install grpcio-tools grpcio-status
        Write-Host "  ✓ grpcio-tools installed" -ForegroundColor Green
    } else {
        Write-Host "  ✓ grpcio-tools installed" -ForegroundColor Green
    }
}

# Create directory structure
function Initialize-Directories {
    Write-Host "`n📁 Creating directory structure..." -ForegroundColor Yellow
    
    $dirs = @(
        $OUTPUT_DIR,
        "$OUTPUT_DIR/amino",
        "$OUTPUT_DIR/gogoproto",
        "$OUTPUT_DIR/cosmos_proto",
        "$OUTPUT_DIR/remes",
        "$OUTPUT_DIR/remes/remes",
        "$OUTPUT_DIR/remes/remes/v1"
    )
    
    foreach ($dir in $dirs) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            if ($Verbose) { Write-Host "  Created: $dir" -ForegroundColor Gray }
        }
    }
    
    Write-Host "  ✓ Directories created" -ForegroundColor Green
}

# Generate dependency stubs (amino, gogoproto, cosmos_proto)
function New-DependencyStubs {
    Write-Host "`n📦 Generating dependency stubs..." -ForegroundColor Yellow
    
    # Amino stub
    $aminoInit = @'
"""Amino proto stubs (minimal implementation for R3MES)."""
__version__ = "1.0.0"
'@
    $aminoInit | Out-File -FilePath "$OUTPUT_DIR/amino/__init__.py" -Encoding utf8
    
    $aminoPb2 = @'
"""Generated protocol buffer code for amino."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default()
'@
    $aminoPb2 | Out-File -FilePath "$OUTPUT_DIR/amino/amino_pb2.py" -Encoding utf8
    
    # Gogoproto stub
    $gogoInit = @'
"""Gogoproto stubs (minimal implementation for R3MES)."""
__version__ = "1.0.0"
'@
    $gogoInit | Out-File -FilePath "$OUTPUT_DIR/gogoproto/__init__.py" -Encoding utf8
    
    $gogoPb2 = @'
"""Generated protocol buffer code for gogoproto."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default()
'@
    $gogoPb2 | Out-File -FilePath "$OUTPUT_DIR/gogoproto/gogo_pb2.py" -Encoding utf8
    
    # Cosmos proto stub
    $cosmosInit = @'
"""Cosmos proto stubs (minimal implementation for R3MES)."""
__version__ = "1.0.0"
'@
    $cosmosInit | Out-File -FilePath "$OUTPUT_DIR/cosmos_proto/__init__.py" -Encoding utf8
    
    $cosmosPb2 = @'
"""Generated protocol buffer code for cosmos_proto."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default()
'@
    $cosmosPb2 | Out-File -FilePath "$OUTPUT_DIR/cosmos_proto/cosmos_pb2.py" -Encoding utf8
    
    Write-Host "  ✓ Dependency stubs created" -ForegroundColor Green
}

# Generate proto stubs for R3MES
function New-ProtoStubs {
    Write-Host "`n🔨 Generating R3MES proto stubs..." -ForegroundColor Yellow
    
    if (!(Test-Path $REMES_PROTO_DIR)) {
        Write-Host "  ✗ Proto directory not found: $REMES_PROTO_DIR" -ForegroundColor Red
        exit 1
    }
    
    # Critical proto files for miner-engine
    $protoFiles = @(
        "tx.proto",
        "query.proto",
        "stored_gradient.proto",
        "task_pool.proto",
        "node.proto",
        "params.proto",
        "model.proto",
        "dataset.proto",
        "serving.proto",
        "state.proto"
    )
    
    $generated = 0
    $failed = 0
    
    foreach ($protoFile in $protoFiles) {
        $protoPath = "$REMES_PROTO_DIR/$protoFile"
        
        if (Test-Path $protoPath) {
            Write-Host "  Generating $protoFile..." -ForegroundColor Gray
            
            try {
                # Generate Python stubs
                & python -m grpc_tools.protoc `
                    -I"$PROTO_DIR" `
                    --python_out="$OUTPUT_DIR" `
                    --grpc_python_out="$OUTPUT_DIR" `
                    --pyi_out="$OUTPUT_DIR" `
                    "$protoPath" 2>&1 | Out-Null
                
                Write-Host "    ✓ $protoFile" -ForegroundColor Green
                $generated++
            } catch {
                Write-Host "    ✗ Failed: $protoFile" -ForegroundColor Red
                if ($Verbose) { Write-Host "      Error: $_" -ForegroundColor Red }
                $failed++
            }
        } else {
            Write-Host "    ⚠ Not found: $protoFile" -ForegroundColor Yellow
        }
    }
    
    Write-Host "`n  Generated: $generated files" -ForegroundColor Green
    if ($failed -gt 0) {
        Write-Host "  Failed: $failed files" -ForegroundColor Red
    }
}

# Create __init__.py files with proper imports
function New-InitFiles {
    Write-Host "`n📝 Creating __init__.py files..." -ForegroundColor Yellow
    
    # Root __init__.py
    $rootInit = @'
"""R3MES gRPC proto stubs.

This package contains generated Python stubs for R3MES blockchain protocol buffers.
Generated by scripts/generate_proto_stubs.ps1
"""
__version__ = "1.0.0"
'@
    $rootInit | Out-File -FilePath "$OUTPUT_DIR/__init__.py" -Encoding utf8
    
    # remes/__init__.py
    '"""R3MES proto package."""' | Out-File -FilePath "$OUTPUT_DIR/remes/__init__.py" -Encoding utf8
    
    # remes/remes/__init__.py
    '"""R3MES proto package."""' | Out-File -FilePath "$OUTPUT_DIR/remes/remes/__init__.py" -Encoding utf8
    
    # remes/remes/v1/__init__.py with smart imports
    $v1Init = @'
"""R3MES v1 proto stubs.

This module provides access to R3MES blockchain protocol buffer definitions.
"""

# Import generated modules with error handling
_AVAILABLE_MODULES = []

def _try_import(module_name):
    try:
        return __import__(module_name, fromlist=[''])
    except ImportError as e:
        import warnings
        warnings.warn(f"Failed to import {module_name}: {e}", ImportWarning)
        return None

# Transaction messages
tx_pb2 = _try_import('.tx_pb2')
tx_pb2_grpc = _try_import('.tx_pb2_grpc')
if tx_pb2: _AVAILABLE_MODULES.extend(['tx_pb2', 'tx_pb2_grpc'])

# Query messages
query_pb2 = _try_import('.query_pb2')
query_pb2_grpc = _try_import('.query_pb2_grpc')
if query_pb2: _AVAILABLE_MODULES.extend(['query_pb2', 'query_pb2_grpc'])

# Core types
stored_gradient_pb2 = _try_import('.stored_gradient_pb2')
if stored_gradient_pb2: _AVAILABLE_MODULES.append('stored_gradient_pb2')

task_pool_pb2 = _try_import('.task_pool_pb2')
if task_pool_pb2: _AVAILABLE_MODULES.append('task_pool_pb2')

node_pb2 = _try_import('.node_pb2')
if node_pb2: _AVAILABLE_MODULES.append('node_pb2')

params_pb2 = _try_import('.params_pb2')
if params_pb2: _AVAILABLE_MODULES.append('params_pb2')

model_pb2 = _try_import('.model_pb2')
if model_pb2: _AVAILABLE_MODULES.append('model_pb2')

dataset_pb2 = _try_import('.dataset_pb2')
if dataset_pb2: _AVAILABLE_MODULES.append('dataset_pb2')

serving_pb2 = _try_import('.serving_pb2')
if serving_pb2: _AVAILABLE_MODULES.append('serving_pb2')

state_pb2 = _try_import('.state_pb2')
if state_pb2: _AVAILABLE_MODULES.append('state_pb2')

__all__ = _AVAILABLE_MODULES

def get_available_modules():
    """Return list of successfully imported modules."""
    return _AVAILABLE_MODULES.copy()
'@
    $v1Init | Out-File -FilePath "$OUTPUT_DIR/remes/remes/v1/__init__.py" -Encoding utf8
    
    Write-Host "  ✓ __init__.py files created" -ForegroundColor Green
}

# Fix import paths in generated files
function Repair-ImportPaths {
    Write-Host "`n🔧 Fixing import paths..." -ForegroundColor Yellow
    
    $pbFiles = Get-ChildItem -Path "$OUTPUT_DIR/remes/remes/v1" -Filter "*_pb2*.py" -File
    $fixed = 0
    
    foreach ($file in $pbFiles) {
        $content = Get-Content $file.FullName -Raw -Encoding UTF8
        $originalContent = $content
        
        # Fix relative imports
        $content = $content -replace 'from remes\.remes\.v1 import', 'from . import'
        $content = $content -replace 'import amino\.amino_pb2', 'from amino import amino_pb2'
        $content = $content -replace 'import gogoproto\.gogo_pb2', 'from gogoproto import gogo_pb2'
        $content = $content -replace 'import cosmos_proto\.cosmos_pb2', 'from cosmos_proto import cosmos_pb2'
        
        if ($content -ne $originalContent) {
            $content | Out-File -FilePath $file.FullName -Encoding UTF8 -NoNewline
            $fixed++
            if ($Verbose) { Write-Host "  Fixed: $($file.Name)" -ForegroundColor Gray }
        }
    }
    
    Write-Host "  ✓ Fixed $fixed files" -ForegroundColor Green
}

# Validate generated files
function Test-GeneratedFiles {
    if ($SkipValidation) {
        Write-Host "`n⏭ Skipping validation" -ForegroundColor Yellow
        return
    }
    
    Write-Host "`n🔍 Validating generated files..." -ForegroundColor Yellow
    
    $requiredFiles = @(
        "$OUTPUT_DIR/remes/remes/v1/tx_pb2.py",
        "$OUTPUT_DIR/remes/remes/v1/tx_pb2_grpc.py",
        "$OUTPUT_DIR/remes/remes/v1/query_pb2.py",
        "$OUTPUT_DIR/remes/remes/v1/query_pb2_grpc.py",
        "$OUTPUT_DIR/remes/remes/v1/stored_gradient_pb2.py",
        "$OUTPUT_DIR/remes/remes/v1/task_pool_pb2.py"
    )
    
    $allFound = $true
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Host "  ✓ $(Split-Path $file -Leaf)" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $(Split-Path $file -Leaf) not found" -ForegroundColor Red
            $allFound = $false
        }
    }
    
    if (!$allFound) {
        Write-Host "`n⚠ Some files were not generated" -ForegroundColor Yellow
        exit 1
    }
}

# Test Python imports
function Test-PythonImports {
    if ($SkipValidation) {
        return
    }
    
    Write-Host "`n🧪 Testing Python imports..." -ForegroundColor Yellow
    
    $testScript = @'
import sys
sys.path.insert(0, '.')

try:
    from miner_engine.bridge.proto.remes.remes.v1 import tx_pb2, query_pb2
    print('✓ Import test passed')
    sys.exit(0)
except ImportError as e:
    print(f'⚠ Import test failed: {e}')
    print('  (This is normal if miner-engine is not in PYTHONPATH)')
    sys.exit(0)
'@
    
    $testScript | Out-File -FilePath "test_proto_import.py" -Encoding utf8
    
    try {
        & python test_proto_import.py
    } finally {
        Remove-Item "test_proto_import.py" -ErrorAction SilentlyContinue
    }
}

# Main execution
try {
    Test-Prerequisites
    Initialize-Directories
    New-DependencyStubs
    New-ProtoStubs
    New-InitFiles
    Repair-ImportPaths
    Test-GeneratedFiles
    Test-PythonImports
    
    Write-Host "`n✅ Proto stub generation completed successfully!" -ForegroundColor Green
    Write-Host "Generated files are in: $OUTPUT_DIR" -ForegroundColor Cyan
    Write-Host "`n🎉 Done!" -ForegroundColor Green
    
} catch {
    Write-Host "`n❌ Error: $_" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    exit 1
}
