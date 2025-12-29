# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller Spec File for R3MES Miner Engine

Builds a standalone Windows executable (engine.exe) with all dependencies bundled.
Note: The resulting executable will be large (~2.5-3GB) due to PyTorch and CUDA libraries.
"""

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    # Main script entry point
    ['r3mes/cli/commands.py'],
    
    pathex=[
        str(miner_engine_dir),
        str(miner_engine_dir / 'r3mes'),
        str(miner_engine_dir / 'core'),
        str(miner_engine_dir / 'utils'),
        str(miner_engine_dir / 'bridge'),
    ],
    
    binaries=[],
    
    datas=[
        # Add any data files if needed (config templates, etc.)
        # ('path/to/data', 'data'),
    ],
    
    hiddenimports=[
        # Core modules
        'r3mes',
        'r3mes.cli',
        'r3mes.cli.commands',
        'r3mes.cli.wizard',
        'r3mes.cli.wallet',
        'r3mes.cli.config',
        'r3mes.miner',
        'r3mes.miner.engine',
        'r3mes.miner.model_loader',
        'r3mes.miner.llama_loader',
        'r3mes.miner.vram_profiler',
        'r3mes.utils',
        'r3mes.utils.logger',
        'r3mes.utils.ipfs_client',
        'r3mes.utils.faucet',
        'r3mes.utils.version_checker',
        'r3mes.utils.time_sync',
        'r3mes.utils.firewall_check',
        'r3mes.utils.environment_validator',
        'r3mes.utils.gpu_detection',
        'r3mes.utils.shard_assignment',
        'core',
        'core.bitlinear',
        'core.trainer',
        'core.serialization',
        'core.binary_serialization',
        'core.deterministic',
        'core.coordinator',
        'core.gradient_accumulator',
        'core.gradient_compression',
        'bridge',
        'bridge.blockchain_client',
        'bridge.arrow_flight_client',
        # gRPC and Protocol Buffers
        'grpc',
        'grpc._cython',
        'grpc.experimental',
        'google.protobuf',
        'google.protobuf.descriptor',
        'google.protobuf.pyext',
        'google.protobuf.internal',
        # IPFS client
        'ipfshttpclient',
        'ipfshttpclient.client',
        'ipfshttpclient.filescanner',
        # PyTorch hidden imports
        'torch',
        'torch._C',
        'torch._C._fft',
        'torch._C._nn',
        'torch._C._fft_backend',
        'torch._C._linalg',
        'torch._C._sparse',
        'torch._C._special',
        'torch._C._distributed_c10d',
        'torch.distributed',
        'torch.utils',
        'torch.utils.data',
        'torch.utils.data._utils',
        'torch.utils.cpp_extension',
        'torch.backends',
        'torch.backends.cudnn',
        'torch.backends.cuda',
        'torch.nn',
        'torch.nn.modules',
        'torch.nn.functional',
        'torch.nn.parallel',
        'torch.optim',
        'torch.autograd',
        'torch.cuda',
        # NumPy
        'numpy',
        'numpy.core',
        'numpy.core._multiarray_umath',
        'numpy.random',
        'numpy.random._pickle',
        # Crypto libraries
        'Crypto',
        'Crypto.Cipher',
        'Crypto.Hash',
        'Crypto.PublicKey',
        'ecdsa',
        # Other dependencies
        'requests',
        'psutil',
        'websocket',
        'websocket_client',
        'pyarrow',
        'pyarrow.lib',
        'pyarrow._fs',
    ],
    
    hookspath=[],
    
    hooksconfig={},
    
    runtime_hooks=[],
    
    excludes=[
        # Exclude test modules to reduce size
        'pytest',
        'pytest.*',
        'tests',
        'test',
        'testing',
        'unittest',
        'unittest.*',
        # Exclude documentation
        'doc',
        'docs',
        '*.md',
        # Exclude development tools
        'setuptools',
        'pip',
        'wheel',
        'distutils',
        # Exclude Jupyter/IPython (not needed)
        'IPython',
        'jupyter',
        'notebook',
        # Exclude matplotlib (not used in miner)
        'matplotlib',
        'PIL',
        'Pillow',
    ],
    
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Collect PyTorch binaries (CUDA libraries excluded by default to reduce size)
# CUDA libraries will be loaded from system PATH at runtime
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='engine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Enable UPX compression (if available)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console for logging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Can add icon file path here if available
)

# Note: CUDA libraries are NOT bundled to reduce executable size (~500MB vs ~2.5GB)
# The system must have CUDA installed and in PATH for GPU support.
# If CUDA libraries need to be bundled, add them to `a.binaries` manually.

