"""
Build script for creating standalone binary with PyInstaller

This script creates a single executable file from the backend application.
"""

import os
import sys
import shutil
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def build_binary():
    """Build standalone binary using PyInstaller."""
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        logger.warning("PyInstaller is not installed. Installing...")
        os.system(f"{sys.executable} -m pip install pyinstaller")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    backend_dir = project_root / "backend"
    app_dir = backend_dir / "app"
    
    # Create spec file for PyInstaller
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{app_dir}/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('{app_dir}/semantic_router.py', 'app'),
        ('{app_dir}/router.py', 'app'),
        ('{app_dir}/model_manager.py', 'app'),
        ('{app_dir}/database_async.py', 'app'),
        ('{app_dir}/config_manager.py', 'app'),
        ('{app_dir}/config_endpoints.py', 'app'),
        ('{app_dir}/setup_logging.py', 'app'),
        ('{app_dir}/task_queue.py', 'app'),
        ('{app_dir}/inference_executor.py', 'app'),
    ],
    hiddenimports=[
        'fastapi',
        'uvicorn',
        'pydantic',
        'slowapi',
        'aiosqlite',
        'sentence_transformers',
        'torch',
        'transformers',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='r3mes_backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    
    spec_file = backend_dir / "r3mes_backend.spec"
    with open(spec_file, 'w') as f:
        f.write(spec_content)
    
    logger.info("Building binary with PyInstaller...")
    logger.info(f"Spec file: {spec_file}")
    
    # Run PyInstaller
    os.chdir(backend_dir)
    cmd = f"{sys.executable} -m PyInstaller r3mes_backend.spec --clean --noconfirm"
    result = os.system(cmd)
    
    if result == 0:
        logger.info("Binary built successfully!")
        logger.info(f"Output: {backend_dir / 'dist' / 'r3mes_backend'}")
        if sys.platform == "win32":
            logger.info(f"Executable: {backend_dir / 'dist' / 'r3mes_backend.exe'}")
        else:
            logger.info(f"Executable: {backend_dir / 'dist' / 'r3mes_backend'}")
    else:
        logger.error("Build failed!")
        sys.exit(1)

if __name__ == "__main__":
    build_binary()

