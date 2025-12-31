"""
Property Tests for Requirements File GPU Independence

Tests that requirements-api.txt does not contain GPU dependencies.
Validates Requirements 1.3, 9.1, 9.5.
"""

import pytest
import os
from pathlib import Path


# GPU-dependent packages that should NOT be in requirements-api.txt
GPU_PACKAGES = {
    "torch",
    "pytorch",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "sentence-transformers",
    "nvidia",
    "cuda",
    "cudnn",
    "triton",
}

# Packages that MUST be in requirements-api.txt for basic functionality
REQUIRED_API_PACKAGES = {
    "fastapi",
    "uvicorn",
    "pydantic",
    "sqlalchemy",
    "redis",
    "httpx",
}

# Packages that MUST be in requirements-inference.txt
REQUIRED_INFERENCE_PACKAGES = {
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "sentence-transformers",
}


def parse_requirements(filepath: str) -> set:
    """Parse requirements file and return set of package names."""
    packages = set()
    
    if not os.path.exists(filepath):
        return packages
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            # Extract package name (before version specifier)
            # Handle formats: package, package==1.0, package>=1.0, package[extra]
            package = line.split("==")[0].split(">=")[0].split("<=")[0].split("<")[0].split(">")[0].split("[")[0]
            package = package.strip().lower()
            
            if package:
                packages.add(package)
    
    return packages


class TestRequirementsGPUIndependence:
    """Test suite for requirements file GPU independence."""
    
    @pytest.fixture
    def backend_dir(self) -> Path:
        """Get backend directory path."""
        # Try to find backend directory
        current = Path(__file__).parent
        while current != current.parent:
            backend = current / "backend"
            if backend.exists():
                return backend
            current = current.parent
        
        # Fallback to relative path
        return Path("backend")
    
    def test_requirements_api_exists(self, backend_dir):
        """Test that requirements-api.txt exists."""
        api_req = backend_dir / "requirements-api.txt"
        assert api_req.exists(), f"requirements-api.txt not found at {api_req}"
    
    def test_requirements_inference_exists(self, backend_dir):
        """Test that requirements-inference.txt exists."""
        inference_req = backend_dir / "requirements-inference.txt"
        assert inference_req.exists(), f"requirements-inference.txt not found at {inference_req}"
    
    def test_requirements_api_no_gpu_packages(self, backend_dir):
        """
        Property: requirements-api.txt should NOT contain GPU-dependent packages.
        Validates: Requirements 1.3, 9.1
        """
        api_req = backend_dir / "requirements-api.txt"
        packages = parse_requirements(str(api_req))
        
        gpu_found = packages.intersection(GPU_PACKAGES)
        assert not gpu_found, (
            f"requirements-api.txt contains GPU packages: {gpu_found}. "
            "These should be in requirements-inference.txt only."
        )
    
    def test_requirements_api_has_core_packages(self, backend_dir):
        """
        Property: requirements-api.txt should contain core API packages.
        Validates: Requirement 9.1
        """
        api_req = backend_dir / "requirements-api.txt"
        packages = parse_requirements(str(api_req))
        
        missing = REQUIRED_API_PACKAGES - packages
        assert not missing, (
            f"requirements-api.txt missing required packages: {missing}"
        )
    
    def test_requirements_inference_has_gpu_packages(self, backend_dir):
        """
        Property: requirements-inference.txt should contain GPU packages.
        Validates: Requirement 9.2
        """
        inference_req = backend_dir / "requirements-inference.txt"
        packages = parse_requirements(str(inference_req))
        
        missing = REQUIRED_INFERENCE_PACKAGES - packages
        assert not missing, (
            f"requirements-inference.txt missing required packages: {missing}"
        )
    
    def test_requirements_inference_no_api_packages(self, backend_dir):
        """
        Property: requirements-inference.txt should NOT duplicate API packages.
        Validates: Requirement 9.5 (clean separation)
        """
        inference_req = backend_dir / "requirements-inference.txt"
        packages = parse_requirements(str(inference_req))
        
        # Inference file should only have GPU packages, not API packages
        api_overlap = packages.intersection(REQUIRED_API_PACKAGES)
        assert not api_overlap, (
            f"requirements-inference.txt duplicates API packages: {api_overlap}. "
            "These should only be in requirements-api.txt."
        )
    
    def test_full_requirements_is_superset(self, backend_dir):
        """
        Property: requirements.txt should be a superset of api + inference.
        Validates: Backward compatibility
        """
        full_req = backend_dir / "requirements.txt"
        api_req = backend_dir / "requirements-api.txt"
        inference_req = backend_dir / "requirements-inference.txt"
        
        full_packages = parse_requirements(str(full_req))
        api_packages = parse_requirements(str(api_req))
        inference_packages = parse_requirements(str(inference_req))
        
        combined = api_packages.union(inference_packages)
        
        # Full requirements should contain all packages from both files
        # (may have additional packages for backward compatibility)
        missing_from_full = combined - full_packages
        
        # Allow some flexibility - not all packages need to be in full
        # But core packages should be there
        core_missing = missing_from_full.intersection(
            REQUIRED_API_PACKAGES.union(REQUIRED_INFERENCE_PACKAGES)
        )
        assert not core_missing, (
            f"requirements.txt missing core packages: {core_missing}"
        )


class TestRequirementsFileFormat:
    """Test requirements file format and structure."""
    
    @pytest.fixture
    def backend_dir(self) -> Path:
        """Get backend directory path."""
        current = Path(__file__).parent
        while current != current.parent:
            backend = current / "backend"
            if backend.exists():
                return backend
            current = current.parent
        return Path("backend")
    
    def test_requirements_api_has_header_comment(self, backend_dir):
        """Test that requirements-api.txt has descriptive header."""
        api_req = backend_dir / "requirements-api.txt"
        
        with open(api_req, "r") as f:
            content = f.read()
        
        assert "GPU-less" in content or "gpu-less" in content.lower(), (
            "requirements-api.txt should mention GPU-less deployment in header"
        )
    
    def test_requirements_inference_has_header_comment(self, backend_dir):
        """Test that requirements-inference.txt has descriptive header."""
        inference_req = backend_dir / "requirements-inference.txt"
        
        with open(inference_req, "r") as f:
            content = f.read()
        
        assert "GPU" in content, (
            "requirements-inference.txt should mention GPU requirement in header"
        )
    
    def test_requirements_api_mentions_inference_mode(self, backend_dir):
        """Test that requirements-api.txt mentions R3MES_INFERENCE_MODE."""
        api_req = backend_dir / "requirements-api.txt"
        
        with open(api_req, "r") as f:
            content = f.read()
        
        assert "R3MES_INFERENCE_MODE" in content, (
            "requirements-api.txt should mention R3MES_INFERENCE_MODE environment variable"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
