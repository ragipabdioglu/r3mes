#!/usr/bin/env python3
"""
R3MES Download Manager

Centralized download management with:
1. Multiple source fallback (HuggingFace, IPFS, HTTP)
2. Retry with exponential backoff
3. Progress tracking
4. Atomic downloads with rollback
5. Bandwidth limiting (optional)

Architecture:
    DownloadManager
        │
        ├── HuggingFaceDownloader (primary for models)
        ├── IPFSDownloader (primary for datasets/adapters)
        └── HTTPDownloader (fallback)
"""

import logging
import time
import io
import tarfile
import gzip
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from utils.verification import (
    AtomicDownload,
    RetryConfig,
    with_retry,
    calculate_checksum,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class DownloadSource(Enum):
    """Download source types."""
    HUGGINGFACE = "huggingface"
    IPFS = "ipfs"
    HTTP = "http"
    LOCAL = "local"


class DownloadStatus(Enum):
    """Download status."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class DownloadProgress:
    """Download progress information."""
    status: DownloadStatus = DownloadStatus.PENDING
    source: Optional[DownloadSource] = None
    bytes_downloaded: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0
    eta_seconds: float = 0.0
    message: str = ""
    
    @property
    def progress_percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.bytes_downloaded / self.total_bytes) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "source": self.source.value if self.source else None,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "progress_percent": self.progress_percent,
            "speed_bps": self.speed_bps,
            "eta_seconds": self.eta_seconds,
            "message": self.message,
        }


@dataclass
class DownloadResult:
    """Result of a download operation."""
    success: bool
    local_path: Optional[str] = None
    source_used: Optional[DownloadSource] = None
    checksum: Optional[str] = None
    size_bytes: int = 0
    download_time_seconds: float = 0.0
    error_message: Optional[str] = None
    attempts: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "local_path": self.local_path,
            "source_used": self.source_used.value if self.source_used else None,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "download_time_seconds": self.download_time_seconds,
            "error_message": self.error_message,
            "attempts": self.attempts,
        }


@dataclass
class DownloadSpec:
    """Specification for a download."""
    # Identifiers
    name: str
    version: str
    
    # Sources (in priority order)
    ipfs_hash: Optional[str] = None
    huggingface_repo: Optional[str] = None
    huggingface_revision: Optional[str] = None
    http_urls: List[str] = field(default_factory=list)
    
    # Verification
    expected_checksum: Optional[str] = None
    expected_size_bytes: int = 0
    
    # Options
    extract_archive: bool = True
    verify_after_download: bool = True


class BaseDownloader(ABC):
    """Base class for downloaders."""
    
    @abstractmethod
    def download(
        self,
        spec: DownloadSpec,
        dest_path: Path,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> DownloadResult:
        """Download content to destination."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this downloader is available."""
        pass


class HuggingFaceDownloader(BaseDownloader):
    """Download from HuggingFace Hub."""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self._hf_available: Optional[bool] = None
    
    def is_available(self) -> bool:
        if self._hf_available is None:
            try:
                from huggingface_hub import snapshot_download
                self._hf_available = True
            except ImportError:
                self._hf_available = False
                logger.warning("huggingface_hub not installed")
        return self._hf_available
    
    def download(
        self,
        spec: DownloadSpec,
        dest_path: Path,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> DownloadResult:
        if not self.is_available():
            return DownloadResult(
                success=False,
                error_message="HuggingFace Hub not available",
            )
        
        if not spec.huggingface_repo:
            return DownloadResult(
                success=False,
                error_message="No HuggingFace repo specified",
            )
        
        start_time = time.time()
        
        try:
            from huggingface_hub import snapshot_download
            
            if progress_callback:
                progress_callback(DownloadProgress(
                    status=DownloadStatus.DOWNLOADING,
                    source=DownloadSource.HUGGINGFACE,
                    message=f"Downloading from HuggingFace: {spec.huggingface_repo}",
                ))
            
            logger.info(f"Downloading from HuggingFace: {spec.huggingface_repo}")
            
            snapshot_download(
                repo_id=spec.huggingface_repo,
                revision=spec.huggingface_revision,
                local_dir=str(dest_path),
                local_dir_use_symlinks=False,
            )
            
            download_time = time.time() - start_time
            
            # Calculate checksum
            checksum = None
            if spec.verify_after_download:
                checksum = calculate_checksum(dest_path)
            
            # Get size
            size_bytes = sum(f.stat().st_size for f in dest_path.rglob('*') if f.is_file())
            
            return DownloadResult(
                success=True,
                local_path=str(dest_path),
                source_used=DownloadSource.HUGGINGFACE,
                checksum=checksum,
                size_bytes=size_bytes,
                download_time_seconds=download_time,
            )
            
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            return DownloadResult(
                success=False,
                source_used=DownloadSource.HUGGINGFACE,
                error_message=str(e),
                download_time_seconds=time.time() - start_time,
            )


class IPFSDownloader(BaseDownloader):
    """Download from IPFS."""
    
    def __init__(
        self,
        ipfs_client,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.ipfs_client = ipfs_client
        self.retry_config = retry_config or RetryConfig()
    
    def is_available(self) -> bool:
        return self.ipfs_client is not None and self.ipfs_client.is_connected()
    
    def download(
        self,
        spec: DownloadSpec,
        dest_path: Path,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> DownloadResult:
        if not self.is_available():
            return DownloadResult(
                success=False,
                error_message="IPFS client not available",
            )
        
        if not spec.ipfs_hash:
            return DownloadResult(
                success=False,
                error_message="No IPFS hash specified",
            )
        
        start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback(DownloadProgress(
                    status=DownloadStatus.DOWNLOADING,
                    source=DownloadSource.IPFS,
                    message=f"Downloading from IPFS: {spec.ipfs_hash}",
                ))
            
            logger.info(f"Downloading from IPFS: {spec.ipfs_hash}")
            
            content = self._download_with_retry(spec.ipfs_hash)
            
            if content is None:
                return DownloadResult(
                    success=False,
                    source_used=DownloadSource.IPFS,
                    error_message=f"Failed to retrieve content from IPFS: {spec.ipfs_hash}",
                    download_time_seconds=time.time() - start_time,
                )
            
            # Extract if archive
            if progress_callback:
                progress_callback(DownloadProgress(
                    status=DownloadStatus.EXTRACTING,
                    source=DownloadSource.IPFS,
                    bytes_downloaded=len(content),
                    total_bytes=len(content),
                    message="Extracting content...",
                ))
            
            dest_path.mkdir(parents=True, exist_ok=True)
            
            if spec.extract_archive:
                self._extract_content(content, dest_path)
            else:
                # Save as single file
                output_file = dest_path / f"{spec.name}.bin"
                with open(output_file, 'wb') as f:
                    f.write(content)
            
            download_time = time.time() - start_time
            
            # Calculate checksum
            checksum = None
            if spec.verify_after_download:
                checksum = calculate_checksum(dest_path)
            
            # Get size
            if dest_path.is_file():
                size_bytes = dest_path.stat().st_size
            else:
                size_bytes = sum(f.stat().st_size for f in dest_path.rglob('*') if f.is_file())
            
            return DownloadResult(
                success=True,
                local_path=str(dest_path),
                source_used=DownloadSource.IPFS,
                checksum=checksum,
                size_bytes=size_bytes,
                download_time_seconds=download_time,
            )
            
        except Exception as e:
            logger.error(f"IPFS download failed: {e}")
            return DownloadResult(
                success=False,
                source_used=DownloadSource.IPFS,
                error_message=str(e),
                download_time_seconds=time.time() - start_time,
            )
    
    def _download_with_retry(self, ipfs_hash: str) -> Optional[bytes]:
        """Download with retry logic."""
        last_error = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                content = self.ipfs_client.retrieve_content(ipfs_hash)
                if content is not None:
                    return content
            except Exception as e:
                last_error = e
                
                if attempt < self.retry_config.max_attempts - 1:
                    delay = min(
                        self.retry_config.initial_delay * (self.retry_config.exponential_base ** attempt),
                        self.retry_config.max_delay
                    )
                    logger.warning(f"IPFS download failed (attempt {attempt + 1}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
        
        logger.error(f"IPFS download failed after {self.retry_config.max_attempts} attempts: {last_error}")
        return None
    
    def _extract_content(self, content: bytes, dest_path: Path) -> None:
        """Extract archive content."""
        # Detect archive type
        if content[:2] == b'\x1f\x8b':  # gzip
            with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as tar:
                tar.extractall(dest_path)
        elif content[:4] == b'PK\x03\x04':  # zip
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                zf.extractall(dest_path)
        elif content[:6] == b'7z\xbc\xaf\x27\x1c':  # 7z (not supported, save as-is)
            logger.warning("7z archives not supported, saving as raw file")
            with open(dest_path / "data.7z", 'wb') as f:
                f.write(content)
        else:
            # Not an archive, save as single file
            # Try to detect file type
            if content[:1] == b'{' or content[:1] == b'[':
                # JSON
                with open(dest_path / "data.json", 'wb') as f:
                    f.write(content)
            else:
                # Binary or JSONL
                with open(dest_path / "data.bin", 'wb') as f:
                    f.write(content)


class HTTPDownloader(BaseDownloader):
    """Download from HTTP/HTTPS URLs."""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        timeout: int = 300,
    ):
        self.retry_config = retry_config or RetryConfig()
        self.timeout = timeout
    
    def is_available(self) -> bool:
        try:
            import requests
            return True
        except ImportError:
            return False
    
    def download(
        self,
        spec: DownloadSpec,
        dest_path: Path,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> DownloadResult:
        if not self.is_available():
            return DownloadResult(
                success=False,
                error_message="requests library not available",
            )
        
        if not spec.http_urls:
            return DownloadResult(
                success=False,
                error_message="No HTTP URLs specified",
            )
        
        import requests
        
        start_time = time.time()
        last_error = None
        
        for url in spec.http_urls:
            try:
                if progress_callback:
                    progress_callback(DownloadProgress(
                        status=DownloadStatus.DOWNLOADING,
                        source=DownloadSource.HTTP,
                        message=f"Downloading from: {url}",
                    ))
                
                logger.info(f"Downloading from HTTP: {url}")
                
                response = requests.get(url, stream=True, timeout=self.timeout)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                dest_path.mkdir(parents=True, exist_ok=True)
                temp_file = dest_path / f"{spec.name}.download"
                
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if progress_callback and total_size > 0:
                                progress_callback(DownloadProgress(
                                    status=DownloadStatus.DOWNLOADING,
                                    source=DownloadSource.HTTP,
                                    bytes_downloaded=downloaded,
                                    total_bytes=total_size,
                                    message=f"Downloading: {downloaded / (1024*1024):.1f} MB",
                                ))
                
                # Extract if needed
                if spec.extract_archive:
                    with open(temp_file, 'rb') as f:
                        content = f.read()
                    temp_file.unlink()
                    
                    ipfs_downloader = IPFSDownloader(None)
                    ipfs_downloader._extract_content(content, dest_path)
                else:
                    # Rename to final name
                    final_file = dest_path / f"{spec.name}.bin"
                    temp_file.rename(final_file)
                
                download_time = time.time() - start_time
                
                # Calculate checksum
                checksum = None
                if spec.verify_after_download:
                    checksum = calculate_checksum(dest_path)
                
                return DownloadResult(
                    success=True,
                    local_path=str(dest_path),
                    source_used=DownloadSource.HTTP,
                    checksum=checksum,
                    size_bytes=downloaded,
                    download_time_seconds=download_time,
                )
                
            except Exception as e:
                last_error = e
                logger.warning(f"HTTP download from {url} failed: {e}")
                continue
        
        return DownloadResult(
            success=False,
            source_used=DownloadSource.HTTP,
            error_message=f"All HTTP downloads failed. Last error: {last_error}",
            download_time_seconds=time.time() - start_time,
        )


class DownloadManager:
    """
    Unified download manager with fallback support.
    
    Tries sources in order: HuggingFace → IPFS → HTTP
    """
    
    def __init__(
        self,
        ipfs_client=None,
        retry_config: Optional[RetryConfig] = None,
        prefer_huggingface: bool = True,
    ):
        self.retry_config = retry_config or RetryConfig()
        self.prefer_huggingface = prefer_huggingface
        
        # Initialize downloaders
        self.huggingface = HuggingFaceDownloader(retry_config)
        self.ipfs = IPFSDownloader(ipfs_client, retry_config)
        self.http = HTTPDownloader(retry_config)
    
    def download(
        self,
        spec: DownloadSpec,
        dest_path: Path,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        verify_checksum: bool = True,
    ) -> DownloadResult:
        """
        Download content with automatic fallback.
        
        Args:
            spec: Download specification
            dest_path: Destination path
            progress_callback: Optional progress callback
            verify_checksum: Whether to verify checksum after download
            
        Returns:
            DownloadResult with download status
        """
        # Build source priority list
        sources: List[Tuple[BaseDownloader, str]] = []
        
        if self.prefer_huggingface and spec.huggingface_repo:
            sources.append((self.huggingface, "HuggingFace"))
        
        if spec.ipfs_hash:
            sources.append((self.ipfs, "IPFS"))
        
        if not self.prefer_huggingface and spec.huggingface_repo:
            sources.append((self.huggingface, "HuggingFace"))
        
        if spec.http_urls:
            sources.append((self.http, "HTTP"))
        
        if not sources:
            return DownloadResult(
                success=False,
                error_message="No download sources available",
            )
        
        # Try each source
        last_result = None
        total_attempts = 0
        
        for downloader, source_name in sources:
            if not downloader.is_available():
                logger.info(f"{source_name} downloader not available, skipping")
                continue
            
            logger.info(f"Attempting download from {source_name}...")
            total_attempts += 1
            
            result = downloader.download(spec, dest_path, progress_callback)
            
            if result.success:
                # Verify checksum if required
                if verify_checksum and spec.expected_checksum:
                    if progress_callback:
                        progress_callback(DownloadProgress(
                            status=DownloadStatus.VERIFYING,
                            source=result.source_used,
                            message="Verifying checksum...",
                        ))
                    
                    if result.checksum != spec.expected_checksum:
                        logger.error(
                            f"Checksum mismatch from {source_name}: "
                            f"expected {spec.expected_checksum[:16]}..., "
                            f"got {result.checksum[:16] if result.checksum else 'None'}..."
                        )
                        # Clean up failed download
                        self._cleanup(dest_path)
                        last_result = DownloadResult(
                            success=False,
                            source_used=result.source_used,
                            checksum=result.checksum,
                            error_message="Checksum verification failed",
                            attempts=total_attempts,
                        )
                        continue
                
                if progress_callback:
                    progress_callback(DownloadProgress(
                        status=DownloadStatus.COMPLETE,
                        source=result.source_used,
                        bytes_downloaded=result.size_bytes,
                        total_bytes=result.size_bytes,
                        message="Download complete",
                    ))
                
                result.attempts = total_attempts
                return result
            
            last_result = result
            logger.warning(f"Download from {source_name} failed: {result.error_message}")
        
        # All sources failed
        if progress_callback:
            progress_callback(DownloadProgress(
                status=DownloadStatus.FAILED,
                message=f"All download sources failed. Last error: {last_result.error_message if last_result else 'Unknown'}",
            ))
        
        return DownloadResult(
            success=False,
            error_message=f"All download sources failed after {total_attempts} attempts",
            attempts=total_attempts,
        )
    
    def _cleanup(self, path: Path) -> None:
        """Clean up failed download."""
        import shutil
        try:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        except Exception as e:
            logger.warning(f"Failed to cleanup {path}: {e}")
