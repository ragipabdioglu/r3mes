"""
Adapter Sync Service - Blockchain'den Onaylı Adapter'ları Senkronize Eder

KRİTİK EKSİKLİK #2 ÇÖZÜMÜ: Backend Blockchain Adapter Sync

Bu servis:
1. Blockchain'den onaylı adapter listesini periyodik olarak çeker
2. Yeni adapter'ları IPFS'den indirir
3. Checksum doğrulaması yapar
4. Hot-reload ile adapter'ları yükler
"""

import os
import asyncio
import hashlib
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import aiohttp
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class ApprovedAdapter:
    """Blockchain'den gelen onaylı adapter bilgisi."""
    adapter_id: str
    name: str
    adapter_type: str  # "dora", "lora"
    version: str
    ipfs_hash: str
    checksum: str
    size_bytes: int
    domain: str
    description: str
    approved_at: int
    approval_tx_hash: str


@dataclass
class SyncResult:
    """Senkronizasyon sonucu."""
    success: bool
    new_adapters: List[str]
    updated_adapters: List[str]
    failed_adapters: List[str]
    error_message: Optional[str] = None


class AdapterSyncService:
    """
    Blockchain'den onaylı adapter'ları senkronize eden servis.
    
    Kullanım:
        sync_service = AdapterSyncService(
            blockchain_rest_url="http://localhost:1317",
            ipfs_gateway="http://localhost:8080",
            adapters_dir="checkpoints/adapters",
        )
        
        # Tek seferlik sync
        result = await sync_service.sync_adapters()
        
        # Periyodik sync başlat
        await sync_service.start_periodic_sync(interval_seconds=300)
    """
    
    def __init__(
        self,
        blockchain_rest_url: Optional[str] = None,
        ipfs_gateway: Optional[str] = None,
        adapters_dir: Optional[str] = None,
        model_manager: Optional[Any] = None,
    ):
        """
        Initialize adapter sync service.
        
        Args:
            blockchain_rest_url: Blockchain REST API URL
            ipfs_gateway: IPFS gateway URL
            adapters_dir: Local directory for adapter storage
            model_manager: AIModelManager instance for hot-reload
        """
        self.blockchain_rest_url = blockchain_rest_url or os.getenv(
            "BLOCKCHAIN_REST_URL", "http://localhost:1317"
        )
        self.ipfs_gateway = ipfs_gateway or os.getenv(
            "IPFS_GATEWAY_URL", "http://localhost:8080"
        )
        self.adapters_dir = Path(adapters_dir or os.getenv(
            "ADAPTERS_DIR", "checkpoints/adapters"
        ))
        self.model_manager = model_manager
        
        # Create adapters directory if not exists
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        
        # Track synced adapters
        self._synced_adapters: Dict[str, ApprovedAdapter] = {}
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(
            f"AdapterSyncService initialized: "
            f"blockchain={self.blockchain_rest_url}, "
            f"ipfs={self.ipfs_gateway}, "
            f"dir={self.adapters_dir}"
        )
    
    async def query_approved_adapters(self) -> List[ApprovedAdapter]:
        """
        Blockchain'den onaylı adapter listesini çeker.
        
        Returns:
            List of approved adapters from blockchain
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Query approved adapters endpoint
                url = f"{self.blockchain_rest_url}/remes/remes/v1/approved_adapters"
                
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"Failed to query adapters: {response.status}")
                        return []
                    
                    data = await response.json()
                    adapters = []
                    
                    for adapter_data in data.get("adapters", []):
                        adapter = ApprovedAdapter(
                            adapter_id=adapter_data.get("adapter_id", ""),
                            name=adapter_data.get("name", ""),
                            adapter_type=adapter_data.get("adapter_type", "dora"),
                            version=adapter_data.get("version", "1.0.0"),
                            ipfs_hash=adapter_data.get("ipfs_hash", ""),
                            checksum=adapter_data.get("checksum", ""),
                            size_bytes=adapter_data.get("size_bytes", 0),
                            domain=adapter_data.get("domain", "general"),
                            description=adapter_data.get("description", ""),
                            approved_at=adapter_data.get("approved_at", 0),
                            approval_tx_hash=adapter_data.get("approval_tx_hash", ""),
                        )
                        adapters.append(adapter)
                    
                    logger.info(f"Queried {len(adapters)} approved adapters from blockchain")
                    return adapters
                    
        except asyncio.TimeoutError:
            logger.error("Timeout querying approved adapters")
            return []
        except Exception as e:
            logger.error(f"Error querying approved adapters: {e}")
            return []
    
    async def download_adapter_from_ipfs(
        self,
        adapter: ApprovedAdapter,
    ) -> Optional[Path]:
        """
        IPFS'den adapter indirir ve checksum doğrular.
        
        Args:
            adapter: Adapter info from blockchain
            
        Returns:
            Path to downloaded adapter or None if failed
        """
        adapter_path = self.adapters_dir / adapter.adapter_id
        adapter_path.mkdir(parents=True, exist_ok=True)
        
        try:
            async with aiohttp.ClientSession() as session:
                # Download from IPFS gateway
                url = f"{self.ipfs_gateway}/ipfs/{adapter.ipfs_hash}"
                
                logger.info(f"Downloading adapter {adapter.adapter_id} from IPFS: {adapter.ipfs_hash}")
                
                async with session.get(url, timeout=300) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download adapter: {response.status}")
                        return None
                    
                    # Save to file
                    adapter_file = adapter_path / "adapter_model.bin"
                    content = await response.read()
                    
                    async with aiofiles.open(adapter_file, "wb") as f:
                        await f.write(content)
                    
                    # Verify checksum
                    actual_checksum = hashlib.sha256(content).hexdigest()
                    if actual_checksum != adapter.checksum:
                        logger.error(
                            f"Checksum mismatch for {adapter.adapter_id}: "
                            f"expected={adapter.checksum[:16]}..., "
                            f"actual={actual_checksum[:16]}..."
                        )
                        # Remove corrupted file
                        adapter_file.unlink()
                        return None
                    
                    # Save metadata
                    metadata_file = adapter_path / "adapter_config.json"
                    import json
                    metadata = {
                        "adapter_id": adapter.adapter_id,
                        "name": adapter.name,
                        "adapter_type": adapter.adapter_type,
                        "version": adapter.version,
                        "ipfs_hash": adapter.ipfs_hash,
                        "checksum": adapter.checksum,
                        "domain": adapter.domain,
                        "description": adapter.description,
                        "approved_at": adapter.approved_at,
                        "downloaded_at": datetime.utcnow().isoformat(),
                    }
                    async with aiofiles.open(metadata_file, "w") as f:
                        await f.write(json.dumps(metadata, indent=2))
                    
                    logger.info(f"Successfully downloaded adapter {adapter.adapter_id}")
                    return adapter_path
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout downloading adapter {adapter.adapter_id}")
            return None
        except Exception as e:
            logger.error(f"Error downloading adapter {adapter.adapter_id}: {e}")
            return None

    async def hot_reload_adapter(self, adapter_id: str, adapter_path: Path) -> bool:
        """
        Adapter'ı model manager'a hot-reload yapar.
        
        Args:
            adapter_id: Adapter identifier
            adapter_path: Path to adapter directory
            
        Returns:
            True if successful
        """
        if self.model_manager is None:
            logger.warning("Model manager not available, skipping hot-reload")
            return False
        
        try:
            # Load adapter into model manager
            success = self.model_manager.load_adapter(adapter_id, str(adapter_path))
            
            if success:
                logger.info(f"Hot-reloaded adapter: {adapter_id}")
            else:
                logger.error(f"Failed to hot-reload adapter: {adapter_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error hot-reloading adapter {adapter_id}: {e}")
            return False
    
    async def sync_adapters(self) -> SyncResult:
        """
        Blockchain'den adapter'ları senkronize eder.
        
        Returns:
            SyncResult with details of sync operation
        """
        result = SyncResult(
            success=True,
            new_adapters=[],
            updated_adapters=[],
            failed_adapters=[],
        )
        
        try:
            # 1. Query approved adapters from blockchain
            approved_adapters = await self.query_approved_adapters()
            
            if not approved_adapters:
                logger.info("No approved adapters found on blockchain")
                return result
            
            # 2. Check each adapter
            for adapter in approved_adapters:
                adapter_path = self.adapters_dir / adapter.adapter_id
                
                # Check if already synced with same version
                if adapter.adapter_id in self._synced_adapters:
                    synced = self._synced_adapters[adapter.adapter_id]
                    if synced.version == adapter.version and synced.checksum == adapter.checksum:
                        continue  # Already up to date
                    else:
                        # Version changed, need to update
                        is_update = True
                else:
                    is_update = False
                
                # Check if already downloaded locally
                if adapter_path.exists():
                    metadata_file = adapter_path / "adapter_config.json"
                    if metadata_file.exists():
                        import json
                        with open(metadata_file) as f:
                            local_metadata = json.load(f)
                        
                        if local_metadata.get("checksum") == adapter.checksum:
                            # Already downloaded, just track it
                            self._synced_adapters[adapter.adapter_id] = adapter
                            continue
                
                # 3. Download from IPFS
                downloaded_path = await self.download_adapter_from_ipfs(adapter)
                
                if downloaded_path is None:
                    result.failed_adapters.append(adapter.adapter_id)
                    continue
                
                # 4. Hot-reload if model manager available
                if self.model_manager:
                    await self.hot_reload_adapter(adapter.adapter_id, downloaded_path)
                
                # 5. Track synced adapter
                self._synced_adapters[adapter.adapter_id] = adapter
                
                if is_update:
                    result.updated_adapters.append(adapter.adapter_id)
                else:
                    result.new_adapters.append(adapter.adapter_id)
            
            logger.info(
                f"Adapter sync complete: "
                f"new={len(result.new_adapters)}, "
                f"updated={len(result.updated_adapters)}, "
                f"failed={len(result.failed_adapters)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during adapter sync: {e}")
            result.success = False
            result.error_message = str(e)
            return result
    
    async def sync_single_adapter(
        self,
        adapter_id: str,
        ipfs_hash: str,
        checksum: str,
    ) -> bool:
        """
        Tek bir adapter'ı senkronize eder (event-driven için).
        
        Args:
            adapter_id: Adapter identifier
            ipfs_hash: IPFS hash of adapter
            checksum: Expected checksum
            
        Returns:
            True if successful
        """
        try:
            # Create temporary adapter object
            adapter = ApprovedAdapter(
                adapter_id=adapter_id,
                name=adapter_id,
                adapter_type="dora",
                version="latest",
                ipfs_hash=ipfs_hash,
                checksum=checksum,
                size_bytes=0,
                domain="general",
                description="Event-driven sync",
                approved_at=int(datetime.utcnow().timestamp()),
                approval_tx_hash="",
            )
            
            # Download from IPFS
            downloaded_path = await self.download_adapter_from_ipfs(adapter)
            
            if downloaded_path is None:
                logger.error(f"Failed to download adapter {adapter_id}")
                return False
            
            # Hot-reload if model manager available
            if self.model_manager:
                success = await self.hot_reload_adapter(adapter_id, downloaded_path)
                if not success:
                    logger.error(f"Failed to hot-reload adapter {adapter_id}")
                    return False
            
            # Track synced adapter
            self._synced_adapters[adapter_id] = adapter
            
            logger.info(f"Successfully synced single adapter: {adapter_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing single adapter {adapter_id}: {e}")
            return False
    
    async def start_periodic_sync(self, interval_seconds: int = 300):
        """
        Periyodik senkronizasyon başlatır.
        
        Args:
            interval_seconds: Sync interval in seconds (default: 5 minutes)
        """
        if self._running:
            logger.warning("Periodic sync already running")
            return
        
        self._running = True
        
        async def sync_loop():
            while self._running:
                try:
                    await self.sync_adapters()
                except Exception as e:
                    logger.error(f"Error in periodic sync: {e}")
                
                await asyncio.sleep(interval_seconds)
        
        self._sync_task = asyncio.create_task(sync_loop())
        logger.info(f"Started periodic adapter sync (interval: {interval_seconds}s)")
    
    async def stop_periodic_sync(self):
        """Periyodik senkronizasyonu durdurur."""
        self._running = False
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
        
        logger.info("Stopped periodic adapter sync")
    
    def get_synced_adapters(self) -> Dict[str, ApprovedAdapter]:
        """Returns currently synced adapters."""
        return self._synced_adapters.copy()
    
    def get_adapter_path(self, adapter_id: str) -> Optional[Path]:
        """Returns local path for an adapter."""
        adapter_path = self.adapters_dir / adapter_id
        if adapter_path.exists():
            return adapter_path
        return None


# Singleton instance
_adapter_sync_service: Optional[AdapterSyncService] = None


def get_adapter_sync_service() -> AdapterSyncService:
    """Get or create the adapter sync service singleton."""
    global _adapter_sync_service
    
    if _adapter_sync_service is None:
        _adapter_sync_service = AdapterSyncService()
    
    return _adapter_sync_service


async def init_adapter_sync_service(
    model_manager: Optional[Any] = None,
    start_periodic: bool = True,
    sync_interval: int = 300,
) -> AdapterSyncService:
    """
    Initialize and start the adapter sync service.
    
    Args:
        model_manager: AIModelManager instance
        start_periodic: Whether to start periodic sync
        sync_interval: Sync interval in seconds
        
    Returns:
        Initialized AdapterSyncService
    """
    global _adapter_sync_service
    
    _adapter_sync_service = AdapterSyncService(model_manager=model_manager)
    
    # Initial sync
    await _adapter_sync_service.sync_adapters()
    
    # Start periodic sync
    if start_periodic:
        await _adapter_sync_service.start_periodic_sync(sync_interval)
    
    return _adapter_sync_service
