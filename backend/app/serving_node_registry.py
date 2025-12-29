"""
Serving Node Registry

Manages LoRA registry and serving node registration/status tracking.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

from .database_config import DatabaseConfig

logger = logging.getLogger(__name__)


class ServingNodeRegistry:
    """Registry for LoRA adapters and serving nodes."""
    
    def __init__(self, database):
        """
        Initialize serving node registry.
        
        Args:
            database: AsyncDatabase instance
        """
        self.database = database
        self.config = DatabaseConfig()
    
    def _json_dumps(self, data: Any) -> str:
        """Convert data to JSON string for database storage."""
        return json.dumps(data)
    
    def _json_loads(self, data: Any) -> Any:
        """Parse JSON string from database."""
        if isinstance(data, str):
            return json.loads(data)
        return data
    
    async def register_lora(
        self,
        name: str,
        ipfs_hash: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a new LoRA adapter.
        
        Args:
            name: LoRA adapter name (unique)
            ipfs_hash: IPFS hash of the LoRA file
            description: Optional description
            category: Optional category (e.g., "coding", "legal")
            version: Optional version string
            
        Returns:
            Dictionary with registration result
        """
        try:
            if self.config.is_postgresql():
                # PostgreSQL
                if not self.database._db or not self.database._db.pool:
                    await self.database.connect()
                
                async with self.database._db.pool.acquire() as conn:
                    # Check if LoRA exists
                    existing = await conn.fetchrow(
                        "SELECT id FROM lora_registry WHERE name = $1",
                        name
                    )
                    
                    now = datetime.utcnow()
                    if existing:
                        # Update existing
                        await conn.execute("""
                            UPDATE lora_registry 
                            SET ipfs_hash = $1, description = $2, category = $3, 
                                version = $4, updated_at = $5
                            WHERE id = $6
                        """, ipfs_hash, description, category, version, now, existing['id'])
                        logger.info(f"Updated LoRA registry entry: {name}")
                        return {"success": True, "action": "updated", "lora_id": existing['id']}
                    else:
                        # Insert new
                        lora_id = await conn.fetchval("""
                            INSERT INTO lora_registry (name, ipfs_hash, description, category, version, created_at, updated_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            RETURNING id
                        """, name, ipfs_hash, description, category, version, now, now)
                        logger.info(f"Registered new LoRA: {name} (IPFS: {ipfs_hash})")
                        return {"success": True, "action": "created", "lora_id": lora_id}
            else:
                # SQLite - use aiosqlite connection
                if not self.database._connection:
                    await self.database.connect()
                
                cursor = await self.database._connection.cursor()
                
                # Check if exists
                await cursor.execute("SELECT id FROM lora_registry WHERE name = ?", (name,))
                existing = await cursor.fetchone()
                
                now = datetime.utcnow()
                if existing:
                    # Update
                    await cursor.execute("""
                        UPDATE lora_registry 
                        SET ipfs_hash = ?, description = ?, category = ?, 
                            version = ?, updated_at = ?
                        WHERE id = ?
                    """, (ipfs_hash, description, category, version, now, existing[0]))
                    await self.database._connection.commit()
                    logger.info(f"Updated LoRA registry entry: {name}")
                    return {"success": True, "action": "updated", "lora_id": existing[0]}
                else:
                    # Insert
                    await cursor.execute("""
                        INSERT INTO lora_registry (name, ipfs_hash, description, category, version, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (name, ipfs_hash, description, category, version, now, now))
                    await self.database._connection.commit()
                    lora_id = cursor.lastrowid
                    logger.info(f"Registered new LoRA: {name} (IPFS: {ipfs_hash})")
                    return {"success": True, "action": "created", "lora_id": lora_id}
        except Exception as e:
            logger.error(f"Failed to register LoRA {name}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def get_lora_list(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get list of available LoRA adapters.
        
        Args:
            active_only: Only return active LoRAs
            
        Returns:
            List of LoRA dictionaries
        """
        try:
            if self.config.is_postgresql():
                if not self.database._db or not self.database._db.pool:
                    await self.database.connect()
                
                async with self.database._db.pool.acquire() as conn:
                    if active_only:
                        rows = await conn.fetch(
                            "SELECT id, name, ipfs_hash, description, category, version, created_at FROM lora_registry WHERE is_active = TRUE"
                        )
                    else:
                        rows = await conn.fetch(
                            "SELECT id, name, ipfs_hash, description, category, version, created_at FROM lora_registry"
                        )
                    
                    return [
                        {
                            "id": row['id'],
                            "name": row['name'],
                            "ipfs_hash": row['ipfs_hash'],
                            "description": row['description'],
                            "category": row['category'],
                            "version": row['version'],
                            "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                        }
                        for row in rows
                    ]
            else:
                # SQLite
                if not self.database._connection:
                    await self.database.connect()
                
                cursor = await self.database._connection.cursor()
                if active_only:
                    await cursor.execute(
                        "SELECT id, name, ipfs_hash, description, category, version, created_at FROM lora_registry WHERE is_active = 1"
                    )
                else:
                    await cursor.execute(
                        "SELECT id, name, ipfs_hash, description, category, version, created_at FROM lora_registry"
                    )
                
                rows = await cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "ipfs_hash": row[2],
                        "description": row[3],
                        "category": row[4],
                        "version": row[5],
                        "created_at": row[6],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get LoRA list: {e}", exc_info=True)
            return []
    
    async def register_serving_node(
        self,
        wallet_address: str,
        endpoint_url: str,
        available_lora_list: List[str]
    ) -> Dict[str, Any]:
        """
        Register or update a serving node.
        
        Args:
            wallet_address: Miner wallet address
            endpoint_url: Serving node HTTP endpoint URL
            available_lora_list: List of LoRA adapter names this node can serve
            
        Returns:
            Dictionary with registration result
        """
        try:
            lora_list_json = self._json_dumps(available_lora_list)
            now = datetime.utcnow()
            
            if self.config.is_postgresql():
                if not self.database._db or not self.database._db.pool:
                    await self.database.connect()
                
                async with self.database._db.pool.acquire() as conn:
                    # Check if exists (using JSONB for PostgreSQL)
                    existing = await conn.fetchrow(
                        "SELECT id FROM serving_nodes WHERE wallet_address = $1",
                        wallet_address
                    )
                    
                    if existing:
                        # Update
                        await conn.execute("""
                            UPDATE serving_nodes 
                            SET endpoint_url = $1, available_lora_list = $2::jsonb, 
                                status = 'active', last_heartbeat = $3, updated_at = $4
                            WHERE wallet_address = $5
                        """, endpoint_url, lora_list_json, now, now, wallet_address)
                        logger.info(f"Updated serving node: {wallet_address}")
                        return {"success": True, "action": "updated", "node_id": existing['id']}
                    else:
                        # Insert
                        node_id = await conn.fetchval("""
                            INSERT INTO serving_nodes (wallet_address, endpoint_url, available_lora_list, status, last_heartbeat, created_at, updated_at)
                            VALUES ($1, $2, $3::jsonb, 'active', $4, $5, $6)
                            RETURNING id
                        """, wallet_address, endpoint_url, lora_list_json, now, now, now)
                        logger.info(f"Registered new serving node: {wallet_address}")
                        return {"success": True, "action": "created", "node_id": node_id}
            else:
                # SQLite
                if not self.database._connection:
                    await self.database.connect()
                
                cursor = await self.database._connection.cursor()
                
                await cursor.execute("SELECT id FROM serving_nodes WHERE wallet_address = ?", (wallet_address,))
                existing = await cursor.fetchone()
                
                if existing:
                    # Update
                    await cursor.execute("""
                        UPDATE serving_nodes 
                        SET endpoint_url = ?, available_lora_list = ?, 
                            status = 'active', last_heartbeat = ?, updated_at = ?
                        WHERE wallet_address = ?
                    """, (endpoint_url, lora_list_json, now, now, wallet_address))
                    await self.database._connection.commit()
                    logger.info(f"Updated serving node: {wallet_address}")
                    return {"success": True, "action": "updated", "node_id": existing[0]}
                else:
                    # Insert
                    await cursor.execute("""
                        INSERT INTO serving_nodes (wallet_address, endpoint_url, available_lora_list, status, last_heartbeat, created_at, updated_at)
                        VALUES (?, ?, ?, 'active', ?, ?, ?)
                    """, (wallet_address, endpoint_url, lora_list_json, now, now, now))
                    await self.database._connection.commit()
                    node_id = cursor.lastrowid
                    logger.info(f"Registered new serving node: {wallet_address}")
                    return {"success": True, "action": "created", "node_id": node_id}
        except Exception as e:
            logger.error(f"Failed to register serving node {wallet_address}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def update_serving_node_heartbeat(
        self,
        wallet_address: str,
        status: Optional[str] = None,
        current_load: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update serving node heartbeat.
        
        Args:
            wallet_address: Miner wallet address
            status: Optional status update (active, idle, busy)
            current_load: Optional current load (number of active requests)
            
        Returns:
            Dictionary with update result
        """
        try:
            now = datetime.utcnow()
            
            if self.config.is_postgresql():
                if not self.database._db or not self.database._db.pool:
                    await self.database.connect()
                
                async with self.database._db.pool.acquire() as conn:
                    # Check if exists
                    existing = await conn.fetchrow(
                        "SELECT id FROM serving_nodes WHERE wallet_address = $1",
                        wallet_address
                    )
                    
                    if not existing:
                        return {"success": False, "error": "Serving node not found"}
                    
                    # Build update query
                    if status and current_load is not None:
                        await conn.execute(
                            "UPDATE serving_nodes SET last_heartbeat = $1, updated_at = $2, status = $3, current_load = $4 WHERE wallet_address = $5",
                            now, now, status, current_load, wallet_address
                        )
                    elif status:
                        await conn.execute(
                            "UPDATE serving_nodes SET last_heartbeat = $1, updated_at = $2, status = $3 WHERE wallet_address = $4",
                            now, now, status, wallet_address
                        )
                    elif current_load is not None:
                        await conn.execute(
                            "UPDATE serving_nodes SET last_heartbeat = $1, updated_at = $2, current_load = $3 WHERE wallet_address = $4",
                            now, now, current_load, wallet_address
                        )
                    else:
                        await conn.execute(
                            "UPDATE serving_nodes SET last_heartbeat = $1, updated_at = $2 WHERE wallet_address = $3",
                            now, now, wallet_address
                        )
                    return {"success": True}
            else:
                # SQLite
                if not self.database._connection:
                    await self.database.connect()
                
                cursor = await self.database._connection.cursor()
                
                await cursor.execute("SELECT id FROM serving_nodes WHERE wallet_address = ?", (wallet_address,))
                existing = await cursor.fetchone()
                
                if not existing:
                    return {"success": False, "error": "Serving node not found"}
                
                # Build update query
                updates = ["last_heartbeat = ?", "updated_at = ?"]
                params = [now, now]
                
                if status:
                    updates.append("status = ?")
                    params.append(status)
                
                if current_load is not None:
                    updates.append("current_load = ?")
                    params.append(current_load)
                
                params.append(wallet_address)
                
                await cursor.execute(
                    f"UPDATE serving_nodes SET {', '.join(updates)} WHERE wallet_address = ?",
                    params
                )
                await self.database._connection.commit()
                return {"success": True}
        except Exception as e:
            logger.error(f"Failed to update heartbeat for {wallet_address}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def get_serving_nodes_for_lora(
        self,
        lora_name: str,
        max_age_seconds: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Get list of serving nodes that can serve a specific LoRA.
        
        Args:
            lora_name: LoRA adapter name
            max_age_seconds: Maximum age of last heartbeat (default: 60s)
            
        Returns:
            List of serving node dictionaries
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=max_age_seconds)
            
            if self.config.is_postgresql():
                if not self.database._db or not self.database._db.pool:
                    await self.database.connect()
                
                async with self.database._db.pool.acquire() as conn:
                    # PostgreSQL: Use JSONB contains operator
                    rows = await conn.fetch("""
                        SELECT wallet_address, endpoint_url, available_lora_list, status, current_load, last_heartbeat
                        FROM serving_nodes
                        WHERE status = 'active' AND last_heartbeat >= $1
                          AND available_lora_list @> $2::jsonb
                    """, cutoff_time, self._json_dumps([lora_name]))
                    
                    return [
                        {
                            "wallet_address": row['wallet_address'],
                            "endpoint_url": row['endpoint_url'],
                            "available_lora_list": row['available_lora_list'],
                            "status": row['status'],
                            "current_load": row['current_load'],
                            "last_heartbeat": row['last_heartbeat'].isoformat() if row['last_heartbeat'] else None,
                        }
                        for row in rows
                    ]
            else:
                # SQLite: Fetch all active nodes and filter in Python
                if not self.database._connection:
                    await self.database.connect()
                
                cursor = await self.database._connection.cursor()
                await cursor.execute("""
                    SELECT wallet_address, endpoint_url, available_lora_list, status, current_load, last_heartbeat
                    FROM serving_nodes
                    WHERE status = 'active' AND last_heartbeat >= ?
                """, (cutoff_time,))
                
                rows = await cursor.fetchall()
                filtered_nodes = []
                for row in rows:
                    lora_list = self._json_loads(row[2])
                    if isinstance(lora_list, list) and lora_name in lora_list:
                        filtered_nodes.append({
                            "wallet_address": row[0],
                            "endpoint_url": row[1],
                            "available_lora_list": lora_list,
                            "status": row[3],
                            "current_load": row[4],
                            "last_heartbeat": row[5],
                        })
                
                return filtered_nodes
        except Exception as e:
            logger.error(f"Failed to get serving nodes for LoRA {lora_name}: {e}", exc_info=True)
            return []
    
    async def cleanup_stale_nodes(self, max_age_seconds: int = 120) -> int:
        """
        Remove serving nodes that haven't sent heartbeat in max_age_seconds.
        
        Args:
            max_age_seconds: Maximum age before marking as offline (default: 120s)
            
        Returns:
            Number of nodes cleaned up
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=max_age_seconds)
            now = datetime.utcnow()
            
            if self.config.is_postgresql():
                if not self.database._db or not self.database._db.pool:
                    await self.database.connect()
                
                async with self.database._db.pool.acquire() as conn:
                    result = await conn.execute("""
                        UPDATE serving_nodes 
                        SET status = 'offline', updated_at = $1
                        WHERE last_heartbeat < $2
                    """, now, cutoff_time)
                    count = result.split()[-1] if ' ' in result else 0
                    try:
                        count = int(count)
                    except (ValueError, AttributeError):
                        count = 0
                    
                    if count > 0:
                        logger.info(f"Marked {count} stale serving nodes as offline")
                    return count
            else:
                # SQLite
                if not self.database._connection:
                    await self.database.connect()
                
                cursor = await self.database._connection.cursor()
                await cursor.execute("""
                    UPDATE serving_nodes 
                    SET status = 'offline', updated_at = ?
                    WHERE last_heartbeat < ?
                """, (now, cutoff_time))
                count = cursor.rowcount
                await self.database._connection.commit()
                
                if count > 0:
                    logger.info(f"Marked {count} stale serving nodes as offline")
                return count
        except Exception as e:
            logger.error(f"Failed to cleanup stale nodes: {e}", exc_info=True)
            return 0

