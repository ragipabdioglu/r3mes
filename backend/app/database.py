"""
Database - Kredi ve Cüzdan Veritabanı Yönetimi

Production-ready database management with proper error handling and security.
"""

import sqlite3
import json
import threading
import time
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
import os
import requests
import logging

from .exceptions import InvalidInputError, DatabaseError, ProductionConfigurationError
from .config import get_config, validate_localhost_usage
from .constants import (
    CREDITS_PER_BLOCK, API_KEY_PREFIX, API_KEY_LENGTH,
    DEFAULT_DATABASE_CACHE_SIZE, DEFAULT_DATABASE_TIMEOUT,
    DEFAULT_DATABASE_PATH, DEFAULT_CHAIN_JSON_PATH
)

logger = logging.getLogger(__name__)


class Database:
    """
    Production-ready Kredi ve Cüzdan Veritabanı Yönetimi
    
    Features:
    - SQLite-based user and credit management
    - Blockchain synchronization
    - Automatic credit distribution
    - Production security validation
    """
    
    def __init__(self, db_path: Optional[str] = None, chain_json_path: Optional[str] = None):
        """
        Initialize database with production-ready configuration.
        
        Args:
            db_path: SQLite database file path
            chain_json_path: Blockchain JSON file path
        """
        config = get_config()
        
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", DEFAULT_DATABASE_PATH)
        
        if chain_json_path is None:
            chain_json_path = os.getenv("CHAIN_JSON_PATH", DEFAULT_CHAIN_JSON_PATH)
        
        self.db_path = db_path
        self.chain_json_path = chain_json_path
        self.lock = threading.Lock()
        
        # RPC endpoint configuration with production validation
        self.rpc_endpoint = config.BLOCKCHAIN_RPC_URL or config.RPC_URL
        
        # Validate RPC endpoint for production
        if config.ENV == "production":
            validate_localhost_usage(self.rpc_endpoint, "BLOCKCHAIN_RPC_URL")
        
        # Create database directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create database and tables
        self._init_database()
        
        logger.info(f"Database initialized: {db_path}")
        logger.info(f"RPC endpoint: {self.rpc_endpoint}")
    
    def _init_database(self):
        """Veritabanı tablolarını oluşturur."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable WAL (Write-Ahead Logging) mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")  # Better performance with WAL
        cursor.execute(f"PRAGMA cache_size={DEFAULT_DATABASE_CACHE_SIZE}")  # 64MB cache
        cursor.execute("PRAGMA foreign_keys=ON")
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                wallet_address TEXT PRIMARY KEY,
                credits REAL DEFAULT 0.0,
                is_miner BOOLEAN DEFAULT 0,
                last_mining_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Mining stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mining_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT NOT NULL,
                hashrate REAL DEFAULT 0.0,
                gpu_temperature REAL DEFAULT 0.0,
                blocks_found INTEGER DEFAULT 0,
                uptime_percentage REAL DEFAULT 0.0,
                network_difficulty REAL DEFAULT 0.0,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES users(wallet_address)
            )
        """)
        
        # Earnings history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS earnings_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT NOT NULL,
                earnings REAL DEFAULT 0.0,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES users(wallet_address)
            )
        """)
        
        # Hashrate history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hashrate_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT NOT NULL,
                hashrate REAL DEFAULT 0.0,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES users(wallet_address)
            )
        """)
        
        # API Keys table (with hashed keys for security)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key_hash TEXT UNIQUE NOT NULL,
                wallet_address TEXT NOT NULL,
                name TEXT,
                is_active BOOLEAN DEFAULT 1,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES users(wallet_address)
            )
        """)
        
        # Create index on api_key_hash for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_key_hash ON api_keys(api_key_hash)
        """)
        
        # Apply additional performance indexes
        self._apply_performance_indexes(cursor)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized with WAL mode")
    
    def _apply_performance_indexes(self, cursor):
        """Apply performance indexes for better query performance."""
        from .database_optimization import get_sqlite_index_queries
        
        indexes = get_sqlite_index_queries()
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                logger.debug(f"Applied index: {index_sql}")
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
                # Continue with other indexes
    
    def add_credits(self, wallet: str, amount: float) -> bool:
        """
        Kullanıcıya kredi ekler.
        
        Args:
            wallet: Cüzdan adresi
            amount: Eklenecek kredi miktarı
            
        Returns:
            True if successful
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT credits FROM users WHERE wallet_address = ?", (wallet,))
            result = cursor.fetchone()
            
            if result:
                # Update existing user
                new_credits = result[0] + amount
                cursor.execute(
                    "UPDATE users SET credits = ? WHERE wallet_address = ?",
                    (new_credits, wallet)
                )
            else:
                # Create new user
                cursor.execute(
                    "INSERT INTO users (wallet_address, credits) VALUES (?, ?)",
                    (wallet, amount)
                )
            
            conn.commit()
            conn.close()
            return True
    
    def deduct_credit(self, wallet: str, amount: float) -> bool:
        """
        Kullanıcıdan kredi düşer.
        
        Args:
            wallet: Cüzdan adresi
            amount: Düşülecek kredi miktarı
            
        Returns:
            True if successful, False if insufficient credits
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT credits FROM users WHERE wallet_address = ?", (wallet,))
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return False
            
            current_credits = result[0]
            if current_credits < amount:
                conn.close()
                return False
            
            new_credits = current_credits - amount
            cursor.execute(
                "UPDATE users SET credits = ? WHERE wallet_address = ?",
                (new_credits, wallet)
            )
            
            conn.commit()
            conn.close()
            return True
    
    def check_credits(self, wallet: str) -> float:
        """
        Kullanıcının kalan kredisini döndürür.
        
        Args:
            wallet: Cüzdan adresi
            
        Returns:
            Kalan kredi miktarı
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT credits FROM users WHERE wallet_address = ?", (wallet,))
        result = cursor.fetchone()
        
        conn.close()
        return result[0] if result else 0.0
    
    def sync_with_blockchain(self):
        """
        Blockchain JSON dosyasını okur ve yeni blokları işler.
        Eğer yeni bloklar varsa, o blokları bulan cüzdan adreslerine otomatik kredi yükler.
        (Örn: 1 Blok = 100 Kredi)
        
        Raises:
            FileNotFoundError: If chain_json_path doesn't exist
            json.JSONDecodeError: If JSON file is invalid
            ValueError: If block data is invalid
            RuntimeError: If credit addition fails
        """
        if not Path(self.chain_json_path).exists():
            logger.warning(f"Chain JSON file not found: {self.chain_json_path}")
            return
        
        try:
            with open(self.chain_json_path, 'r') as f:
                chain_data = json.load(f)
            
            # Process new blocks (simplified - in production would track last processed block)
            blocks = chain_data.get('blocks', [])
            
            if not isinstance(blocks, list):
                raise InvalidInputError(f"Invalid blocks format: expected list, got {type(blocks)}")
            
            processed_count = 0
            for block in blocks:
                if not isinstance(block, dict):
                    logger.warning(f"Invalid block format: {type(block)}, skipping")
                    continue
                
                # Extract miner address from block
                miner_address = block.get('miner', '')
                if miner_address:
                    try:
                        # Award credits for mining (1 block = 100 credits)
                        success = self.add_credits(miner_address, 100.0)
                        if success:
                            processed_count += 1
                            logger.info(f"Awarded 100 credits to {miner_address} for block {block.get('height', 'unknown')}")
                        else:
                            logger.warning(f"Failed to award credits to {miner_address} (non-critical, continuing)")
                    except (sqlite3.Error, sqlite3.OperationalError) as e:
                        # Database errors - log but continue with other blocks
                        logger.error(f"Database error awarding credits to {miner_address}: {e}", exc_info=True)
                        continue
                    except Exception as e:
                        # Other errors - log but continue with other blocks (non-critical)
                        logger.warning(f"Error awarding credits to {miner_address}: {e}")
                        continue
            
            if processed_count > 0:
                logger.info(f"Blockchain sync completed: {processed_count} blocks processed")
        
        except FileNotFoundError:
            # Non-critical: chain.json might not exist yet
            logger.warning(f"Chain JSON file not found: {self.chain_json_path}")
            return
        except json.JSONDecodeError as e:
            # Critical: Invalid JSON means corrupted file
            logger.error(f"Invalid JSON in chain file: {e}", exc_info=True)
            raise InvalidInputError(f"Chain JSON file is corrupted: {e}") from e
        except ValueError as e:
            # Critical: Invalid data format
            logger.error(f"Invalid data format in chain file: {e}", exc_info=True)
            raise InvalidInputError(f"Invalid data format in chain file: {e}") from e
        except InvalidInputError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Unexpected errors - log and re-raise as critical
            logger.error(f"Unexpected error during blockchain sync: {e}", exc_info=True)
            raise RuntimeError(f"Blockchain sync failed: {e}") from e
            # Re-raise to prevent silent failures
            raise DatabaseError(f"Blockchain sync failed: {e}") from e
    
    # _sync_loop method removed - polling eliminated
    # Sync is now handled by AsyncDatabase via async tasks and event-driven mechanisms
    # For manual sync, call sync_with_blockchain() directly when needed
    
    def get_user_info(self, wallet: str) -> Optional[Dict]:
        """
        Kullanıcı bilgilerini döndürür.
        
        Args:
            wallet: Cüzdan adresi
            
        Returns:
            Kullanıcı bilgileri dict veya None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT wallet_address, credits, is_miner, last_mining_time FROM users WHERE wallet_address = ?",
            (wallet,)
        )
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return {
                "wallet_address": result[0],
                "credits": result[1],
                "is_miner": bool(result[2]),
                "last_mining_time": result[3]
            }
        return None
    
    def get_network_stats(self) -> Dict:
        """
        Ağ istatistiklerini döndürür.
        
        Returns:
            Ağ istatistikleri dict
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count active miners
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_miner = 1")
        active_miners = cursor.fetchone()[0]
        
        # Count total users
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        # Total credits in system
        cursor.execute("SELECT SUM(credits) FROM users")
        total_credits = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        # Get block height from chain.json if available
        block_height = None
        if Path(self.chain_json_path).exists():
            try:
                with open(self.chain_json_path, 'r') as f:
                    chain_data = json.load(f)
                    blocks = chain_data.get('blocks', [])
                    if blocks:
                        # Get the highest block height
                        block_height = max((block.get('height', 0) for block in blocks), default=0)
            except Exception as e:
                logger.warning(f"Failed to read block height from chain.json: {e}")
        
        return {
            "active_miners": active_miners,
            "total_users": total_users,
            "total_credits": total_credits,
            "block_height": block_height
        }
    
    def get_recent_blocks(self, limit: int = 10) -> List[Dict]:
        """
        Son blokları döndürür.
        
        Args:
            limit: Döndürülecek blok sayısı
            
        Returns:
            Blok listesi
        """
        result = []
        
        # Try blockchain RPC first
        try:
            # Get latest block height
            database_timeout = int(os.getenv("BACKEND_DATABASE_TIMEOUT", "2"))
            status_response = requests.get(f"{self.rpc_endpoint}/status", timeout=database_timeout)
            if status_response.status_code == 200:
                status_data = status_response.json()
                latest_height = int(status_data.get('result', {}).get('sync_info', {}).get('latest_block_height', 0))
                
                # Fetch recent blocks
                for i in range(min(limit, latest_height)):
                    height = latest_height - i
                    if height <= 0:
                        break
                    
                    database_timeout = int(os.getenv("BACKEND_DATABASE_TIMEOUT", "2"))
                    block_response = requests.get(f"{self.rpc_endpoint}/block?height={height}", timeout=database_timeout)
                    if block_response.status_code == 200:
                        block_data = block_response.json()
                        block_info = block_data.get('result', {}).get('block', {})
                        
                        # Extract block information
                        result.append({
                            "height": height,
                            "miner": block_info.get('proposer_address', ''),
                            "timestamp": block_info.get('header', {}).get('time', ''),
                            "hash": block_info.get('header', {}).get('hash', '')
                        })
        except Exception as e:
            # Fallback to chain.json if RPC fails
            if Path(self.chain_json_path).exists():
                try:
                    with open(self.chain_json_path, 'r') as f:
                        chain_data = json.load(f)
                    
                    blocks = chain_data.get('blocks', [])
                    
                    # Sort by height (descending) and take the most recent
                    sorted_blocks = sorted(
                        blocks,
                        key=lambda b: b.get('height', 0),
                        reverse=True
                    )[:limit]
                    
                    # Format blocks
                    for block in sorted_blocks:
                        result.append({
                            "height": block.get('height', 0),
                            "miner": block.get('miner', ''),
                            "timestamp": block.get('timestamp', ''),
                            "hash": block.get('hash', '')
                        })
                except Exception as e2:
                    logger.warning(f"Failed to read recent blocks from chain.json: {e2}")
        
        return result
    
    def get_miner_stats(self, wallet: str) -> Dict:
        """
        Miner istatistiklerini döndürür.
        
        Args:
            wallet: Cüzdan adresi
            
        Returns:
            Miner istatistikleri dict
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user info
        user_info = self.get_user_info(wallet)
        if not user_info:
            conn.close()
            # Try to get network difficulty from blockchain or use default
            network_difficulty = self._get_network_difficulty()
            return {
                "wallet_address": wallet,
                "total_earnings": 0.0,
                "hashrate": 0.0,
                "gpu_temperature": 0.0,
                "blocks_found": 0,
                "uptime_percentage": 0.0,
                "network_difficulty": network_difficulty
            }
        
        # Get latest mining stats
        cursor.execute("""
            SELECT hashrate, gpu_temperature, blocks_found, uptime_percentage, network_difficulty, recorded_at
            FROM mining_stats
            WHERE wallet_address = ?
            ORDER BY recorded_at DESC
            LIMIT 1
        """, (wallet,))
        stats = cursor.fetchone()
        
        conn.close()
        
        # Get network difficulty: prefer from stats, then blockchain, then default
        network_difficulty = 1234.0  # Default fallback
        if stats and stats[4]:
            network_difficulty = float(stats[4])
        else:
            # Try to fetch from blockchain
            network_difficulty = self._get_network_difficulty()
        
        return {
            "wallet_address": wallet,
            "total_earnings": user_info['credits'],
            "hashrate": stats[0] if stats and stats[0] else 0.0,
            "gpu_temperature": stats[1] if stats and stats[1] else 0.0,
            "blocks_found": stats[2] if stats and stats[2] else 0,
            "uptime_percentage": stats[3] if stats and stats[3] else 0.0,
            "network_difficulty": network_difficulty
        }
    
    def _get_network_difficulty(self) -> float:
        """
        Get network difficulty from blockchain or return default.
        
        Returns:
            Network difficulty value (default: 1234.0)
        """
        try:
            from .blockchain_query_client import BlockchainQueryClient
            client = BlockchainQueryClient()
            # Query params from blockchain
            params_data = client._query_rest("/remes/remes/v1/params")
            if "params" in params_data:
                params = params_data["params"]
                # Extract mining difficulty from params (if available)
                difficulty_str = params.get("mining_difficulty") or params.get("network_difficulty")
                if difficulty_str:
                    try:
                        difficulty = float(difficulty_str)
                        logger.debug(f"Fetched network difficulty from blockchain: {difficulty}")
                        return difficulty
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid difficulty value from blockchain: {difficulty_str}")
        except Exception as e:
            logger.debug(f"Could not fetch network difficulty from blockchain: {e}, using default")
        
        # Default fallback
        return 1234.0
    
    def get_earnings_history(self, wallet: str, days: int = 7) -> List[Dict]:
        """
        Earnings geçmişini döndürür.
        
        Args:
            wallet: Cüzdan adresi
            days: Kaç günlük geçmiş
            
        Returns:
            Earnings geçmişi listesi
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get earnings for the last N days
        cursor.execute("""
            SELECT DATE(recorded_at) as date, SUM(earnings) as total_earnings
            FROM earnings_history
            WHERE wallet_address = ? AND recorded_at >= datetime('now', '-' || ? || ' days')
            GROUP BY DATE(recorded_at)
            ORDER BY date ASC
        """, (wallet, days))
        
        results = cursor.fetchall()
        conn.close()
        
        # Format results
        earnings_data = []
        for row in results:
            earnings_data.append({
                "date": row[0],
                "earnings": row[1] or 0.0
            })
        
        # Fill missing days with 0
        today = datetime.now().date()
        all_dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days-1, -1, -1)]
        
        earnings_dict = {item['date']: item['earnings'] for item in earnings_data}
        formatted_data = []
        for date_str in all_dates:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            formatted_data.append({
                "date": date_obj.strftime('%b %d'),
                "earnings": earnings_dict.get(date_str, 0.0)
            })
        
        return formatted_data
    
    def get_hashrate_history(self, wallet: str, days: int = 7) -> List[Dict]:
        """
        Hashrate geçmişini döndürür.
        
        Args:
            wallet: Cüzdan adresi
            days: Kaç günlük geçmiş
            
        Returns:
            Hashrate geçmişi listesi
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get hashrate for the last N days
        cursor.execute("""
            SELECT DATE(recorded_at) as date, AVG(hashrate) as avg_hashrate
            FROM hashrate_history
            WHERE wallet_address = ? AND recorded_at >= datetime('now', '-' || ? || ' days')
            GROUP BY DATE(recorded_at)
            ORDER BY date ASC
        """, (wallet, days))
        
        results = cursor.fetchall()
        conn.close()
        
        # Format results
        hashrate_data = []
        for row in results:
            hashrate_data.append({
                "date": row[0],
                "hashrate": row[1] or 0.0
            })
        
        # Fill missing days with 0
        today = datetime.now().date()
        all_dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days-1, -1, -1)]
        
        hashrate_dict = {item['date']: item['hashrate'] for item in hashrate_data}
        formatted_data = []
        for date_str in all_dates:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            formatted_data.append({
                "date": date_obj.strftime('%b %d'),
                "hashrate": hashrate_dict.get(date_str, 0.0)
            })
        
        return formatted_data
    
    def update_mining_stats(self, wallet: str, hashrate: float, gpu_temp: float, 
                           blocks_found: int, uptime: float, difficulty: float):
        """
        Mining istatistiklerini günceller.
        
        Args:
            wallet: Cüzdan adresi
            hashrate: Hashrate değeri
            gpu_temp: GPU sıcaklığı
            blocks_found: Bulunan blok sayısı
            uptime: Uptime yüzdesi
            difficulty: Network zorluğu
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO mining_stats 
                (wallet_address, hashrate, gpu_temperature, blocks_found, uptime_percentage, network_difficulty)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (wallet, hashrate, gpu_temp, blocks_found, uptime, difficulty))
            
            conn.commit()
            conn.close()
    
    # ========== API Key Management ==========
    
    def create_api_key(self, wallet_address: str, name: Optional[str] = None, 
                      expires_days: Optional[int] = None) -> Dict[str, str]:
        """
        Yeni bir API key oluşturur.
        
        Args:
            wallet_address: Cüzdan adresi
            name: API key için isim (opsiyonel)
            expires_days: Kaç gün sonra expire olacak (None ise expire olmaz)
            
        Returns:
            {"api_key": "...", "name": "...", "created_at": "..."}
        """
        # Generate secure API key (32 bytes = 64 hex characters)
        api_key = f"r3mes_{secrets.token_urlsafe(32)}"
        
        # Hash the API key for secure storage
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Calculate expiration date
        expires_at = None
        if expires_days:
            expires_at = (datetime.now() + timedelta(days=expires_days)).isoformat()
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO api_keys (api_key_hash, wallet_address, name, expires_at)
                    VALUES (?, ?, ?, ?)
                """, (api_key_hash, wallet_address, name, expires_at))
                
                conn.commit()
                
                return {
                    "api_key": api_key,  # Return plain key to user (only time they see it)
                    "name": name or "Unnamed",
                    "created_at": datetime.now().isoformat(),
                    "expires_at": expires_at
                }
            except sqlite3.IntegrityError:
                # API key hash collision (extremely unlikely), try again
                conn.close()
                return self.create_api_key(wallet_address, name, expires_days)
            finally:
                conn.close()
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, any]]:
        """
        API key'i doğrular ve sahibini döndürür.
        
        Args:
            api_key: Doğrulanacak API key
            
        Returns:
            {"wallet_address": "...", "name": "...", "is_active": True/False} veya None
        """
        # Hash the provided API key for comparison
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT wallet_address, name, is_active, expires_at, last_used
                FROM api_keys
                WHERE api_key_hash = ?
            """, (api_key_hash,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return None
            
            wallet_address, name, is_active, expires_at, last_used = result
            
            # Check if expired
            if expires_at:
                expires_dt = datetime.fromisoformat(expires_at)
                if datetime.now() > expires_dt:
                    return None
            
            # Update last_used timestamp
            self._update_api_key_last_used(api_key_hash)
            
            return {
                "wallet_address": wallet_address,
                "name": name,
                "is_active": bool(is_active),
                "expires_at": expires_at,
                "last_used": last_used
            }
    
    def _update_api_key_last_used(self, api_key_hash: str):
        """API key'in son kullanım zamanını günceller."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE api_keys
                SET last_used = CURRENT_TIMESTAMP
                WHERE api_key_hash = ?
            """, (api_key_hash,))
            
            conn.commit()
            conn.close()
    
    def list_api_keys(self, wallet_address: str) -> List[Dict[str, any]]:
        """
        Bir cüzdan için tüm API key'leri listeler.
        
        Args:
            wallet_address: Cüzdan adresi
            
        Returns:
            API key listesi (hashed keys are not returned for security)
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT api_key_hash, name, is_active, created_at, expires_at, last_used
                FROM api_keys
                WHERE wallet_address = ?
                ORDER BY created_at DESC
            """, (wallet_address,))
            
            results = cursor.fetchall()
            conn.close()
            
            keys = []
            for row in results:
                api_key_hash, name, is_active, created_at, expires_at, last_used = row
                
                # Show only hash prefix for identification (first 12 chars of hash)
                key_id = f"r3mes_***{api_key_hash[:8]}"
                
                keys.append({
                    "key_id": key_id,  # For identification only
                    "name": name,
                    "is_active": bool(is_active),
                    "created_at": created_at,
                    "expires_at": expires_at,
                    "last_used": last_used
                })
            
            return keys
    
    def revoke_api_key(self, api_key: str, wallet_address: str) -> bool:
        """
        Bir API key'i iptal eder (sadece sahibi iptal edebilir).
        
        Args:
            api_key: İptal edilecek API key
            wallet_address: Cüzdan adresi (doğrulama için)
            
        Returns:
            True if successful, False otherwise
        """
        # Hash the API key for lookup
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verify ownership
            cursor.execute("""
                SELECT wallet_address FROM api_keys
                WHERE api_key_hash = ?
            """, (api_key_hash,))
            
            result = cursor.fetchone()
            if not result or result[0] != wallet_address:
                conn.close()
                return False
            
            # Revoke (set is_active to False)
            cursor.execute("""
                UPDATE api_keys
                SET is_active = 0
                WHERE api_key_hash = ? AND wallet_address = ?
            """, (api_key_hash, wallet_address))
            
            conn.commit()
            conn.close()
            return True
    
    def delete_api_key(self, api_key: str, wallet_address: str) -> bool:
        """
        Bir API key'i tamamen siler (sadece sahibi silebilir).
        
        Args:
            api_key: Silinecek API key
            wallet_address: Cüzdan adresi (doğrulama için)
            
        Returns:
            True if successful, False otherwise
        """
        # Hash the API key for lookup
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verify ownership
            cursor.execute("""
                SELECT wallet_address FROM api_keys
                WHERE api_key_hash = ?
            """, (api_key_hash,))
            
            result = cursor.fetchone()
            if not result or result[0] != wallet_address:
                conn.close()
                return False
            
            # Delete
            cursor.execute("""
                DELETE FROM api_keys
                WHERE api_key_hash = ? AND wallet_address = ?
            """, (api_key_hash, wallet_address))
            
            conn.commit()
            conn.close()
            return True

