"""Shared state management for data persistence and coordination"""

import asyncio
import json
import pickle
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import redis
from loguru import logger

from .data_processor import TickData
from .pattern_recognition import PatternMatch
from .statistical_analyzer import TradingSignal
from .volume_analyzer import VolumeAnomaly
from .watchlist_manager import TradingOpportunity


@dataclass
class StateSnapshot:
    """Snapshot of the entire market analyst state"""

    timestamp: datetime
    symbols_monitored: list[str]
    latest_ticks: dict[str, dict[str, Any]]
    active_opportunities: dict[str, list[dict[str, Any]]]
    watchlists: dict[str, list[dict[str, Any]]]
    market_statistics: dict[str, dict[str, Any]]
    system_metrics: dict[str, Any]


class DatabaseManager:
    """Manages SQLite database for persistent storage"""

    def __init__(self, db_path: str = "data/market_analyst.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Trading signals table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    technical_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Trading opportunities table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_opportunities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    opportunity_type TEXT NOT NULL,
                    score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    expected_move REAL,
                    entry_price REAL,
                    target_price REAL,
                    stop_loss REAL,
                    risk_reward_ratio REAL,
                    alert_level INTEGER,
                    signals TEXT,
                    reasoning TEXT,
                    supporting_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Volume anomalies table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS volume_anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity REAL NOT NULL,
                    current_volume INTEGER,
                    expected_volume INTEGER,
                    volume_ratio REAL,
                    description TEXT,
                    supporting_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Pattern matches table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pattern_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    direction TEXT,
                    description TEXT,
                    key_levels TEXT,
                    timeframe TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Market statistics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS market_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price_stats TEXT,
                    volume_stats TEXT,
                    volatility_stats TEXT,
                    trend_stats TEXT,
                    momentum_stats TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Watchlist items table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS watchlist_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    watchlist_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    priority INTEGER,
                    target_price REAL,
                    stop_loss REAL,
                    notes TEXT,
                    tags TEXT,
                    alert_conditions TEXT,
                    added_date DATETIME,
                    last_analysis DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(watchlist_name, symbol)
                )
            """
            )

            # System state table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS system_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    state_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indices for better performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON trading_signals(symbol, timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_opportunities_symbol_score ON trading_opportunities(symbol, score)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_anomalies_symbol_timestamp ON volume_anomalies(symbol, timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_symbol_timestamp ON pattern_matches(symbol, timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_stats_symbol_timestamp ON market_statistics(symbol, timestamp)"
            )

            conn.commit()
            logger.info("Database initialized successfully")

    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()

    def store_trading_signal(self, signal: TradingSignal):
        """Store trading signal in database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO trading_signals
                    (symbol, signal_type, strength, confidence, reasoning, technical_data, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        signal.symbol,
                        signal.signal_type,
                        signal.strength,
                        signal.confidence,
                        json.dumps(signal.reasoning),
                        json.dumps(signal.technical_data),
                        signal.timestamp,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing trading signal: {str(e)}")

    def store_trading_opportunity(self, opportunity: TradingOpportunity):
        """Store trading opportunity in database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO trading_opportunities
                    (symbol, opportunity_type, score, confidence, expected_move, entry_price,
                     target_price, stop_loss, risk_reward_ratio, alert_level, signals,
                     reasoning, supporting_data, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        opportunity.symbol,
                        opportunity.opportunity_type.value,
                        opportunity.score,
                        opportunity.confidence,
                        opportunity.expected_move,
                        opportunity.entry_price,
                        opportunity.target_price,
                        opportunity.stop_loss,
                        opportunity.risk_reward_ratio,
                        opportunity.alert_level.value,
                        json.dumps(opportunity.signals),
                        json.dumps(opportunity.reasoning),
                        json.dumps(opportunity.supporting_data),
                        opportunity.timestamp,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing trading opportunity: {str(e)}")

    def store_volume_anomaly(self, anomaly: VolumeAnomaly):
        """Store volume anomaly in database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO volume_anomalies
                    (symbol, anomaly_type, severity, current_volume, expected_volume,
                     volume_ratio, description, supporting_data, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        anomaly.symbol,
                        anomaly.anomaly_type,
                        anomaly.severity,
                        anomaly.current_volume,
                        anomaly.expected_volume,
                        anomaly.volume_ratio,
                        anomaly.description,
                        json.dumps(anomaly.supporting_data),
                        anomaly.timestamp,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing volume anomaly: {str(e)}")

    def get_recent_opportunities(
        self, symbol: str = None, hours: int = 24
    ) -> list[dict[str, Any]]:
        """Get recent trading opportunities"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if symbol:
                    query = """
                        SELECT * FROM trading_opportunities
                        WHERE symbol = ? AND timestamp > datetime('now', '-' || ? || ' hours')
                        ORDER BY score DESC, timestamp DESC
                    """
                    cursor.execute(query, (symbol, hours))
                else:
                    query = """
                        SELECT * FROM trading_opportunities
                        WHERE timestamp > datetime('now', '-' || ? || ' hours')
                        ORDER BY score DESC, timestamp DESC
                    """
                    cursor.execute(query, (hours,))

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error getting recent opportunities: {str(e)}")
            return []

    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                tables = [
                    "trading_signals",
                    "trading_opportunities",
                    "volume_anomalies",
                    "pattern_matches",
                    "market_statistics",
                ]

                for table in tables:
                    # Use parameterized query with table name validation
                    if table not in [
                        "trading_signals",
                        "trading_opportunities",
                        "volume_anomalies",
                        "pattern_matches",
                        "market_statistics",
                    ]:
                        continue

                    query = f"""
                        DELETE FROM {table}
                        WHERE created_at < datetime('now', '-' || ? || ' days')
                    """
                    cursor.execute(query, (days_to_keep,))

                conn.commit()
                logger.info(f"Cleaned up data older than {days_to_keep} days")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")


class RedisStateManager:
    """Manages real-time state using Redis"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()  # Test connection
            logger.info("Connected to Redis for state management")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory fallback: {str(e)}")
            self.redis_client = None

        # Fallback in-memory storage
        self.memory_store: dict[str, Any] = {}
        self.lock = threading.RLock()

    def set_state(self, key: str, value: Any, ttl: int | None = None):
        """Set state value with optional TTL"""
        try:
            if self.redis_client:
                if isinstance(value, dict | list):
                    value = json.dumps(value)

                if ttl:
                    self.redis_client.setex(key, ttl, value)
                else:
                    self.redis_client.set(key, value)
            else:
                with self.lock:
                    self.memory_store[key] = value

        except Exception as e:
            logger.error(f"Error setting state {key}: {str(e)}")
            # Fallback to memory
            with self.lock:
                self.memory_store[key] = value

    def get_state(self, key: str) -> Any | None:
        """Get state value"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return value
                return None
            else:
                with self.lock:
                    return self.memory_store.get(key)

        except Exception as e:
            logger.error(f"Error getting state {key}: {str(e)}")
            with self.lock:
                return self.memory_store.get(key)

    def delete_state(self, key: str):
        """Delete state value"""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                with self.lock:
                    self.memory_store.pop(key, None)

        except Exception as e:
            logger.error(f"Error deleting state {key}: {str(e)}")
            with self.lock:
                self.memory_store.pop(key, None)

    def set_latest_tick(self, symbol: str, tick_data: TickData):
        """Store latest tick data"""
        key = f"tick:{symbol}"
        self.set_state(key, tick_data.to_dict(), ttl=300)  # 5 minute TTL

    def get_latest_tick(self, symbol: str) -> dict[str, Any] | None:
        """Get latest tick data"""
        key = f"tick:{symbol}"
        return self.get_state(key)

    def set_analysis_result(self, symbol: str, analysis_type: str, result: Any):
        """Store analysis result"""
        key = f"analysis:{symbol}:{analysis_type}"
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        self.set_state(key, result, ttl=1800)  # 30 minute TTL

    def get_analysis_result(self, symbol: str, analysis_type: str) -> Any | None:
        """Get analysis result"""
        key = f"analysis:{symbol}:{analysis_type}"
        return self.get_state(key)

    def set_market_summary(self, summary: dict[str, Any]):
        """Store market summary"""
        self.set_state("market:summary", summary, ttl=300)

    def get_market_summary(self) -> dict[str, Any] | None:
        """Get market summary"""
        return self.get_state("market:summary")


class SharedStateManager:
    """Coordinates shared state across all market analyst components"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db_path: str = "data/market_analyst.db",
    ):
        self.db_manager = DatabaseManager(db_path)
        self.redis_manager = RedisStateManager(redis_url)

        # State synchronization
        self.sync_interval = 60  # seconds
        self.last_sync = datetime.now()

        # Component registration
        self.registered_components: set[str] = set()

        # Event callbacks
        self.event_callbacks: dict[str, list[callable]] = {
            "signal_generated": [],
            "opportunity_found": [],
            "anomaly_detected": [],
            "pattern_matched": [],
            "state_synced": [],
        }

    def register_component(self, component_name: str):
        """Register a component for state management"""
        self.registered_components.add(component_name)
        logger.info(f"Registered component: {component_name}")

    def store_trading_signal(self, signal: TradingSignal):
        """Store trading signal in both Redis and database"""
        # Store in Redis for real-time access
        self.redis_manager.set_analysis_result(signal.symbol, "signal", signal)

        # Store in database for persistence
        self.db_manager.store_trading_signal(signal)

        # Trigger callbacks
        self._trigger_event_callbacks("signal_generated", signal)

    def store_trading_opportunity(self, opportunity: TradingOpportunity):
        """Store trading opportunity"""
        # Store in Redis
        key = f"opportunity:{opportunity.symbol}"
        opportunities = self.redis_manager.get_state(key) or []
        opportunities.append(opportunity.to_dict())

        # Keep only top 5 opportunities per symbol
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        opportunities = opportunities[:5]

        self.redis_manager.set_state(key, opportunities, ttl=3600)

        # Store in database
        self.db_manager.store_trading_opportunity(opportunity)

        # Trigger callbacks
        self._trigger_event_callbacks("opportunity_found", opportunity)

    def store_volume_anomaly(self, anomaly: VolumeAnomaly):
        """Store volume anomaly"""
        # Store in Redis
        self.redis_manager.set_analysis_result(
            anomaly.symbol, "volume_anomaly", anomaly
        )

        # Store in database
        self.db_manager.store_volume_anomaly(anomaly)

        # Trigger callbacks
        self._trigger_event_callbacks("anomaly_detected", anomaly)

    def store_pattern_match(self, symbol: str, pattern: PatternMatch, timeframe: str):
        """Store pattern match"""
        pattern_data = {
            "symbol": symbol,
            "pattern_name": pattern.pattern_name,
            "confidence": pattern.confidence,
            "direction": pattern.direction,
            "description": pattern.description,
            "key_levels": pattern.key_levels,
            "timeframe": timeframe,
            "timestamp": datetime.now(),
        }

        # Store in Redis
        key = f"pattern:{symbol}"
        patterns = self.redis_manager.get_state(key) or []
        patterns.append(pattern_data)

        # Keep only recent patterns (last 10)
        patterns = sorted(patterns, key=lambda x: x["timestamp"], reverse=True)[:10]
        self.redis_manager.set_state(key, patterns, ttl=1800)

        # Store in database
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO pattern_matches
                    (symbol, pattern_name, confidence, direction, description, key_levels, timeframe, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol,
                        pattern.pattern_name,
                        pattern.confidence,
                        pattern.direction,
                        pattern.description,
                        json.dumps(pattern.key_levels),
                        timeframe,
                        datetime.now(),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing pattern match: {str(e)}")

        # Trigger callbacks
        self._trigger_event_callbacks("pattern_matched", pattern_data)

    def get_symbol_state(self, symbol: str) -> dict[str, Any]:
        """Get complete state for a symbol"""
        state = {
            "symbol": symbol,
            "latest_tick": self.redis_manager.get_latest_tick(symbol),
            "signal": self.redis_manager.get_analysis_result(symbol, "signal"),
            "opportunities": self.redis_manager.get_state(f"opportunity:{symbol}")
            or [],
            "patterns": self.redis_manager.get_state(f"pattern:{symbol}") or [],
            "volume_anomaly": self.redis_manager.get_analysis_result(
                symbol, "volume_anomaly"
            ),
            "timestamp": datetime.now(),
        }

        return state

    def get_market_state(self) -> dict[str, Any]:
        """Get overall market state"""
        summary = self.redis_manager.get_market_summary() or {}

        # Get recent opportunities from database
        recent_opportunities = self.db_manager.get_recent_opportunities(hours=6)

        state = {
            "market_summary": summary,
            "recent_opportunities": recent_opportunities,
            "registered_components": list(self.registered_components),
            "last_sync": self.last_sync,
            "timestamp": datetime.now(),
        }

        return state

    def create_state_snapshot(self, symbols: list[str]) -> StateSnapshot:
        """Create a complete state snapshot"""
        try:
            # Collect latest ticks
            latest_ticks = {}
            for symbol in symbols:
                tick = self.redis_manager.get_latest_tick(symbol)
                if tick:
                    latest_ticks[symbol] = tick

            # Collect active opportunities
            active_opportunities = {}
            for symbol in symbols:
                opportunities = self.redis_manager.get_state(f"opportunity:{symbol}")
                if opportunities:
                    active_opportunities[symbol] = opportunities

            # Get market summary
            market_summary = self.redis_manager.get_market_summary() or {}

            snapshot = StateSnapshot(
                timestamp=datetime.now(),
                symbols_monitored=symbols,
                latest_ticks=latest_ticks,
                active_opportunities=active_opportunities,
                watchlists={},  # Would be populated from watchlist manager
                market_statistics=market_summary,
                system_metrics={
                    "registered_components": list(self.registered_components),
                    "last_sync": self.last_sync,
                },
            )

            return snapshot

        except Exception as e:
            logger.error(f"Error creating state snapshot: {str(e)}")
            return StateSnapshot(
                timestamp=datetime.now(),
                symbols_monitored=[],
                latest_ticks={},
                active_opportunities={},
                watchlists={},
                market_statistics={},
                system_metrics={},
            )

    def save_snapshot(self, snapshot: StateSnapshot, filepath: str):
        """Save state snapshot to file"""
        try:
            with open(filepath, "wb") as f:
                pickle.dump(snapshot, f)
            logger.info(f"Saved state snapshot to {filepath}")

        except Exception as e:
            logger.error(f"Error saving snapshot: {str(e)}")

    def load_snapshot(self, filepath: str) -> StateSnapshot | None:
        """Load state snapshot from file"""
        try:
            with open(filepath, "rb") as f:
                snapshot = pickle.load(f)
            logger.info(f"Loaded state snapshot from {filepath}")
            return snapshot

        except Exception as e:
            logger.error(f"Error loading snapshot: {str(e)}")
            return None

    async def sync_state(self):
        """Synchronize state between Redis and database"""
        try:
            # This could include operations like:
            # - Backing up Redis data to database
            # - Cleaning up expired entries
            # - Validating data consistency

            await self.db_manager.cleanup_old_data()
            self.last_sync = datetime.now()

            # Trigger sync callbacks
            self._trigger_event_callbacks("state_synced", {"timestamp": self.last_sync})

            logger.debug("State synchronization completed")

        except Exception as e:
            logger.error(f"Error during state sync: {str(e)}")

    def add_event_callback(self, event_type: str, callback: callable):
        """Add event callback"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)

    def remove_event_callback(self, event_type: str, callback: callable):
        """Remove event callback"""
        if (
            event_type in self.event_callbacks
            and callback in self.event_callbacks[event_type]
        ):
            self.event_callbacks[event_type].remove(callback)

    def _trigger_event_callbacks(self, event_type: str, data: Any):
        """Trigger event callbacks"""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(data))
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in event callback for {event_type}: {str(e)}")

    async def start_background_sync(self):
        """Start background state synchronization"""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                await self.sync_state()
            except Exception as e:
                logger.error(f"Error in background sync: {str(e)}")
                await asyncio.sleep(10)  # Wait before retrying
