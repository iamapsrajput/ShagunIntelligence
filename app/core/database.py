"""
Enhanced Database Configuration with Connection Pooling and Error Handling
Production-ready database setup with monitoring and resilience
"""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Optional

from loguru import logger
from sqlalchemy import create_engine, event, pool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from app.core.config import get_settings

# Database models base
Base = declarative_base()

# Global database instances
async_engine: create_async_engine | None = None
sync_engine: create_engine | None = None
AsyncSessionLocal: async_sessionmaker | None = None
SessionLocal: sessionmaker | None = None


class DatabaseManager:
    """Enhanced database manager with connection pooling and monitoring"""

    def __init__(self):
        self.settings = get_settings()
        self.async_engine = None
        self.sync_engine = None
        self.async_session_factory = None
        self.sync_session_factory = None

        # Connection monitoring
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "last_connection_time": None,
            "average_connection_time": 0,
        }

    async def initialize(self):
        """Initialize database connections with enhanced configuration"""
        try:
            # Async engine configuration
            async_database_url = self.settings.DATABASE_URL.replace(
                "postgresql://", "postgresql+asyncpg://"
            )

            self.async_engine = create_async_engine(
                async_database_url,
                # Connection pool settings
                poolclass=pool.QueuePool,
                pool_size=20,  # Number of connections to maintain
                max_overflow=30,  # Additional connections when pool is full
                pool_timeout=30,  # Timeout when getting connection from pool
                pool_recycle=3600,  # Recycle connections after 1 hour
                pool_pre_ping=True,  # Validate connections before use
                # Performance settings
                echo=self.settings.ENVIRONMENT == "development",
                echo_pool=self.settings.ENVIRONMENT == "development",
                future=True,
                # Connection arguments
                connect_args={
                    "server_settings": {
                        "application_name": "shagunintelligence_trading",
                        "jit": "off",  # Disable JIT for better performance in some cases
                    },
                    "command_timeout": 60,
                    "prepared_statement_cache_size": 0,  # Disable prepared statement cache
                },
            )

            # Sync engine configuration (for migrations and admin tasks)
            self.sync_engine = create_engine(
                self.settings.DATABASE_URL,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=self.settings.ENVIRONMENT == "development",
            )

            # Session factories
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )

            self.sync_session_factory = sessionmaker(
                bind=self.sync_engine, autoflush=True, autocommit=False
            )

            # Set up event listeners for monitoring
            self._setup_event_listeners()

            # Test connections
            await self._test_connections()

            logger.info("Database connections initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _setup_event_listeners(self):
        """Set up SQLAlchemy event listeners for monitoring"""

        @event.listens_for(self.async_engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Handle new database connections"""
            self.connection_stats["total_connections"] += 1
            self.connection_stats["active_connections"] += 1
            self.connection_stats["last_connection_time"] = time.time()

            logger.debug(
                f"New database connection established. Active: {self.connection_stats['active_connections']}"
            )

        @event.listens_for(self.async_engine.sync_engine, "close")
        def on_close(dbapi_connection, connection_record):
            """Handle closed database connections"""
            self.connection_stats["active_connections"] = max(
                0, self.connection_stats["active_connections"] - 1
            )
            logger.debug(
                f"Database connection closed. Active: {self.connection_stats['active_connections']}"
            )

        @event.listens_for(self.async_engine.sync_engine, "close_detached")
        def on_close_detached(dbapi_connection):
            """Handle detached connection closures"""
            self.connection_stats["active_connections"] = max(
                0, self.connection_stats["active_connections"] - 1
            )

        @event.listens_for(self.async_engine.sync_engine, "invalid")
        def on_invalid(dbapi_connection, connection_record, exception):
            """Handle invalid connections"""
            self.connection_stats["failed_connections"] += 1
            logger.warning(f"Database connection invalidated: {exception}")

    async def _test_connections(self):
        """Test database connections"""
        try:
            # Test async connection
            async with self.async_engine.begin() as conn:
                result = await conn.execute("SELECT 1")
                await result.fetchone()

            # Test sync connection
            with self.sync_engine.connect() as conn:
                conn.execute("SELECT 1")

            logger.info("Database connection tests passed")

        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with proper error handling"""
        if not self.async_session_factory:
            raise RuntimeError("Database not initialized")

        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()

    def get_sync_session(self):
        """Get sync database session"""
        if not self.sync_session_factory:
            raise RuntimeError("Database not initialized")

        return self.sync_session_factory()

    async def health_check(self) -> dict:
        """Perform database health check"""
        try:
            start_time = time.time()

            # Test async connection
            async with self.async_engine.begin() as conn:
                result = await conn.execute("SELECT 1, NOW()")
                row = await result.fetchone()

            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Get pool status
            pool_status = {
                "size": self.async_engine.pool.size(),
                "checked_in": self.async_engine.pool.checkedin(),
                "checked_out": self.async_engine.pool.checkedout(),
                "overflow": self.async_engine.pool.overflow(),
                "invalid": self.async_engine.pool.invalid(),
            }

            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "pool_status": pool_status,
                "connection_stats": self.connection_stats,
                "database_time": str(row[1]) if row else None,
            }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection_stats": self.connection_stats,
            }

    async def close(self):
        """Close database connections"""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
                logger.info("Async database engine disposed")

            if self.sync_engine:
                self.sync_engine.dispose()
                logger.info("Sync database engine disposed")

        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

    def get_connection_stats(self) -> dict:
        """Get current connection statistics"""
        return self.connection_stats.copy()


# Global database manager instance
db_manager = DatabaseManager()


async def init_database():
    """Initialize database connections"""
    global async_engine, sync_engine, AsyncSessionLocal, SessionLocal

    await db_manager.initialize()

    # Set global variables for backward compatibility
    async_engine = db_manager.async_engine
    sync_engine = db_manager.sync_engine
    AsyncSessionLocal = db_manager.async_session_factory
    SessionLocal = db_manager.sync_session_factory


async def close_database():
    """Close database connections"""
    await db_manager.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session"""
    async with db_manager.get_async_session() as session:
        yield session


def get_sync_db():
    """Dependency for getting sync database session"""
    session = db_manager.get_sync_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def database_health_check() -> dict:
    """Get database health status"""
    return await db_manager.health_check()


def get_database_stats() -> dict:
    """Get database connection statistics"""
    return db_manager.get_connection_stats()
