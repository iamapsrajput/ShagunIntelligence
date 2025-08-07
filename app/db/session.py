from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

settings = get_settings()

# Create engines based on database type
if settings.DATABASE_URL.startswith("sqlite"):
    # For SQLite, use sync engine since SQLite doesn't support async
    engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_pre_ping=True,
    )
    async_engine = None
else:
    # For PostgreSQL, ensure we use the async driver
    database_url = settings.DATABASE_URL
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

    async_engine = create_async_engine(
        database_url,
        echo=settings.DEBUG,
        future=True,
        pool_pre_ping=True,
        poolclass=NullPool,  # Use NullPool for better connection management
    )

    # Also create sync engine for compatibility
    sync_database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
    engine = create_engine(
        sync_database_url,
        echo=settings.DEBUG,
        pool_pre_ping=True,
    )

# Create session factories
if settings.DATABASE_URL.startswith("sqlite"):
    # For SQLite, only sync sessions
    AsyncSessionLocal = None
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    # For PostgreSQL, both async and sync sessions
    AsyncSessionLocal = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Dependency to get DB session
async def get_db():
    if settings.DATABASE_URL.startswith("sqlite"):
        # For SQLite, use sync session
        db = SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    else:
        # For PostgreSQL, use async session
        async with AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


def get_sync_db():
    """Get synchronous database session for compatibility"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
