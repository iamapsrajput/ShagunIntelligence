from app.core.config import get_settings
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

settings = get_settings()

# Create async engine
if settings.DATABASE_URL.startswith("sqlite"):
    # For SQLite, use sync engine since SQLite doesn't support async
    engine = None
else:
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        future=True,
        pool_pre_ping=True,
        poolclass=NullPool,  # Use NullPool for better connection management
    )

# Create async session factory
if settings.DATABASE_URL.startswith("sqlite"):
    AsyncSessionLocal = None
else:
    AsyncSessionLocal = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


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


# For synchronous operations (if needed)
from sqlalchemy import create_engine

sync_engine = create_engine(
    settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://"),
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


def get_sync_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
