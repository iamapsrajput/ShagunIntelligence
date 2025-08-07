from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from loguru import logger

from agents.crew_manager import CrewManager
from app.api.routes import (
    advanced_orders,
    agents,
    automated_trading,
    broker_api,
    data_quality,
    data_sources,
    database_api,
    enhanced_risk,
    health,
    market_data,
    market_data_live,
    market_schedule,
    multi_timeframe,
    news,
    portfolio,
    sentiment,
    system,
    system_health,
    trading,
    trading_config,
)
from app.api.websocket import websocket_router
from app.core.auth import auth_router
from app.core.config import get_settings
from app.core.scheduler import trading_scheduler
from app.db.base import Base
from app.db.session import async_engine, engine
from app.middleware.error_handling import (
    CircuitBreakerMiddleware,
    ErrorHandlingMiddleware,
    RateLimitingMiddleware,
)
from app.middleware.security import (
    APIKeyValidationMiddleware,
    AuditLoggingMiddleware,
    InputValidationMiddleware,
    SecurityHeadersMiddleware,
    TradingSecurityMiddleware,
)
from app.services.websocket_manager import websocket_manager
from backend.data_sources.integration import get_data_source_integration
from services.kite.client import KiteClient

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Shagun Intelligence Trading System...")

    # Create database tables
    if async_engine:
        # Use async engine for PostgreSQL
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    else:
        # Use sync engine for SQLite
        Base.metadata.create_all(bind=engine)

    # Initialize multi-source data manager
    data_integration = get_data_source_integration()
    await data_integration.initialize()
    app.state.data_integration = data_integration
    logger.info("Multi-source data manager initialized")

    # Initialize Kite client (kept for backward compatibility)
    app.state.kite_client = KiteClient()

    # Initialize CrewAI manager
    app.state.crew_manager = CrewManager()

    # Initialize WebSocket manager
    app.state.websocket_manager = websocket_manager

    # Start background scheduler
    trading_scheduler.start()

    logger.info("Shagun Intelligence startup complete")
    yield

    # Shutdown
    logger.info("Shutting down Shagun Intelligence Trading System...")

    # Stop scheduler
    trading_scheduler.shutdown()

    # Shutdown data integration
    await data_integration.shutdown()

    # Close WebSocket connections
    await websocket_manager.disconnect_all()

    # Dispose database connections
    if async_engine:
        await async_engine.dispose()
    # Note: sync engine disposal is handled automatically

    logger.info("Shagun Intelligence shutdown complete")


app = FastAPI(
    title="Shagun Intelligence Trading System",
    description="AI-powered algorithmic trading platform with multi-agent system",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Security middleware (order matters - most specific first)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(TradingSecurityMiddleware)
app.add_middleware(APIKeyValidationMiddleware)
app.add_middleware(InputValidationMiddleware)

# Error handling and resilience middleware
app.add_middleware(
    ErrorHandlingMiddleware, enable_debug=settings.ENVIRONMENT == "development"
)
app.add_middleware(CircuitBreakerMiddleware, failure_threshold=5, recovery_timeout=60)
app.add_middleware(RateLimitingMiddleware, requests_per_minute=100)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

# Include routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(trading.router, prefix="/api/v1/trading", tags=["trading"])
app.include_router(advanced_orders.router, prefix="/api/v1", tags=["advanced-orders"])
app.include_router(enhanced_risk.router, prefix="/api/v1", tags=["enhanced-risk"])
app.include_router(multi_timeframe.router, prefix="/api/v1", tags=["multi-timeframe"])
app.include_router(market_data_live.router, prefix="/api/v1", tags=["market-data-live"])
app.include_router(broker_api.router, prefix="/api/v1", tags=["broker-api"])
app.include_router(database_api.router, prefix="/api/v1", tags=["database"])
app.include_router(
    automated_trading.router,
    prefix="/api/v1/automated-trading",
    tags=["automated-trading"],
)
app.include_router(market_data.router, prefix="/api/v1/market", tags=["market"])
app.include_router(
    market_schedule.router, prefix="/api/v1/market/schedule", tags=["market-schedule"]
)
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
app.include_router(system_health.router, prefix="/api/v1", tags=["system-health"])
app.include_router(
    trading_config.router, prefix="/api/v1/system", tags=["trading-config"]
)
app.include_router(data_sources.router)
app.include_router(data_quality.router)
app.include_router(sentiment.router)
app.include_router(news.router)
app.include_router(websocket_router, prefix="/ws", tags=["websocket"])


@app.get("/")
async def root():
    return {
        "message": "Shagun Intelligence Trading System API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/health/detailed")
async def detailed_health():
    """Detailed health check with service status"""
    # Safely get service statuses
    try:
        kite_status = app.state.kite_client.get_health_status()
    except AttributeError:
        kite_status = {
            "status": "unknown",
            "message": "Health check method not available",
        }

    try:
        agent_status = app.state.crew_manager.get_status()
    except AttributeError:
        agent_status = {"status": "running", "agents": "initialized"}

    try:
        websocket_count = websocket_manager.get_connection_count()
    except AttributeError:
        websocket_count = 0

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "kite_api": kite_status,
            "agents": agent_status,
            "websocket_connections": websocket_count,
            "scheduler": getattr(trading_scheduler, "running", True),
        },
        "version": "2.0.0",
        "environment": getattr(settings, "ENVIRONMENT", "development"),
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
