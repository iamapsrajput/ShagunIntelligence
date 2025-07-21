from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
from datetime import datetime

from app.api.routes import health, trading, market_data, agents, portfolio, system, data_sources, data_quality, sentiment, news
from app.api.websocket import websocket_router
from app.core.config import get_settings
from app.core.auth import auth_router
from app.db.session import engine, SessionLocal
from app.db.base import Base
from app.core.scheduler import trading_scheduler
from services.kite.client import KiteClient
from agents.crew_manager import CrewManager
from app.services.websocket_manager import websocket_manager
from backend.data_sources.integration import get_data_source_integration

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AlgoHive Trading System...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
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
    
    logger.info("AlgoHive startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down AlgoHive Trading System...")
    
    # Stop scheduler
    trading_scheduler.shutdown()
    
    # Shutdown data integration
    await data_integration.shutdown()
    
    # Close WebSocket connections
    await websocket_manager.disconnect_all()
    
    # Dispose database connections
    await engine.dispose()
    
    logger.info("AlgoHive shutdown complete")

app = FastAPI(
    title="AlgoHive Trading System",
    description="AI-powered algorithmic trading platform with multi-agent system",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Include routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(trading.router, prefix="/api/v1/trading", tags=["trading"])
app.include_router(market_data.router, prefix="/api/v1/market", tags=["market"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
app.include_router(data_sources.router)
app.include_router(data_quality.router)
app.include_router(sentiment.router)
app.include_router(news.router)
app.include_router(websocket_router, prefix="/ws", tags=["websocket"])

@app.get("/")
async def root():
    return {
        "message": "AlgoHive Trading System API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/health/detailed")
async def detailed_health():
    """Detailed health check with service status"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "kite_api": app.state.kite_client.is_connected(),
            "agents": app.state.crew_manager.get_status(),
            "websocket_connections": websocket_manager.get_connection_count(),
            "scheduler": trading_scheduler.running
        },
        "version": "2.0.0",
        "environment": settings.ENVIRONMENT
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )