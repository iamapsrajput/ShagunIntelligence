from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger

from app.api.routes import health, trading, market_data
from app.core.config import get_settings
from services.kite.client import KiteClient
from agents.crew_manager import CrewManager

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AI Trading Platform...")
    
    # Initialize Kite client
    app.state.kite_client = KiteClient()
    
    # Initialize CrewAI manager
    app.state.crew_manager = CrewManager()
    
    logger.info("Application startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down AI Trading Platform...")

app = FastAPI(
    title="AI Trading Platform",
    description="An AI-powered intraday trading platform using CrewAI and Zerodha Kite API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(trading.router, prefix="/api/v1/trading", tags=["trading"])
app.include_router(market_data.router, prefix="/api/v1/market", tags=["market"])

@app.get("/")
async def root():
    return {"message": "AI Trading Platform API", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )