from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from app.core.dependencies import get_current_user
from app.schemas.user import User
from backend.data_sources.integration import get_data_source_integration

router = APIRouter(
    prefix="/api/v1/sentiment",
    tags=["sentiment"]
)


@router.get("/score/{symbol}")
async def get_sentiment_score(
    symbol: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get fused sentiment score from all available sources.
    
    Returns sentiment data including:
    - Fused sentiment score (-1 to 1)
    - Sentiment label (bearish/neutral/bullish)
    - Confidence score
    - Agreement score between sources
    - Individual source results
    """
    try:
        integration = get_data_source_integration()
        result = await integration.get_sentiment_score(symbol)
        return result
        
    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/{symbol}")
async def get_sentiment_trends(
    symbol: str,
    hours: int = Query(24, ge=1, le=168, description="Number of hours to look back"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get sentiment trends over time.
    
    Shows how sentiment has evolved with hourly data points.
    """
    try:
        integration = get_data_source_integration()
        result = await integration.get_sentiment_trends(symbol, hours)
        return result
        
    except Exception as e:
        logger.error(f"Error getting sentiment trends for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{symbol}")
async def get_sentiment_alerts(
    symbol: str,
    threshold: float = Query(0.3, ge=0.1, le=1.0, description="Alert threshold"),
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get real-time sentiment alerts when threshold is exceeded.
    """
    try:
        integration = get_data_source_integration()
        alerts = await integration.get_sentiment_alerts(symbol, threshold)
        return alerts
        
    except Exception as e:
        logger.error(f"Error getting sentiment alerts for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/stats")
async def get_sentiment_sources_stats(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get statistics about all sentiment sources.
    
    Shows reliability scores, success rates, and weights for each source.
    """
    try:
        integration = get_data_source_integration()
        stats = integration.get_sentiment_sources_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting sentiment source stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/grok/{symbol}")
async def get_grok_analysis(
    symbol: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed Grok AI analysis for a symbol.
    
    Provides advanced sentiment analysis with:
    - Real-time X platform data
    - Key factors driving sentiment
    - Influential posts
    - Confidence levels
    - Trend direction
    """
    try:
        integration = get_data_source_integration()
        result = await integration.get_grok_analysis(symbol)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Grok analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/grok/batch")
async def get_batch_grok_analysis(
    symbols: List[str],
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get Grok AI analysis for multiple symbols.
    
    Optimized batch processing for cost efficiency.
    Maximum 10 symbols per request.
    """
    try:
        if len(symbols) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 symbols allowed per batch request"
            )
        
        integration = get_data_source_integration()
        result = await integration.get_batch_grok_analysis(symbols)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch Grok analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/grok/cost-stats")
async def get_grok_cost_stats(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get Grok API usage and cost statistics.
    
    Shows:
    - Daily budget and usage
    - Token consumption
    - Remaining budget
    - Request count
    """
    try:
        integration = get_data_source_integration()
        stats = integration.get_grok_cost_stats()
        
        if "error" in stats:
            raise HTTPException(status_code=400, detail=stats["error"])
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Grok cost stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track-symbols")
async def track_symbols_for_sentiment(
    symbols: List[str],
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Update symbols to track for real-time sentiment analysis.
    
    This updates streaming filters for Twitter and other real-time sources.
    """
    try:
        if len(symbols) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 symbols allowed for tracking"
            )
        
        integration = get_data_source_integration()
        integration.track_symbols_for_sentiment(symbols)
        
        return {
            "message": f"Now tracking {len(symbols)} symbols for sentiment",
            "symbols": ", ".join(symbols)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating tracked symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))