from typing import Any, Dict, List, Optional

from backend.data_sources.integration import get_data_source_integration
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from app.core.auth import get_current_user
from app.schemas.auth import UserResponse as User

router = APIRouter(prefix="/api/v1/news", tags=["news"])


@router.get("/")
async def get_news(
    symbols: Optional[str] = Query(None, description="Comma-separated stock symbols"),
    categories: Optional[str] = Query(None, description="Comma-separated categories"),
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    min_relevance: float = Query(0.3, ge=0.0, le=1.0, description="Minimum relevance score"),
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """
    Get aggregated financial news from multiple sources.

    Features:
    - Multi-source aggregation (Alpha Vantage, EODHD, Marketaux)
    - Automatic deduplication
    - Sentiment analysis
    - Relevance scoring
    - Market impact assessment
    """
    try:
        integration = get_data_source_integration()

        # Parse symbols
        symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None

        # Parse categories
        category_list = [c.strip() for c in categories.split(",")] if categories else None

        news = await integration.get_news(
            symbols=symbol_list, categories=category_list, hours=hours, min_relevance=min_relevance
        )

        return news

    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/breaking")
async def get_breaking_news(
    symbols: Optional[str] = Query(None, description="Comma-separated stock symbols"),
    minutes: int = Query(15, ge=5, le=60, description="Minutes to look back"),
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """
    Get breaking/recent news with high market impact.

    Focuses on market-moving news from the last few minutes.
    """
    try:
        integration = get_data_source_integration()

        # Parse symbols
        symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None

        breaking = await integration.get_breaking_news(symbols=symbol_list, minutes=minutes)

        return breaking

    except Exception as e:
        logger.error(f"Error fetching breaking news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/{symbol}")
async def get_news_sentiment_summary(
    symbol: str,
    hours: int = Query(24, ge=1, le=168, description="Hours to analyze"),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get news sentiment summary for a specific symbol.

    Provides:
    - Average sentiment score
    - Sentiment distribution
    - Top news categories
    - Overall market impact assessment
    """
    try:
        integration = get_data_source_integration()

        summary = await integration.get_news_sentiment_summary(symbol=symbol, hours=hours)

        return summary

    except Exception as e:
        logger.error(f"Error getting news sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitor")
async def monitor_news_symbols(symbols: List[str], current_user: User = Depends(get_current_user)) -> Dict[str, str]:
    """
    Start monitoring news for specific symbols.

    Enables real-time news alerts for market-moving events.
    """
    try:
        if len(symbols) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed for monitoring")

        integration = get_data_source_integration()
        await integration.monitor_news_for_symbols(symbols)

        return {"message": f"Now monitoring news for {len(symbols)} symbols", "symbols": ", ".join(symbols)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up news monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/stats")
async def get_news_sources_stats(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get statistics about news sources.

    Shows:
    - Active sources
    - Source health status
    - Reliability scores
    """
    try:
        integration = get_data_source_integration()
        stats = integration.get_news_sources_stats()
        return stats

    except Exception as e:
        logger.error(f"Error getting news source stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_news_categories(current_user: User = Depends(get_current_user)) -> List[str]:
    """
    Get available news categories.
    """
    from backend.data_sources.news.base import NewsCategory

    return [category.value for category in NewsCategory]


@router.get("/search")
async def search_news(
    query: str = Query(..., description="Search query"),
    symbols: Optional[str] = Query(None, description="Filter by symbols"),
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """
    Search news articles by query.

    Searches across titles, summaries, and content.
    """
    try:
        integration = get_data_source_integration()

        # For now, use general news fetch and filter
        # In production, this would use dedicated search endpoints
        symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None

        all_news = await integration.get_news(symbols=symbol_list, hours=hours)

        # Filter by query
        query_lower = query.lower()
        filtered_news = [
            article
            for article in all_news
            if query_lower in article.get("title", "").lower() or query_lower in article.get("summary", "").lower()
        ]

        return filtered_news[:50]  # Limit results

    except Exception as e:
        logger.error(f"Error searching news: {e}")
        raise HTTPException(status_code=500, detail=str(e))
