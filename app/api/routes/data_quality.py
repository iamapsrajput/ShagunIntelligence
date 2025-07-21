from fastapi import APIRouter, HTTPException, Query, Depends, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio
import json
from loguru import logger

from app.core.dependencies import get_current_user
from app.schemas.user import User
from backend.data_sources.integration import get_data_source_integration
from backend.data_sources.data_quality_validator import QualityGrade
from backend.data_sources.multi_source_manager import (
    MultiSourceDataManager, 
    DataQualityLevel,
    DataQualityMetrics
)
from backend.streaming.realtime_pipeline import RealTimeDataPipeline

router = APIRouter(
    prefix="/api/v1/data-quality",
    tags=["data-quality"]
)

# Initialize managers
data_manager = MultiSourceDataManager()
streaming_pipeline = RealTimeDataPipeline()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.quality_subscribers: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # Remove from all subscriptions
        for symbol in list(self.quality_subscribers.keys()):
            if websocket in self.quality_subscribers[symbol]:
                self.quality_subscribers[symbol].remove(websocket)
                if not self.quality_subscribers[symbol]:
                    del self.quality_subscribers[symbol]
    
    async def subscribe_to_quality(self, websocket: WebSocket, symbol: str):
        if symbol not in self.quality_subscribers:
            self.quality_subscribers[symbol] = []
        if websocket not in self.quality_subscribers[symbol]:
            self.quality_subscribers[symbol].append(websocket)
    
    async def broadcast_quality_update(self, symbol: str, data: dict):
        if symbol in self.quality_subscribers:
            disconnected = []
            for websocket in self.quality_subscribers[symbol]:
                try:
                    await websocket.send_json(data)
                except:
                    disconnected.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                self.disconnect(ws)

manager = ConnectionManager()


@router.get("/validate/{symbol}")
async def validate_symbol_data(
    symbol: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Validate data quality for a specific symbol across all sources.
    
    Returns quality metrics including:
    - Freshness score (0-1)
    - Accuracy score (0-1)
    - Completeness score (0-1)
    - Reliability score (0-1)
    - Overall score (0-1)
    - Grade (Excellent/Good/Acceptable/Poor/Failed)
    - Detected anomalies
    """
    try:
        data_integration = get_data_source_integration()
        
        # Get quote with validation
        quote, metrics = await data_integration.manager.validate_and_get_quote(symbol)
        
        return {
            "symbol": symbol,
            "quote": quote,
            "quality_metrics": metrics.to_dict(),
            "recommendation": _get_quality_recommendation(metrics.grade)
        }
        
    except Exception as e:
        logger.error(f"Error validating data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best-quality/{symbol}")
async def get_best_quality_quote(
    symbol: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get quote from the source with the best quality score.
    
    This endpoint compares data from all available sources and returns
    the one with the highest quality score.
    """
    try:
        data_integration = get_data_source_integration()
        
        # Get best quality quote
        quote = await data_integration.manager.get_quote_with_best_quality(symbol)
        
        return {
            "symbol": symbol,
            "quote": quote,
            "source": quote.get("source", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Error getting best quality quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def get_quality_report(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive data quality report.
    
    Includes:
    - Source reliability scores
    - Recent quality metrics
    - Active alerts and anomalies
    """
    try:
        data_integration = get_data_source_integration()
        report = data_integration.manager.get_quality_report()
        
        # Add summary statistics
        report["summary"] = _calculate_report_summary(report)
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating quality report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/source-reliability")
async def get_source_reliability(
    source_name: Optional[str] = Query(None, description="Filter by source name"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get reliability scores and metrics for data sources.
    
    Shows historical reliability, success rates, and quality grades.
    """
    try:
        data_integration = get_data_source_integration()
        report = data_integration.manager._quality_validator.get_source_reliability_report()
        
        if source_name:
            if source_name in report:
                return {source_name: report[source_name]}
            else:
                raise HTTPException(status_code=404, detail=f"Source {source_name} not found")
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting source reliability: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbol-trend/{symbol}")
async def get_symbol_quality_trend(
    symbol: str,
    hours: int = Query(24, ge=1, le=168, description="Number of hours to look back"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get quality trend for a specific symbol over time.
    
    Shows price and volume history along with quality metrics.
    """
    try:
        data_integration = get_data_source_integration()
        trend = data_integration.manager.get_symbol_quality_trend(symbol, hours)
        
        return trend
        
    except Exception as e:
        logger.error(f"Error getting quality trend for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_quality_alerts(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of alerts"),
    severity: Optional[QualityGrade] = Query(None, description="Filter by severity"),
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get recent data quality alerts.
    
    Returns anomalies and quality issues detected across all sources.
    """
    try:
        data_integration = get_data_source_integration()
        report = data_integration.manager.get_quality_report()
        
        alerts = report.get("alerts", [])
        
        # Filter by severity if requested
        if severity:
            # Get metrics for each alert to check grade
            filtered_alerts = []
            for alert in alerts:
                # This is a simplified approach - in production you'd store grade with alerts
                if severity == QualityGrade.FAILED:
                    filtered_alerts.append(alert)
            alerts = filtered_alerts
        
        # Sort by timestamp (most recent first) and limit
        alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return alerts[:limit]
        
    except Exception as e:
        logger.error(f"Error getting quality alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threshold")
async def update_quality_threshold(
    alert_threshold: float = Query(..., ge=0.0, le=1.0, description="Quality score threshold for alerts"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update the quality score threshold for triggering alerts.
    
    Lower values will trigger more alerts.
    """
    try:
        data_integration = get_data_source_integration()
        data_integration.manager._quality_validator.alert_threshold = alert_threshold
        
        return {
            "message": "Alert threshold updated successfully",
            "new_threshold": alert_threshold
        }
        
    except Exception as e:
        logger.error(f"Error updating quality threshold: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_quality_recommendation(grade: QualityGrade) -> str:
    """Get recommendation based on quality grade."""
    recommendations = {
        QualityGrade.EXCELLENT: "Data quality is excellent. Safe to use for trading decisions.",
        QualityGrade.GOOD: "Data quality is good. Suitable for most trading scenarios.",
        QualityGrade.ACCEPTABLE: "Data quality is acceptable but monitor for improvements.",
        QualityGrade.POOR: "Data quality is poor. Use with caution and cross-reference.",
        QualityGrade.FAILED: "Data quality check failed. Do not use for trading."
    }
    return recommendations.get(grade, "Unknown quality grade")


def _calculate_report_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics for the quality report."""
    source_reliability = report.get("source_reliability", {})
    
    if not source_reliability:
        return {
            "total_sources": 0,
            "healthy_sources": 0,
            "average_reliability": 0.0,
            "alert_count": len(report.get("alerts", []))
        }
    
    total_sources = len(source_reliability)
    healthy_sources = sum(
        1 for source in source_reliability.values()
        if source.get("grade") in ["Excellent", "Good"]
    )
    
    reliability_scores = [
        source.get("reliability_score", 0.0)
        for source in source_reliability.values()
    ]
    average_reliability = sum(reliability_scores) / len(reliability_scores) if reliability_scores else 0.0
    
    return {
        "total_sources": total_sources,
        "healthy_sources": healthy_sources,
        "average_reliability": round(average_reliability, 3),
        "alert_count": len(report.get("alerts", []))
    }


# New multi-source monitoring endpoints

@router.get("/metrics/{symbol}")
async def get_data_quality_metrics(
    symbol: str,
    lookback_minutes: int = Query(default=30, description="Minutes to look back for metrics"),
    current_user: User = Depends(get_current_user)
):
    """Get real-time data quality metrics for a symbol across all sources."""
    try:
        # Get current quality metrics from all sources
        quality_data = await data_manager.get_quality_metrics(symbol)
        
        # Get historical quality for trending
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        historical_quality = await data_manager.get_historical_quality(
            symbol, 
            start_time=cutoff_time
        )
        
        # Calculate aggregated metrics
        total_sources = len(quality_data["sources"])
        high_quality_sources = sum(
            1 for s in quality_data["sources"].values() 
            if s["quality_level"] == DataQualityLevel.HIGH
        )
        
        # Get recent failover events
        failover_events = await data_manager.get_failover_events(
            symbol,
            start_time=cutoff_time
        )
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "overall_quality": quality_data["overall_quality"],
            "overall_quality_score": quality_data["overall_quality_score"],
            "sources": quality_data["sources"],
            "aggregated_metrics": {
                "total_sources": total_sources,
                "high_quality_sources": high_quality_sources,
                "average_latency_ms": quality_data["average_latency"],
                "data_consistency_score": quality_data["consistency_score"],
                "failover_count": len(failover_events)
            },
            "quality_history": historical_quality,
            "recent_failovers": failover_events[-5:]  # Last 5 failovers
        }
        
    except Exception as e:
        logger.error(f"Error getting data quality metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/source-health-multi")
async def get_multi_source_health_status(
    current_user: User = Depends(get_current_user)
):
    """Get health status for all data sources with multi-source integration."""
    try:
        health_data = await data_manager.get_all_source_health()
        
        # Calculate summary statistics
        total_sources = len(health_data["sources"])
        healthy_sources = sum(
            1 for s in health_data["sources"].values() 
            if s["status"] == "healthy"
        )
        
        # Get API usage and costs
        api_costs = await data_manager.get_api_usage_summary()
        
        return {
            "timestamp": datetime.now(),
            "sources": health_data["sources"],
            "summary": {
                "total_sources": total_sources,
                "healthy_sources": healthy_sources,
                "degraded_sources": sum(
                    1 for s in health_data["sources"].values() 
                    if s["status"] == "degraded"
                ),
                "unhealthy_sources": sum(
                    1 for s in health_data["sources"].values() 
                    if s["status"] == "unhealthy"
                ),
                "overall_health_score": (healthy_sources / total_sources * 100) if total_sources > 0 else 0
            },
            "api_usage": api_costs
        }
        
    except Exception as e:
        logger.error(f"Error getting source health status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment-fusion/{symbol}")
async def get_sentiment_fusion(
    symbol: str,
    lookback_hours: int = Query(default=24, description="Hours to look back for sentiment"),
    current_user: User = Depends(get_current_user)
):
    """Get multi-source sentiment fusion analysis."""
    try:
        # Get sentiment from all available sources
        sentiment_data = await data_manager.get_fused_sentiment(symbol)
        
        # Get historical sentiment
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        historical_sentiment = await data_manager.get_historical_sentiment(
            symbol,
            start_time=cutoff_time
        )
        
        # Calculate sentiment consensus
        source_sentiments = sentiment_data["source_sentiments"]
        consensus = {
            "bullish": sum(1 for s in source_sentiments.values() if s["sentiment"] > 0.3),
            "neutral": sum(1 for s in source_sentiments.values() if -0.3 <= s["sentiment"] <= 0.3),
            "bearish": sum(1 for s in source_sentiments.values() if s["sentiment"] < -0.3)
        }
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "fused_sentiment": sentiment_data["fused_sentiment"],
            "confidence_score": sentiment_data["confidence_score"],
            "source_sentiments": source_sentiments,
            "consensus": consensus,
            "sentiment_history": historical_sentiment,
            "quality_weighted": sentiment_data["quality_weighted"],
            "metadata": {
                "sources_used": len(source_sentiments),
                "high_quality_sources": sum(
                    1 for s in source_sentiments.values() 
                    if s.get("quality_score", 0) > 0.8
                ),
                "divergence_score": sentiment_data.get("divergence_score", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting sentiment fusion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/failover-logs")
async def get_failover_logs(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(default=100, le=1000),
    current_user: User = Depends(get_current_user)
):
    """Get failover event logs."""
    try:
        if not start_time:
            start_time = datetime.now() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.now()
        
        events = await data_manager.get_failover_logs(
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return events
        
    except Exception as e:
        logger.error(f"Error getting failover logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/quality-monitor")
async def quality_monitor_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time quality monitoring."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            if data["type"] == "subscribe":
                symbol = data.get("symbol")
                if symbol:
                    await manager.subscribe_to_quality(websocket, symbol)
                    await websocket.send_json({
                        "type": "subscribed",
                        "symbol": symbol,
                        "message": f"Subscribed to quality updates for {symbol}"
                    })
                    
                    # Send initial quality data
                    quality_data = await data_manager.get_quality_metrics(symbol)
                    await websocket.send_json({
                        "type": "quality_update",
                        "symbol": symbol,
                        "data": quality_data
                    })
            
            elif data["type"] == "unsubscribe":
                symbol = data.get("symbol")
                if symbol and symbol in manager.quality_subscribers:
                    if websocket in manager.quality_subscribers[symbol]:
                        manager.quality_subscribers[symbol].remove(websocket)
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "symbol": symbol
                        })
            
            elif data["type"] == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
        await websocket.close()


# Background task to broadcast quality updates
async def broadcast_quality_updates():
    """Background task to send quality updates to subscribed clients."""
    while True:
        try:
            # Get all subscribed symbols
            symbols = list(manager.quality_subscribers.keys())
            
            for symbol in symbols:
                # Get latest quality metrics
                quality_data = await data_manager.get_quality_metrics(symbol)
                
                # Broadcast to subscribers
                await manager.broadcast_quality_update(symbol, {
                    "type": "quality_update",
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "data": quality_data
                })
            
            # Wait before next update
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in quality broadcast: {e}")
            await asyncio.sleep(10)


# Start background task on module load
asyncio.create_task(broadcast_quality_updates())


@router.get("/stream-health")
async def get_stream_health(
    current_user: User = Depends(get_current_user)
):
    """Get real-time streaming health status."""
    try:
        stream_status = streaming_pipeline.get_all_stream_status()
        
        return {
            "timestamp": datetime.now(),
            "streams": stream_status,
            "summary": {
                "total_streams": len(stream_status),
                "active_streams": sum(1 for s in stream_status.values() if s["status"] == "connected"),
                "disconnected_streams": sum(1 for s in stream_status.values() if s["status"] == "disconnected"),
                "error_streams": sum(1 for s in stream_status.values() if s["status"] == "error")
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting stream health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api-costs/summary")
async def get_api_cost_summary(
    period: str = Query(default="month", description="Period: day, week, month"),
    current_user: User = Depends(get_current_user)
):
    """Get API cost summary."""
    try:
        from app.core.api_rate_limiter import get_rate_limiter
        
        rate_limiter = await get_rate_limiter()
        cost_report = rate_limiter.get_cost_report()
        
        # Add period-specific calculations
        if period == "day":
            # Estimate daily costs
            for provider, data in cost_report["providers"].items():
                data["estimated_daily_cost"] = data["cost"] / datetime.now().day
        elif period == "week":
            # Estimate weekly costs
            for provider, data in cost_report["providers"].items():
                data["estimated_weekly_cost"] = data["cost"] / (datetime.now().day / 7)
        
        return cost_report
        
    except Exception as e:
        logger.error(f"Error getting API cost summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))