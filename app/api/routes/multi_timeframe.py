"""
Multi-Timeframe Analysis API Routes
Provides endpoints for advanced multi-timeframe technical analysis
"""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.auth import get_current_user
from app.services.multi_timeframe_analysis import MultiTimeFrameEngine, TimeFrame

router = APIRouter(prefix="/multi-timeframe", tags=["multi-timeframe"])

# Global instance - in production, this would be dependency injected
multi_timeframe_engine = MultiTimeFrameEngine()


class TimeFrameAnalysisRequest(BaseModel):
    """Request model for multi-timeframe analysis"""

    symbol: str = Field(..., description="Trading symbol")
    timeframes: list[str] | None = Field(
        None, description="List of timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)"
    )
    include_levels: bool = Field(
        default=True, description="Include support/resistance levels"
    )
    include_conditions: bool = Field(
        default=True, description="Include entry/exit conditions"
    )


class TimeFrameSignalResponse(BaseModel):
    """Response model for individual timeframe signal"""

    timeframe: str
    signal_type: str
    strength: int
    confidence: float
    trend_direction: str
    key_levels: dict[str, Any]
    indicators: dict[str, Any]
    reasoning: list[str]
    timestamp: datetime


class MultiTimeFrameAnalysisResponse(BaseModel):
    """Response model for multi-timeframe analysis"""

    symbol: str
    consensus_signal: str
    consensus_strength: float
    consensus_confidence: float
    trend_alignment: dict[str, Any]
    key_levels: dict[str, Any]
    risk_reward_ratio: float
    entry_conditions: list[str]
    exit_conditions: list[str]
    timeframe_signals: dict[str, TimeFrameSignalResponse]
    timestamp: datetime


@router.post("/analyze", response_model=MultiTimeFrameAnalysisResponse)
async def analyze_multi_timeframe(
    request: TimeFrameAnalysisRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Perform comprehensive multi-timeframe analysis

    Analyzes multiple timeframes to provide:
    - Cross-timeframe signal confirmation
    - Trend alignment analysis
    - Key support/resistance levels
    - Risk-reward ratio calculation
    - Entry/exit conditions
    """
    try:
        # Convert string timeframes to TimeFrame enums
        timeframes = None
        if request.timeframes:
            timeframe_map = {
                "1m": TimeFrame.M1,
                "5m": TimeFrame.M5,
                "15m": TimeFrame.M15,
                "30m": TimeFrame.M30,
                "1h": TimeFrame.H1,
                "4h": TimeFrame.H4,
                "1d": TimeFrame.D1,
                "1w": TimeFrame.W1,
            }

            timeframes = []
            for tf_str in request.timeframes:
                if tf_str in timeframe_map:
                    timeframes.append(timeframe_map[tf_str])
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid timeframe: {tf_str}. Valid options: {list(timeframe_map.keys())}",
                    )

        # Perform analysis
        analysis = await multi_timeframe_engine.analyze_symbol(
            request.symbol, timeframes
        )

        # Convert timeframe signals to response format
        timeframe_signals_response = {}
        for tf, signal in analysis.timeframe_signals.items():
            timeframe_signals_response[tf.value] = TimeFrameSignalResponse(
                timeframe=tf.value,
                signal_type=signal.signal_type,
                strength=signal.strength.value,
                confidence=signal.confidence,
                trend_direction=signal.trend_direction.value,
                key_levels=signal.key_levels if request.include_levels else {},
                indicators=signal.indicators,
                reasoning=signal.reasoning,
                timestamp=signal.timestamp,
            )

        return MultiTimeFrameAnalysisResponse(
            symbol=analysis.symbol,
            consensus_signal=analysis.consensus_signal,
            consensus_strength=analysis.consensus_strength,
            consensus_confidence=analysis.consensus_confidence,
            trend_alignment=analysis.trend_alignment,
            key_levels=analysis.key_levels if request.include_levels else {},
            risk_reward_ratio=analysis.risk_reward_ratio,
            entry_conditions=(
                analysis.entry_conditions if request.include_conditions else []
            ),
            exit_conditions=(
                analysis.exit_conditions if request.include_conditions else []
            ),
            timeframe_signals=timeframe_signals_response,
            timestamp=analysis.timestamp,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/analyze/{symbol}")
async def analyze_symbol_quick(
    symbol: str,
    timeframes: str | None = Query(
        None, description="Comma-separated timeframes (e.g., '15m,1h,4h,1d')"
    ),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Quick multi-timeframe analysis for a symbol

    Simplified endpoint for getting multi-timeframe analysis with default settings
    """
    try:
        # Parse timeframes if provided
        timeframe_list = None
        if timeframes:
            timeframe_map = {
                "1m": TimeFrame.M1,
                "5m": TimeFrame.M5,
                "15m": TimeFrame.M15,
                "30m": TimeFrame.M30,
                "1h": TimeFrame.H1,
                "4h": TimeFrame.H4,
                "1d": TimeFrame.D1,
                "1w": TimeFrame.W1,
            }

            timeframe_list = []
            for tf_str in timeframes.split(","):
                tf_str = tf_str.strip()
                if tf_str in timeframe_map:
                    timeframe_list.append(timeframe_map[tf_str])

        # Perform analysis
        analysis = await multi_timeframe_engine.analyze_symbol(symbol, timeframe_list)

        # Get formatted summary
        summary = multi_timeframe_engine.get_analysis_summary(analysis)

        return {"success": True, "data": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/timeframes")
async def get_supported_timeframes():
    """Get list of supported timeframes"""
    return {
        "success": True,
        "data": {
            "timeframes": [
                {
                    "code": "1m",
                    "name": "1 Minute",
                    "description": "Ultra-short term scalping",
                },
                {
                    "code": "5m",
                    "name": "5 Minutes",
                    "description": "Short-term scalping",
                },
                {
                    "code": "15m",
                    "name": "15 Minutes",
                    "description": "Short-term trading",
                },
                {
                    "code": "30m",
                    "name": "30 Minutes",
                    "description": "Intraday trading",
                },
                {
                    "code": "1h",
                    "name": "1 Hour",
                    "description": "Intraday to short swing",
                },
                {"code": "4h", "name": "4 Hours", "description": "Swing trading"},
                {"code": "1d", "name": "1 Day", "description": "Position trading"},
                {"code": "1w", "name": "1 Week", "description": "Long-term trends"},
            ],
            "default_combination": ["15m", "1h", "4h", "1d"],
            "recommended_combinations": {
                "scalping": ["1m", "5m", "15m"],
                "day_trading": ["5m", "15m", "1h"],
                "swing_trading": ["1h", "4h", "1d"],
                "position_trading": ["4h", "1d", "1w"],
            },
        },
    }


@router.get("/indicators/{symbol}")
async def get_adaptive_indicators(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe for indicator calculation"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Get adaptive technical indicators for a specific timeframe

    Returns detailed indicator values with adaptive parameters based on market conditions
    """
    try:
        # Validate timeframe
        timeframe_map = {
            "1m": TimeFrame.M1,
            "5m": TimeFrame.M5,
            "15m": TimeFrame.M15,
            "30m": TimeFrame.M30,
            "1h": TimeFrame.H1,
            "4h": TimeFrame.H4,
            "1d": TimeFrame.D1,
            "1w": TimeFrame.W1,
        }

        if timeframe not in timeframe_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timeframe: {timeframe}. Valid options: {list(timeframe_map.keys())}",
            )

        tf_enum = timeframe_map[timeframe]

        # Get market data
        data = await multi_timeframe_engine._get_market_data(symbol, tf_enum)
        if data is None or len(data) < 50:
            raise HTTPException(status_code=404, detail="Insufficient market data")

        # Calculate adaptive indicators
        calculator = multi_timeframe_engine.indicator_calculator

        rsi = calculator.calculate_adaptive_rsi(data, tf_enum)
        macd = calculator.calculate_adaptive_macd(data, tf_enum)
        bb = calculator.calculate_adaptive_bollinger_bands(data, tf_enum)
        levels = calculator.calculate_support_resistance(data, tf_enum)

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicators": {
                    "rsi": rsi,
                    "macd": macd,
                    "bollinger_bands": bb,
                    "support_resistance": levels,
                },
                "current_price": data["close"].iloc[-1],
                "timestamp": datetime.now(),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Indicator calculation failed: {str(e)}"
        )


@router.get("/trend-alignment/{symbol}")
async def get_trend_alignment(
    symbol: str,
    timeframes: str | None = Query(
        "15m,1h,4h,1d", description="Comma-separated timeframes"
    ),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Get trend alignment analysis across multiple timeframes

    Shows how trends align across different timeframes and identifies conflicts
    """
    try:
        # Parse timeframes
        timeframe_map = {
            "1m": TimeFrame.M1,
            "5m": TimeFrame.M5,
            "15m": TimeFrame.M15,
            "30m": TimeFrame.M30,
            "1h": TimeFrame.H1,
            "4h": TimeFrame.H4,
            "1d": TimeFrame.D1,
            "1w": TimeFrame.W1,
        }

        timeframe_list = []
        for tf_str in timeframes.split(","):
            tf_str = tf_str.strip()
            if tf_str in timeframe_map:
                timeframe_list.append(timeframe_map[tf_str])

        if not timeframe_list:
            timeframe_list = [TimeFrame.M15, TimeFrame.H1, TimeFrame.H4, TimeFrame.D1]

        # Perform analysis
        analysis = await multi_timeframe_engine.analyze_symbol(symbol, timeframe_list)

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "trend_alignment": analysis.trend_alignment,
                "timeframe_trends": {
                    tf.value: {
                        "trend": signal.trend_direction.value,
                        "signal": signal.signal_type,
                        "confidence": signal.confidence,
                    }
                    for tf, signal in analysis.timeframe_signals.items()
                },
                "consensus": {
                    "signal": analysis.consensus_signal,
                    "confidence": analysis.consensus_confidence,
                },
                "timestamp": analysis.timestamp,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Trend alignment analysis failed: {str(e)}"
        )


@router.get("/levels/{symbol}")
async def get_key_levels(
    symbol: str,
    timeframes: str | None = Query(
        "1h,4h,1d", description="Comma-separated timeframes"
    ),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Get key support and resistance levels across multiple timeframes

    Identifies the most important price levels based on multi-timeframe analysis
    """
    try:
        # Parse timeframes
        timeframe_map = {
            "1m": TimeFrame.M1,
            "5m": TimeFrame.M5,
            "15m": TimeFrame.M15,
            "30m": TimeFrame.M30,
            "1h": TimeFrame.H1,
            "4h": TimeFrame.H4,
            "1d": TimeFrame.D1,
            "1w": TimeFrame.W1,
        }

        timeframe_list = []
        for tf_str in timeframes.split(","):
            tf_str = tf_str.strip()
            if tf_str in timeframe_map:
                timeframe_list.append(timeframe_map[tf_str])

        if not timeframe_list:
            timeframe_list = [TimeFrame.H1, TimeFrame.H4, TimeFrame.D1]

        # Perform analysis
        analysis = await multi_timeframe_engine.analyze_symbol(symbol, timeframe_list)

        # Get current price for context
        current_price = 2500  # Would get from actual data

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "current_price": current_price,
                "key_levels": analysis.key_levels,
                "timeframe_levels": {
                    tf.value: signal.key_levels
                    for tf, signal in analysis.timeframe_signals.items()
                },
                "risk_reward_ratio": analysis.risk_reward_ratio,
                "timestamp": analysis.timestamp,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Key levels analysis failed: {str(e)}"
        )


@router.get("/signals/summary")
async def get_signals_summary(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    timeframe: str = Query("1h", description="Primary timeframe for summary"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Get multi-timeframe signals summary for multiple symbols

    Provides a quick overview of signals across multiple symbols
    """
    try:
        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        # Validate timeframe
        timeframe_map = {
            "1m": TimeFrame.M1,
            "5m": TimeFrame.M5,
            "15m": TimeFrame.M15,
            "30m": TimeFrame.M30,
            "1h": TimeFrame.H1,
            "4h": TimeFrame.H4,
            "1d": TimeFrame.D1,
            "1w": TimeFrame.W1,
        }

        if timeframe not in timeframe_map:
            raise HTTPException(
                status_code=400, detail=f"Invalid timeframe: {timeframe}"
            )

        tf_enum = timeframe_map[timeframe]

        # Analyze each symbol
        results = {}
        for symbol in symbol_list:
            try:
                analysis = await multi_timeframe_engine.analyze_symbol(
                    symbol, [tf_enum]
                )

                results[symbol] = {
                    "consensus_signal": analysis.consensus_signal,
                    "consensus_confidence": analysis.consensus_confidence,
                    "trend_direction": analysis.trend_alignment.get(
                        "trend_direction", "NEUTRAL"
                    ),
                    "risk_reward_ratio": analysis.risk_reward_ratio,
                    "timestamp": analysis.timestamp,
                }
            except Exception as e:
                results[symbol] = {"error": str(e), "timestamp": datetime.now()}

        return {
            "success": True,
            "data": {
                "timeframe": timeframe,
                "symbols": results,
                "summary": {
                    "total_symbols": len(symbol_list),
                    "buy_signals": len(
                        [
                            r
                            for r in results.values()
                            if r.get("consensus_signal") == "BUY"
                        ]
                    ),
                    "sell_signals": len(
                        [
                            r
                            for r in results.values()
                            if r.get("consensus_signal") == "SELL"
                        ]
                    ),
                    "hold_signals": len(
                        [
                            r
                            for r in results.values()
                            if r.get("consensus_signal") == "HOLD"
                        ]
                    ),
                    "errors": len([r for r in results.values() if "error" in r]),
                },
                "timestamp": datetime.now(),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signals summary failed: {str(e)}")
