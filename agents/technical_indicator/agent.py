"""Technical Indicator Agent for calculating real-time indicators with quality-weighted data."""

from datetime import datetime
from typing import Any

import pandas as pd
from crewai import Agent
from loguru import logger

# Quality levels will be imported in model_post_init
from .indicator_calculator import IndicatorCalculator
from .rolling_data_manager import RollingDataManager
from .signal_generator import SignalGenerator
from .visualization_formatter import VisualizationFormatter


class TechnicalIndicatorAgent(Agent):
    """Agent for calculating technical indicators with data quality weighting."""

    def __init__(self):
        """Initialize the Technical Indicator Agent with quality awareness."""
        super().__init__(
            role="Quality-Aware Technical Analyst",
            goal="Calculate technical indicators using quality-weighted price data for accurate signals",
            backstory="""You are an advanced technical analyst who understands that indicator accuracy
            depends on data quality. You adjust your calculations and confidence based on:
            - Data source reliability and consensus
            - Price data freshness and completeness
            - Cross-source validation
            When data quality is high (>80%), you provide full technical analysis with all indicators.
            With medium quality (60-80%), you focus on robust indicators like moving averages.
            With low quality (<60%), you only provide basic trend analysis with warnings.""",
            verbose=True,
            allow_delegation=False,
        )

    def model_post_init(self, __context):
        """Initialize components after Pydantic model creation."""
        from backend.data_sources.data_quality_validator import (
            DataQualityValidator,
            QualityGrade,
        )
        from backend.data_sources.integration import get_data_source_integration

        # Initialize quality components
        object.__setattr__(self, "data_integration", get_data_source_integration())
        object.__setattr__(self, "quality_validator", DataQualityValidator())
        object.__setattr__(
            self,
            "quality_thresholds",
            {
                QualityGrade.EXCELLENT: 0.9,
                QualityGrade.GOOD: 0.75,
                QualityGrade.ACCEPTABLE: 0.6,
            },
        )

        object.__setattr__(self, "indicator_calculator", IndicatorCalculator())
        object.__setattr__(self, "signal_generator", SignalGenerator())
        object.__setattr__(self, "rolling_data_manager", RollingDataManager())
        object.__setattr__(self, "visualization_formatter", VisualizationFormatter())

        # Quality-specific settings
        object.__setattr__(
            self,
            "quality_indicator_sets",
            {
                QualityGrade.EXCELLENT: [
                    "RSI",
                    "MACD",
                    "BB",
                    "SMA",
                    "EMA",
                    "STOCH",
                    "ADX",
                ],
                QualityGrade.GOOD: [
                    "SMA",
                    "EMA",
                    "BB",
                    "VOLUME",
                ],  # More robust indicators
                QualityGrade.ACCEPTABLE: ["SMA", "TREND"],  # Basic only
                QualityGrade.POOR: [],  # No indicators
                QualityGrade.FAILED: [],  # No indicators
            },
        )

        logger.info("Quality-Aware Technical Indicator Agent initialized")

    async def analyze_symbol(
        self,
        symbol: str,
        data: pd.DataFrame,
        timeframe: str = "5min",
        indicators: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze a symbol with quality-weighted technical indicators.

        Args:
            symbol: Stock symbol to analyze
            data: Historical price data (OHLCV)
            timeframe: Time interval (1min, 5min, 15min)
            indicators: List of indicators to calculate

        Returns:
            Quality-aware analysis with adjusted confidence
        """
        try:
            # Get quality-weighted data
            quality_data, quality_metrics = await self._get_quality_weighted_ohlcv(
                symbol, data
            )

            # Update rolling data manager with quality-weighted data
            self.rolling_data_manager.update_data(symbol, timeframe, quality_data)

            # Get optimized data window
            calculation_data = self.rolling_data_manager.get_calculation_window(
                symbol, timeframe
            )

            # Determine which indicators to use based on quality
            quality_level = quality_metrics["quality_level"]
            if indicators is None:
                indicators = self.quality_indicator_sets[quality_level]
            else:
                # Filter requested indicators based on quality
                allowed_indicators = self.quality_indicator_sets[quality_level]
                indicators = [ind for ind in indicators if ind in allowed_indicators]

            # Calculate indicators with quality adjustments
            indicator_results = {}
            for indicator in indicators:
                result = await self._calculate_quality_adjusted_indicator(
                    indicator, calculation_data, timeframe, quality_metrics
                )
                if result:
                    indicator_results[indicator] = result

            # Generate signals with quality-adjusted confidence
            signals = self._generate_quality_aware_signals(
                calculation_data, indicator_results, quality_metrics
            )

            # Format for visualization
            visualization_data = self.visualization_formatter.format_data(
                calculation_data, indicator_results, signals
            )

            # Compile analysis results
            analysis = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "indicators": indicator_results,
                "signals": signals,
                "visualization_data": visualization_data,
                "data_points": len(calculation_data),
                "data_quality": quality_metrics,
                "confidence_adjustment": self._get_confidence_adjustment(quality_level),
                "warnings": self._generate_quality_warnings(quality_level),
            }

            logger.info(
                f"Quality-aware technical analysis completed for {symbol} "
                f"(quality: {quality_metrics['overall_score']:.1%})"
            )
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            raise

    async def _calculate_indicator(
        self, indicator_type: str, data: pd.DataFrame, timeframe: str
    ) -> dict[str, Any] | None:
        """
        Calculate a specific indicator.

        Args:
            indicator_type: Type of indicator (RSI, MACD, etc.)
            data: Price data for calculation
            timeframe: Time interval

        Returns:
            Indicator calculation results
        """
        try:
            if indicator_type == "RSI":
                return self.indicator_calculator.calculate_rsi(data)
            elif indicator_type == "MACD":
                return self.indicator_calculator.calculate_macd(data)
            elif indicator_type == "BB":
                return self.indicator_calculator.calculate_bollinger_bands(data)
            elif indicator_type == "SMA":
                return self.indicator_calculator.calculate_sma(data)
            elif indicator_type == "EMA":
                return self.indicator_calculator.calculate_ema(data)
            else:
                logger.warning(f"Unknown indicator type: {indicator_type}")
                return None

        except Exception as e:
            logger.error(f"Error calculating {indicator_type}: {str(e)}")
            return None

    async def get_real_time_signals(
        self, symbol: str, timeframe: str = "5min"
    ) -> dict[str, Any]:
        """
        Get real-time trading signals for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Time interval

        Returns:
            Real-time signals with confidence levels
        """
        try:
            # Get latest data
            data = self.rolling_data_manager.get_latest_data(symbol, timeframe)
            if data is None or len(data) < 50:
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                    "message": "Insufficient data for analysis",
                }

            # Quick indicator calculation for signals
            indicators = {}
            indicators["RSI"] = self.indicator_calculator.calculate_rsi(data, period=14)
            indicators["MACD"] = self.indicator_calculator.calculate_macd(data)

            # Generate signal
            signal = self.signal_generator.get_current_signal(data, indicators)

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                **signal,
            }

        except Exception as e:
            logger.error(f"Error getting real-time signals for {symbol}: {str(e)}")
            raise

    async def batch_analyze(
        self, symbols: list[str], timeframe: str = "5min"
    ) -> list[dict[str, Any]]:
        """
        Analyze multiple symbols in batch.

        Args:
            symbols: List of stock symbols
            timeframe: Time interval

        Returns:
            List of analysis results
        """
        results = []
        for symbol in symbols:
            try:
                # Get data for symbol (placeholder - would integrate with data service)
                data = self.rolling_data_manager.get_latest_data(symbol, timeframe)
                if data is not None and len(data) >= 50:
                    analysis = await self.analyze_symbol(symbol, data, timeframe)
                    results.append(analysis)
                else:
                    logger.warning(f"Insufficient data for {symbol}")
            except Exception as e:
                logger.error(f"Error in batch analysis for {symbol}: {str(e)}")

        return results

    async def _get_quality_weighted_ohlcv(
        self, symbol: str, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Get quality-weighted OHLCV data with multi-source validation."""
        try:
            # Get current quote for validation
            (
                quote_data,
                quality_score,
                quality_level,
            ) = await self.get_quality_weighted_data(symbol, "quote")

            # Get consensus data if available
            (
                consensus_data,
                consensus_confidence,
            ) = await self.get_multi_source_consensus(symbol)

            # Validate and adjust historical data
            adjusted_data = data.copy()

            if quote_data and len(data) > 0:
                # Check if latest historical price matches current quote
                latest_hist_price = data.iloc[-1]["close"]
                current_price = quote_data.get("current_price", latest_hist_price)

                # Calculate adjustment factor if significant difference
                price_diff_pct = (
                    abs(current_price - latest_hist_price) / latest_hist_price
                )

                if price_diff_pct > 0.02:  # More than 2% difference
                    logger.warning(
                        f"Price discrepancy detected for {symbol}: "
                        f"Historical: {latest_hist_price:.2f}, Current: {current_price:.2f}"
                    )

                    # Apply adjustment if high quality data
                    from backend.data_sources.data_quality_validator import QualityGrade

                    if quality_level in [
                        QualityGrade.EXCELLENT,
                        QualityGrade.GOOD,
                    ]:
                        adjustment_factor = current_price / latest_hist_price
                        adjusted_data[
                            ["open", "high", "low", "close"]
                        ] *= adjustment_factor

            quality_metrics = {
                "overall_score": quality_score,
                "quality_level": quality_level,
                "consensus_confidence": consensus_confidence,
                "data_source": (
                    quote_data.get("data_source", "unknown")
                    if quote_data
                    else "unknown"
                ),
                "has_multi_source": consensus_confidence > 0.5,
            }

            return adjusted_data, quality_metrics

        except Exception as e:
            logger.error(f"Error getting quality-weighted OHLCV: {e}")
            from backend.data_sources.data_quality_validator import QualityGrade

            return data, {
                "overall_score": 0.5,
                "quality_level": QualityGrade.ACCEPTABLE,
                "consensus_confidence": 0.0,
                "data_source": "unknown",
                "has_multi_source": False,
            }

    async def _calculate_quality_adjusted_indicator(
        self,
        indicator: str,
        data: pd.DataFrame,
        timeframe: str,
        quality_metrics: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Calculate indicator with quality-based adjustments."""
        try:
            # Calculate base indicator
            result = await self._calculate_indicator(indicator, data, timeframe)

            if not result:
                return None

            # Adjust parameters based on quality
            from backend.data_sources.data_quality_validator import QualityGrade

            quality_level = quality_metrics["quality_level"]

            if quality_level == QualityGrade.GOOD:
                # Use more conservative parameters
                if indicator == "RSI":
                    # Recalculate with longer period for stability
                    result = self.indicator_calculator.calculate_rsi(data, period=21)
                elif indicator == "BB":
                    # Wider bands for uncertainty
                    result = self.indicator_calculator.calculate_bollinger_bands(
                        data, period=20, std_dev=2.5
                    )
            elif quality_level == QualityGrade.ACCEPTABLE:
                # Only basic calculations
                if indicator not in ["SMA", "TREND"]:
                    return None

            # Add quality metadata
            if isinstance(result, dict):
                result["quality_adjusted"] = quality_level != QualityGrade.EXCELLENT
                result["confidence_multiplier"] = self._get_confidence_adjustment(
                    quality_level
                )

            return result

        except Exception as e:
            logger.error(
                f"Error calculating quality-adjusted indicator {indicator}: {e}"
            )
            return None

    def _generate_quality_aware_signals(
        self,
        data: pd.DataFrame,
        indicators: dict[str, Any],
        quality_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate trading signals with quality-based confidence adjustment."""
        # Get base signals
        base_signals = self.signal_generator.generate_signals(data, indicators)

        # Adjust confidence based on data quality
        quality_level = quality_metrics["quality_level"]
        confidence_multiplier = self._get_confidence_adjustment(quality_level)

        adjusted_signals = base_signals.copy()

        # Adjust signal confidence
        if "confidence" in adjusted_signals:
            adjusted_signals["confidence"] *= confidence_multiplier

        # Add quality warnings
        from backend.data_sources.data_quality_validator import QualityGrade

        if quality_level == QualityGrade.POOR:
            adjusted_signals["quality_warning"] = (
                "Low data quality - signals unreliable"
            )
            adjusted_signals["recommended_action"] = "WAIT"
        elif quality_level == QualityGrade.ACCEPTABLE:
            adjusted_signals["quality_warning"] = (
                "Medium data quality - use with caution"
            )

        # Add multi-source validation flag
        adjusted_signals["multi_source_validated"] = quality_metrics.get(
            "has_multi_source", False
        )

        return adjusted_signals

    def _get_confidence_adjustment(self, quality_level) -> float:
        """Get confidence adjustment multiplier based on quality."""
        from backend.data_sources.data_quality_validator import QualityGrade

        adjustments = {
            QualityGrade.EXCELLENT: 1.0,
            QualityGrade.GOOD: 0.8,
            QualityGrade.ACCEPTABLE: 0.6,
            QualityGrade.POOR: 0.3,
            QualityGrade.FAILED: 0.0,
        }
        return adjustments.get(quality_level, 0.5)

    def _generate_quality_warnings(self, quality_level) -> list[str]:
        """Generate warnings based on data quality level."""
        from backend.data_sources.data_quality_validator import QualityGrade

        warnings = []

        if quality_level == QualityGrade.FAILED:
            warnings.append("CRITICAL: Data quality too low for analysis")
            warnings.append("No technical indicators calculated")
        elif quality_level == QualityGrade.POOR:
            warnings.append("WARNING: Low data quality detected")
            warnings.append("Only basic trend analysis available")
            warnings.append("Trading not recommended")
        elif quality_level == QualityGrade.ACCEPTABLE:
            warnings.append("CAUTION: Medium data quality")
            warnings.append("Limited indicators available")
            warnings.append("Reduce position sizes")

        return warnings

    def get_indicator_status(self) -> dict[str, Any]:
        """Get the status of the technical indicator agent."""
        return {
            "agent": "QualityAwareTechnicalIndicatorAgent",
            "status": "active",
            "quality_indicator_sets": {
                level.value: indicators
                for level, indicators in self.quality_indicator_sets.items()
            },
            "supported_timeframes": ["1min", "5min", "15min"],
            "data_manager_status": self.rolling_data_manager.get_status(),
            "signal_generator_status": self.signal_generator.get_status(),
        }
