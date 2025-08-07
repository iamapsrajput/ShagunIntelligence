"""Scheduled end-of-day analysis and next-day preparation"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from typing import Any

import numpy as np
import pandas as pd
import pytz
import schedule
from loguru import logger

from .data_processor import RealTimeDataProcessor
from .pattern_recognition import PatternRecognitionEngine
from .state_manager import SharedStateManager
from .statistical_analyzer import StatisticalAnalyzer
from .volume_analyzer import VolumeSignalGenerator
from .watchlist_manager import WatchlistManager


@dataclass
class DailyAnalysisReport:
    """Daily analysis report structure"""

    date: datetime
    symbols_analyzed: list[str]
    top_opportunities: list[dict[str, Any]]
    market_summary: dict[str, Any]
    performance_metrics: dict[str, Any]
    recommendations: list[str]
    next_day_watchlist: list[str]
    key_levels: dict[str, list[float]]
    sector_analysis: dict[str, Any]
    risk_assessment: dict[str, Any]


class MarketScheduler:
    """Handles scheduled market analysis tasks"""

    def __init__(
        self,
        data_processor: RealTimeDataProcessor,
        statistical_analyzer: StatisticalAnalyzer,
        pattern_engine: PatternRecognitionEngine,
        volume_analyzer: VolumeSignalGenerator,
        watchlist_manager: WatchlistManager,
        state_manager: SharedStateManager,
    ):
        self.data_processor = data_processor
        self.statistical_analyzer = statistical_analyzer
        self.pattern_engine = pattern_engine
        self.volume_analyzer = volume_analyzer
        self.watchlist_manager = watchlist_manager
        self.state_manager = state_manager

        # Timezone for Indian markets
        self.market_timezone = pytz.timezone("Asia/Kolkata")

        # Market hours (9:15 AM to 3:30 PM IST)
        self.market_open = dt_time(9, 15)
        self.market_close = dt_time(15, 30)

        # Scheduled tasks
        self.scheduled_tasks: dict[str, Callable] = {}
        self.running_tasks: dict[str, asyncio.Task] = {}

        # Analysis settings
        self.eod_analysis_time = dt_time(16, 0)  # 4:00 PM IST
        self.premarket_analysis_time = dt_time(8, 30)  # 8:30 AM IST
        self.intraday_scan_interval = 300  # 5 minutes during market hours

        # Callbacks for reports
        self.report_callbacks: list[Callable] = []

        self._setup_schedule()

    def _setup_schedule(self):
        """Set up the scheduling system"""
        try:
            # End-of-day analysis
            schedule.every().monday.at("16:00").do(self._schedule_eod_analysis)
            schedule.every().tuesday.at("16:00").do(self._schedule_eod_analysis)
            schedule.every().wednesday.at("16:00").do(self._schedule_eod_analysis)
            schedule.every().thursday.at("16:00").do(self._schedule_eod_analysis)
            schedule.every().friday.at("16:00").do(self._schedule_eod_analysis)

            # Pre-market analysis
            schedule.every().monday.at("08:30").do(self._schedule_premarket_analysis)
            schedule.every().tuesday.at("08:30").do(self._schedule_premarket_analysis)
            schedule.every().wednesday.at("08:30").do(self._schedule_premarket_analysis)
            schedule.every().thursday.at("08:30").do(self._schedule_premarket_analysis)
            schedule.every().friday.at("08:30").do(self._schedule_premarket_analysis)

            # Weekend analysis
            schedule.every().saturday.at("10:00").do(self._schedule_weekend_analysis)

            logger.info("Market scheduler initialized with daily tasks")

        except Exception as e:
            logger.error(f"Error setting up schedule: {str(e)}")

    def _schedule_eod_analysis(self):
        """Schedule end-of-day analysis"""
        task = asyncio.create_task(self.run_eod_analysis())
        self.running_tasks["eod_analysis"] = task

    def _schedule_premarket_analysis(self):
        """Schedule pre-market analysis"""
        task = asyncio.create_task(self.run_premarket_analysis())
        self.running_tasks["premarket_analysis"] = task

    def _schedule_weekend_analysis(self):
        """Schedule weekend analysis"""
        task = asyncio.create_task(self.run_weekend_analysis())
        self.running_tasks["weekend_analysis"] = task

    async def run_eod_analysis(self) -> DailyAnalysisReport:
        """Run comprehensive end-of-day analysis"""
        logger.info("Starting end-of-day analysis")

        try:
            analysis_date = datetime.now(self.market_timezone)

            # Get all monitored symbols
            symbols = await self._get_monitored_symbols()

            # Generate comprehensive analysis
            report = await self._generate_daily_report(symbols, analysis_date)

            # Store analysis results
            await self._store_eod_results(report)

            # Prepare next day watchlist
            await self._prepare_next_day_watchlist(report)

            # Send notifications
            await self._notify_eod_completion(report)

            logger.info(f"End-of-day analysis completed for {len(symbols)} symbols")
            return report

        except Exception as e:
            logger.error(f"Error in end-of-day analysis: {str(e)}")
            raise

    async def run_premarket_analysis(self) -> dict[str, Any]:
        """Run pre-market analysis"""
        logger.info("Starting pre-market analysis")

        try:
            # Check overnight news and events
            overnight_events = await self._check_overnight_events()

            # Analyze gap opportunities
            gap_analysis = await self._analyze_gaps()

            # Update watchlist priorities based on overnight developments
            await self._update_premarket_watchlist()

            # Generate pre-market report
            premarket_report = {
                "timestamp": datetime.now(),
                "overnight_events": overnight_events,
                "gap_analysis": gap_analysis,
                "updated_watchlist": await self._get_high_priority_symbols(),
                "market_sentiment": await self._assess_market_sentiment(),
            }

            # Store results
            self.state_manager.redis_manager.set_state(
                "premarket_analysis", premarket_report, ttl=3600
            )

            logger.info("Pre-market analysis completed")
            return premarket_report

        except Exception as e:
            logger.error(f"Error in pre-market analysis: {str(e)}")
            return {}

    async def run_weekend_analysis(self) -> dict[str, Any]:
        """Run weekend analysis and preparation"""
        logger.info("Starting weekend analysis")

        try:
            # Weekly performance review
            weekly_performance = await self._analyze_weekly_performance()

            # Sector rotation analysis
            sector_analysis = await self._analyze_sector_rotation()

            # Strategy optimization
            strategy_review = await self._review_strategies()

            # Prepare for next week
            next_week_preparation = await self._prepare_next_week()

            weekend_report = {
                "timestamp": datetime.now(),
                "weekly_performance": weekly_performance,
                "sector_analysis": sector_analysis,
                "strategy_review": strategy_review,
                "next_week_preparation": next_week_preparation,
            }

            # Store results
            self.state_manager.redis_manager.set_state(
                "weekend_analysis",
                weekend_report,
                ttl=172800,  # 48 hours
            )

            logger.info("Weekend analysis completed")
            return weekend_report

        except Exception as e:
            logger.error(f"Error in weekend analysis: {str(e)}")
            return {}

    async def _generate_daily_report(
        self, symbols: list[str], analysis_date: datetime
    ) -> DailyAnalysisReport:
        """Generate comprehensive daily analysis report"""
        try:
            # Analyze all symbols
            symbol_analyses = {}
            top_opportunities = []

            for symbol in symbols:
                try:
                    # Get comprehensive analysis
                    analysis = await self._analyze_symbol_comprehensive(symbol)
                    symbol_analyses[symbol] = analysis

                    # Extract opportunities
                    if analysis.get("opportunities"):
                        for opp in analysis["opportunities"]:
                            if opp.get("score", 0) > 70:  # High-quality opportunities
                                top_opportunities.append(opp)

                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {str(e)}")
                    continue

            # Sort opportunities by score
            top_opportunities.sort(key=lambda x: x.get("score", 0), reverse=True)
            top_opportunities = top_opportunities[:20]  # Top 20

            # Generate market summary
            market_summary = await self._generate_market_summary(symbol_analyses)

            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(
                symbol_analyses
            )

            # Generate recommendations
            recommendations = await self._generate_recommendations(
                symbol_analyses, market_summary
            )

            # Prepare next day watchlist
            next_day_watchlist = await self._select_next_day_symbols(symbol_analyses)

            # Extract key levels
            key_levels = await self._extract_key_levels(symbol_analyses)

            # Sector analysis
            sector_analysis = await self._analyze_sectors(symbol_analyses)

            # Risk assessment
            risk_assessment = await self._assess_market_risk(
                symbol_analyses, market_summary
            )

            return DailyAnalysisReport(
                date=analysis_date,
                symbols_analyzed=symbols,
                top_opportunities=top_opportunities,
                market_summary=market_summary,
                performance_metrics=performance_metrics,
                recommendations=recommendations,
                next_day_watchlist=next_day_watchlist,
                key_levels=key_levels,
                sector_analysis=sector_analysis,
                risk_assessment=risk_assessment,
            )

        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
            raise

    async def _analyze_symbol_comprehensive(self, symbol: str) -> dict[str, Any]:
        """Comprehensive analysis of a single symbol"""
        try:
            # Get historical data for analysis
            df_daily = self.data_processor.get_candle_data(symbol, "day", 100)
            self.data_processor.get_candle_data(symbol, "60minute", 50)

            if df_daily.empty:
                return {"symbol": symbol, "status": "no_data"}

            # Technical analysis
            signal = self.statistical_analyzer.analyze_symbol(symbol)

            # Pattern analysis
            pattern_analysis = self.pattern_engine.analyze_patterns(symbol)

            # Volume analysis
            volume_signals = self.volume_analyzer.generate_volume_signals(symbol)
            volume_anomalies = self.volume_analyzer.anomaly_detector.detect_anomalies(
                symbol
            )

            # Calculate additional metrics
            latest_price = df_daily["close"].iloc[-1]
            daily_change = (
                (latest_price - df_daily["close"].iloc[-2])
                / df_daily["close"].iloc[-2]
                * 100
            )

            # Volatility analysis
            volatility_20d = (
                df_daily["close"].pct_change().rolling(20).std() * np.sqrt(252) * 100
            )
            current_volatility = volatility_20d.iloc[-1]

            # Support and resistance levels
            support_resistance = self._calculate_support_resistance(df_daily)

            return {
                "symbol": symbol,
                "current_price": latest_price,
                "daily_change": daily_change,
                "volatility": current_volatility,
                "signal": signal.to_dict() if signal else None,
                "pattern_analysis": pattern_analysis,
                "volume_signals": [
                    s.to_dict() if hasattr(s, "to_dict") else s for s in volume_signals
                ],
                "volume_anomalies": [a.to_dict() for a in volume_anomalies],
                "support_resistance": support_resistance,
                "opportunities": await self._extract_opportunities(
                    symbol, signal, pattern_analysis
                ),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {str(e)}")
            return {"symbol": symbol, "error": str(e)}

    def _calculate_support_resistance(self, df: pd.DataFrame) -> dict[str, list[float]]:
        """Calculate support and resistance levels"""
        try:
            import numpy as np
            from scipy.signal import argrelextrema

            highs = df["high"].values
            lows = df["low"].values

            # Find peaks and troughs
            peak_indices = argrelextrema(highs, np.greater, order=5)[0]
            trough_indices = argrelextrema(lows, np.less, order=5)[0]

            # Get recent levels (last 3 months)
            recent_peaks = [highs[i] for i in peak_indices if i > len(highs) - 60]
            recent_troughs = [lows[i] for i in trough_indices if i > len(lows) - 60]

            # Cluster similar levels
            resistance_levels = self._cluster_levels(recent_peaks, tolerance=0.02)
            support_levels = self._cluster_levels(recent_troughs, tolerance=0.02)

            return {
                "resistance": sorted(resistance_levels, reverse=True)[:5],
                "support": sorted(support_levels)[:5],
            }

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return {"resistance": [], "support": []}

    def _cluster_levels(
        self, levels: list[float], tolerance: float = 0.02
    ) -> list[float]:
        """Cluster price levels that are close together"""
        if not levels:
            return []

        clusters = []
        sorted_levels = sorted(levels)

        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                # Finalize current cluster
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        # Add last cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))

        return clusters

    async def _get_monitored_symbols(self) -> list[str]:
        """Get list of symbols being monitored"""
        # This would typically come from the watchlist manager
        default_symbols = [
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "HINDUNILVR",
            "INFY",
            "ICICIBANK",
            "KOTAKBANK",
            "SBIN",
            "BHARTIARTL",
            "ITC",
            "ASIANPAINT",
            "MARUTI",
            "AXISBANK",
            "LT",
            "HCLTECH",
        ]

        # Get symbols from active watchlists
        watchlist_symbols = []
        for watchlist_name in self.watchlist_manager.watchlists:
            items = self.watchlist_manager.watchlists[watchlist_name]
            watchlist_symbols.extend([item.symbol for item in items])

        # Combine and deduplicate
        all_symbols = list(set(default_symbols + watchlist_symbols))
        return all_symbols

    async def _extract_opportunities(
        self, symbol: str, signal, pattern_analysis
    ) -> list[dict[str, Any]]:
        """Extract trading opportunities from analysis"""
        opportunities = []

        # From trading signal
        if signal and signal.signal_type in ["BUY", "SELL"]:
            opportunities.append(
                {
                    "type": "technical_signal",
                    "symbol": symbol,
                    "direction": signal.signal_type,
                    "score": signal.strength * 100,
                    "confidence": signal.confidence,
                    "reasoning": signal.reasoning,
                }
            )

        # From pattern analysis
        if pattern_analysis and "pattern_signals" in pattern_analysis:
            for pattern_signal in pattern_analysis["pattern_signals"]:
                if pattern_signal.get("strength", 0) > 0.6:
                    opportunities.append(
                        {
                            "type": "pattern",
                            "symbol": symbol,
                            "pattern": pattern_signal.get("pattern", ""),
                            "direction": pattern_signal.get("direction", ""),
                            "score": pattern_signal.get("strength", 0) * 100,
                            "timeframe": pattern_signal.get("timeframe", ""),
                        }
                    )

        return opportunities

    async def start_scheduler(self):
        """Start the background scheduler"""
        logger.info("Starting market scheduler")

        while True:
            try:
                schedule.run_pending()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler: {str(e)}")
                await asyncio.sleep(60)

    def add_report_callback(self, callback: Callable):
        """Add callback for analysis reports"""
        self.report_callbacks.append(callback)

    async def _notify_eod_completion(self, report: DailyAnalysisReport):
        """Notify completion of end-of-day analysis"""
        for callback in self.report_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(report)
                else:
                    callback(report)
            except Exception as e:
                logger.error(f"Error in report callback: {str(e)}")

    # Additional placeholder methods would be implemented based on specific requirements
    async def _check_overnight_events(self) -> dict[str, Any]:
        """Check for overnight market events"""
        # Placeholder for news/events checking
        return {"events": [], "sentiment": "neutral"}

    async def _analyze_gaps(self) -> dict[str, Any]:
        """Analyze gap opportunities"""
        # Placeholder for gap analysis
        return {"gaps_found": [], "significant_gaps": []}

    async def _assess_market_sentiment(self) -> str:
        """Assess overall market sentiment"""
        # Placeholder for sentiment analysis
        return "neutral"

    async def _generate_market_summary(self, analyses: dict) -> dict[str, Any]:
        """Generate market summary from symbol analyses"""
        return {"total_symbols": len(analyses), "bullish": 0, "bearish": 0}

    async def _calculate_performance_metrics(self, analyses: dict) -> dict[str, Any]:
        """Calculate performance metrics"""
        return {"accuracy": 0.75, "avg_return": 2.5}

    async def _generate_recommendations(
        self, analyses: dict, market_summary: dict
    ) -> list[str]:
        """Generate trading recommendations"""
        return ["Focus on high-volume breakouts", "Monitor support levels"]

    async def _select_next_day_symbols(self, analyses: dict) -> list[str]:
        """Select symbols for next day monitoring"""
        return list(analyses.keys())[:10]

    async def _extract_key_levels(self, analyses: dict) -> dict[str, list[float]]:
        """Extract key price levels"""
        return {}

    async def _analyze_sectors(self, analyses: dict) -> dict[str, Any]:
        """Analyze sector performance"""
        return {"technology": "bullish", "banking": "neutral"}

    async def _assess_market_risk(
        self, analyses: dict, market_summary: dict
    ) -> dict[str, Any]:
        """Assess overall market risk"""
        return {"risk_level": "medium", "factors": []}

    # Additional placeholder methods...
    async def _update_premarket_watchlist(self):
        pass

    async def _get_high_priority_symbols(self) -> list[str]:
        return []

    async def _analyze_weekly_performance(self) -> dict:
        return {}

    async def _analyze_sector_rotation(self) -> dict:
        return {}

    async def _review_strategies(self) -> dict:
        return {}

    async def _prepare_next_week(self) -> dict:
        return {}

    async def _store_eod_results(self, report):
        pass

    async def _prepare_next_day_watchlist(self, report):
        pass
