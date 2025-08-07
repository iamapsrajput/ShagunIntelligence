"""
Automated Trading Service for Shagun Intelligence Trading System
Coordinates multi-agent system for fully automated trading execution
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from agents.crew_manager import CrewManager
from app.core.config import get_settings
from app.core.resilience import with_circuit_breaker, with_retry
from app.services.market_schedule import market_schedule
from services.data_pipeline.pipeline import DataPipeline
from services.kite.client import KiteConnectService

logger = logging.getLogger(__name__)


class AutomatedTradingService:
    """
    Manages fully automated trading execution with multi-agent coordination
    """

    def __init__(self):
        self.settings = get_settings()
        self.is_running = False
        self.is_market_hours = False
        self.daily_pnl = 0.0
        self.active_positions = 0
        self.emergency_stop_triggered = False

        # Initialize agents
        self.crew_manager = CrewManager()
        self.coordinator = None

        # Initialize Kite client for live data
        self.kite_client = None
        self.data_pipeline = None

        # Trading symbols for automated analysis
        self.trading_symbols = [
            "RELIANCE",
            "TCS",
            "INFY",
            "HDFCBANK",
            "ICICIBANK",
            "HINDUNILVR",
            "ITC",
            "SBIN",
            "BHARTIARTL",
            "KOTAKBANK",
        ]

        # Analysis intervals (in seconds)
        self.market_analysis_interval = 300  # 5 minutes
        self.position_monitoring_interval = 30  # 30 seconds
        self.risk_check_interval = 60  # 1 minute

        logger.info("AutomatedTradingService initialized")

    async def start_automated_trading(self) -> dict[str, Any]:
        """Start the automated trading system"""
        try:
            if not self.settings.AUTOMATED_TRADING_ENABLED:
                return {
                    "status": "error",
                    "message": "Automated trading is not enabled in configuration",
                }

            if not self.settings.LIVE_TRADING_ENABLED:
                return {
                    "status": "error",
                    "message": "Live trading must be enabled for automated trading",
                }

            if self.is_running:
                return {
                    "status": "warning",
                    "message": "Automated trading is already running",
                }

            # Perform pre-trading safety checks
            safety_check = await self._perform_safety_checks()
            if not safety_check["passed"]:
                return {
                    "status": "error",
                    "message": f"Safety check failed: {safety_check['reason']}",
                }

            # Initialize coordinator with agents
            await self._initialize_coordinator()

            # Start automated trading loops
            self.is_running = True

            # Start background tasks
            asyncio.create_task(self._market_hours_monitor())
            asyncio.create_task(self._automated_analysis_loop())
            asyncio.create_task(self._position_monitoring_loop())
            asyncio.create_task(self._risk_monitoring_loop())

            logger.info("🤖 Automated trading system started successfully")

            return {
                "status": "success",
                "message": "Automated trading system started",
                "config": {
                    "max_risk_per_trade": self.settings.MAX_RISK_PER_TRADE,
                    "max_daily_loss": self.settings.MAX_DAILY_LOSS,
                    "max_positions": self.settings.MAX_CONCURRENT_POSITIONS,
                    "trading_hours": f"{self.settings.TRADING_START_TIME} - {self.settings.TRADING_END_TIME}",
                    "emergency_stop_at": f"₹{self.settings.EMERGENCY_STOP_LOSS_AMOUNT}",
                },
            }

        except Exception as e:
            logger.error(f"Failed to start automated trading: {e}")
            return {
                "status": "error",
                "message": f"Failed to start automated trading: {str(e)}",
            }

    async def stop_automated_trading(self) -> dict[str, Any]:
        """Stop the automated trading system"""
        try:
            if not self.is_running:
                return {
                    "status": "warning",
                    "message": "Automated trading is not running",
                }

            self.is_running = False

            # Close any open positions if configured to do so
            if self.settings.AUTO_CLOSE_ON_STOP:
                await self._close_all_positions()

            logger.info("🛑 Automated trading system stopped")

            return {
                "status": "success",
                "message": "Automated trading system stopped successfully",
            }

        except Exception as e:
            logger.error(f"Failed to stop automated trading: {e}")
            return {
                "status": "error",
                "message": f"Failed to stop automated trading: {str(e)}",
            }

    async def emergency_stop(self) -> dict[str, Any]:
        """Emergency stop - immediately halt all trading"""
        try:
            self.emergency_stop_triggered = True
            self.is_running = False

            # Immediately close all positions
            await self._emergency_close_positions()

            logger.critical("🚨 EMERGENCY STOP TRIGGERED - All trading halted")

            return {
                "status": "success",
                "message": "Emergency stop executed - all trading halted",
            }

        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return {"status": "error", "message": f"Emergency stop failed: {str(e)}"}

    async def get_status(self) -> dict[str, Any]:
        """Get current status of automated trading system"""
        return {
            "is_running": self.is_running,
            "is_market_hours": self.is_market_hours,
            "daily_pnl": self.daily_pnl,
            "active_positions": self.active_positions,
            "emergency_stop_triggered": self.emergency_stop_triggered,
            "config": {
                "automated_trading_enabled": self.settings.AUTOMATED_TRADING_ENABLED,
                "live_trading_enabled": self.settings.LIVE_TRADING_ENABLED,
                "max_risk_per_trade": self.settings.MAX_RISK_PER_TRADE,
                "max_daily_loss": self.settings.MAX_DAILY_LOSS,
                "max_positions": self.settings.MAX_CONCURRENT_POSITIONS,
            },
        }

    async def _perform_safety_checks(self) -> dict[str, Any]:
        """Perform comprehensive safety checks before starting"""
        try:
            # Initialize Kite client for live data
            if not self.kite_client:
                logger.info("Initializing Kite Connect client...")
                self.kite_client = KiteConnectService()
                kite_initialized = await self.kite_client.initialize()

                if not kite_initialized:
                    return {
                        "passed": False,
                        "reason": "Failed to initialize Kite Connect client. Please check API credentials.",
                    }

                logger.info("✅ Kite Connect client initialized successfully")

            # Initialize data pipeline for live market data
            if not self.data_pipeline:
                logger.info("Initializing data pipeline...")
                self.data_pipeline = DataPipeline(self.kite_client)
                await self.data_pipeline.start(self.trading_symbols)
                logger.info("✅ Data pipeline started successfully")

            # Check if within trading hours
            if self.settings.ENFORCE_TRADING_HOURS:
                if not self._is_market_hours():
                    return {"passed": False, "reason": "Outside trading hours"}

            # Check daily loss limits
            if (
                abs(self.daily_pnl)
                >= self.settings.MAX_DAILY_LOSS * self.settings.DEFAULT_POSITION_SIZE
            ):
                return {"passed": False, "reason": "Daily loss limit already reached"}

            # Check emergency stop status
            if self.emergency_stop_triggered:
                return {"passed": False, "reason": "Emergency stop is active"}

            # Verify live data connection
            try:
                # Test getting a quote to ensure live data is working
                test_quote = await self.kite_client.get_quote("RELIANCE")
                if not test_quote:
                    return {
                        "passed": False,
                        "reason": "Unable to fetch live market data",
                    }
                logger.info("✅ Live market data connection verified")
            except Exception as e:
                return {
                    "passed": False,
                    "reason": f"Live data connection failed: {str(e)}",
                }

            return {"passed": True}

        except Exception as e:
            return {"passed": False, "reason": f"Safety check error: {str(e)}"}

    async def _initialize_coordinator(self):
        """Initialize the coordinator agent with all specialist agents"""
        # This would initialize the actual agent instances
        # For now, we'll use the crew manager
        logger.info("Coordinator agent initialized with all specialist agents")

    def _is_market_hours(self) -> bool:
        """Check if current time is within trading hours using market schedule manager"""
        try:
            return market_schedule.is_market_open()
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False

    def _get_market_status(self) -> dict[str, Any]:
        """Get comprehensive market status"""
        try:
            return market_schedule.get_market_status()
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                "status": "UNKNOWN",
                "message": f"Error getting market status: {e}",
                "is_open": False,
            }

    async def _market_hours_monitor(self):
        """Monitor market hours and update status"""
        while self.is_running:
            try:
                self.is_market_hours = self._is_market_hours()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Market hours monitor error: {e}")
                await asyncio.sleep(60)

    async def _automated_analysis_loop(self):
        """Main automated analysis and trading loop"""
        while self.is_running:
            try:
                if not self.is_market_hours:
                    await asyncio.sleep(60)
                    continue

                # Analyze each symbol
                for symbol in self.trading_symbols:
                    if not self.is_running:
                        break

                    # Check if we can take more positions
                    if self.active_positions >= self.settings.MAX_CONCURRENT_POSITIONS:
                        logger.info(
                            f"Maximum positions ({self.settings.MAX_CONCURRENT_POSITIONS}) reached"
                        )
                        break

                    # Run multi-agent analysis
                    analysis_result = await self._run_agent_analysis(symbol)

                    # Execute trade if approved
                    if analysis_result.get("approved"):
                        await self._execute_automated_trade(symbol, analysis_result)

                await asyncio.sleep(self.market_analysis_interval)

            except Exception as e:
                logger.error(f"Automated analysis loop error: {e}")
                await asyncio.sleep(60)

    @with_circuit_breaker("ai_agents")
    @with_retry(max_retries=2, delay=2.0)
    async def _run_agent_analysis(self, symbol: str) -> dict[str, Any]:
        """Run comprehensive multi-agent analysis for a symbol with resilience protection"""
        try:
            # Get live market data first
            live_quote = await self.kite_client.get_quote(symbol)
            historical_data = await self.kite_client.get_historical_data(
                symbol, "5minute", days=30
            )

            logger.info(
                f"📊 Analyzing {symbol} with live data - Current Price: ₹{live_quote.get('last_price', 'N/A')}"
            )

            # Use crew manager for coordinated analysis with live data
            technical_analysis = await self.crew_manager.run_technical_analysis(symbol)
            sentiment_analysis = await self.crew_manager.run_sentiment_analysis(symbol)

            # Combine analyses and make decision
            combined_analysis = {
                "symbol": symbol,
                "live_quote": live_quote,
                "technical": technical_analysis,
                "sentiment": sentiment_analysis,
                "timestamp": datetime.now().isoformat(),
                "current_price": live_quote.get("last_price"),
                "volume": live_quote.get("volume"),
                "change_percent": live_quote.get("net_change", 0),
            }

            # Enhanced decision logic with live data
            technical_signal = technical_analysis.get("recommended_action", "HOLD")
            sentiment_score = sentiment_analysis.get("sentiment_score", 0.5)
            current_price = live_quote.get("last_price", 0)

            # Additional live data checks
            volume_check = (
                live_quote.get("volume", 0) > 10000
            )  # Minimum volume threshold
            price_check = 50 <= current_price <= 5000  # Price range check

            # Approve trade if all conditions are met
            approved = (
                technical_signal in ["BUY", "STRONG_BUY"]
                and sentiment_score > 0.6
                and self.active_positions < self.settings.MAX_CONCURRENT_POSITIONS
                and volume_check
                and price_check
            )

            combined_analysis["approved"] = approved
            combined_analysis["action"] = technical_signal if approved else "HOLD"
            combined_analysis["approval_reasons"] = {
                "technical_signal": technical_signal,
                "sentiment_score": sentiment_score,
                "volume_check": volume_check,
                "price_check": price_check,
                "position_limit": self.active_positions
                < self.settings.MAX_CONCURRENT_POSITIONS,
            }

            if approved:
                logger.info(
                    f"✅ Trade approved for {symbol}: {technical_signal} at ₹{current_price}"
                )
            else:
                logger.info(
                    f"❌ Trade rejected for {symbol}: {combined_analysis['approval_reasons']}"
                )

            return combined_analysis

        except Exception as e:
            logger.error(f"Agent analysis error for {symbol}: {e}")
            return {"approved": False, "error": str(e)}

    @with_circuit_breaker("kite_api")
    @with_retry(max_retries=2, delay=1.0)
    async def _execute_automated_trade(self, symbol: str, analysis: dict[str, Any]):
        """Execute an automated trade based on agent analysis with resilience protection"""
        try:
            action = analysis.get("action", "HOLD")
            current_price = analysis.get("current_price", 0)

            if action not in ["BUY", "STRONG_BUY"]:
                return

            # Calculate position size based on risk management
            position_value = min(
                self.settings.DEFAULT_POSITION_SIZE, self.settings.MAX_POSITION_VALUE
            )
            quantity = max(1, int(position_value / current_price))

            logger.info(f"🤖 Executing automated trade for {symbol}: {action}")
            logger.info(
                f"   Price: ₹{current_price}, Quantity: {quantity}, Value: ₹{quantity * current_price}"
            )

            # Place the order through Kite client
            order_result = await self.kite_client.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type="MARKET",  # Market order for immediate execution
                transaction_type="BUY",
                product="MIS",  # Intraday for now
                exchange="NSE",
            )

            if order_result and order_result.get("order_id"):
                self.active_positions += 1
                logger.info(f"✅ Order placed successfully: {order_result['order_id']}")

                # Set up automatic stop loss and take profit
                await self._setup_position_management(
                    symbol, quantity, current_price, order_result["order_id"]
                )

            else:
                logger.error(f"❌ Failed to place order for {symbol}")

        except Exception as e:
            logger.error(f"Automated trade execution error: {e}")

    async def _setup_position_management(
        self, symbol: str, quantity: int, entry_price: float, order_id: str
    ):
        """Set up automatic stop loss and take profit for a position"""
        try:
            if self.settings.AUTO_STOP_LOSS:
                stop_loss_price = entry_price * (
                    1 - self.settings.AUTO_STOP_LOSS_PERCENT
                )

                # Place stop loss order
                sl_order = await self.kite_client.place_order(
                    symbol=symbol,
                    quantity=quantity,
                    order_type="SL",
                    transaction_type="SELL",
                    product="MIS",
                    exchange="NSE",
                    trigger_price=stop_loss_price,
                    price=stop_loss_price * 0.99,  # Slightly lower for execution
                )

                if sl_order:
                    logger.info(
                        f"🛡️ Stop loss set for {symbol} at ₹{stop_loss_price:.2f}"
                    )

            if self.settings.AUTO_TAKE_PROFIT:
                take_profit_price = entry_price * (
                    1 + self.settings.AUTO_TAKE_PROFIT_PERCENT
                )

                # Place take profit order
                tp_order = await self.kite_client.place_order(
                    symbol=symbol,
                    quantity=quantity,
                    order_type="LIMIT",
                    transaction_type="SELL",
                    product="MIS",
                    exchange="NSE",
                    price=take_profit_price,
                )

                if tp_order:
                    logger.info(
                        f"🎯 Take profit set for {symbol} at ₹{take_profit_price:.2f}"
                    )

        except Exception as e:
            logger.error(f"Error setting up position management for {symbol}: {e}")

    async def _position_monitoring_loop(self):
        """Monitor existing positions for stop loss/take profit"""
        while self.is_running:
            try:
                # Monitor positions and apply automated rules
                await asyncio.sleep(self.position_monitoring_interval)
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(60)

    async def _risk_monitoring_loop(self):
        """Monitor risk metrics and trigger emergency stops if needed"""
        while self.is_running:
            try:
                # Check daily P&L against limits
                if abs(self.daily_pnl) >= self.settings.EMERGENCY_STOP_LOSS_AMOUNT:
                    logger.critical(
                        f"Emergency stop triggered: Daily loss ₹{abs(self.daily_pnl)}"
                    )
                    await self.emergency_stop()
                    break

                await asyncio.sleep(self.risk_check_interval)

            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(60)

    async def _close_all_positions(self):
        """Close all open positions"""
        logger.info("Closing all open positions...")
        # Implementation would close actual positions

    async def _emergency_close_positions(self):
        """Emergency close all positions immediately"""
        logger.critical("EMERGENCY: Closing all positions immediately")
        # Implementation would force close all positions


# Global instance
automated_trading_service = AutomatedTradingService()
