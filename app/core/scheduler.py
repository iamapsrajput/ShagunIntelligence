import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, time
from loguru import logger
from typing import Optional

from app.services.websocket_manager import websocket_broadcaster


class TradingScheduler:
    """Manages scheduled tasks for the trading system"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
        self.running = False
        self.app_state = None
        
    def initialize(self, app_state):
        """Initialize scheduler with app state"""
        self.app_state = app_state
        self._setup_jobs()
        
    def _setup_jobs(self):
        """Setup all scheduled jobs"""
        
        # Market hours check (every minute during market hours)
        self.scheduler.add_job(
            self._check_market_hours,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour='9-15',
                minute='*',
                timezone='Asia/Kolkata'
            ),
            id='market_hours_check',
            name='Market Hours Check',
            replace_existing=True
        )
        
        # Pre-market analysis (8:30 AM IST)
        self.scheduler.add_job(
            self._pre_market_analysis,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour=8,
                minute=30,
                timezone='Asia/Kolkata'
            ),
            id='pre_market_analysis',
            name='Pre-Market Analysis',
            replace_existing=True
        )
        
        # Market open tasks (9:15 AM IST)
        self.scheduler.add_job(
            self._market_open_tasks,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour=9,
                minute=15,
                timezone='Asia/Kolkata'
            ),
            id='market_open',
            name='Market Open Tasks',
            replace_existing=True
        )
        
        # Regular position monitoring (every 5 minutes during market hours)
        self.scheduler.add_job(
            self._monitor_positions,
            trigger=IntervalTrigger(minutes=5),
            id='position_monitoring',
            name='Position Monitoring',
            replace_existing=True
        )
        
        # Risk check (every 15 minutes)
        self.scheduler.add_job(
            self._check_risk_limits,
            trigger=IntervalTrigger(minutes=15),
            id='risk_check',
            name='Risk Limit Check',
            replace_existing=True
        )
        
        # Market close tasks (3:30 PM IST)
        self.scheduler.add_job(
            self._market_close_tasks,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour=15,
                minute=30,
                timezone='Asia/Kolkata'
            ),
            id='market_close',
            name='Market Close Tasks',
            replace_existing=True
        )
        
        # End of day report (4:00 PM IST)
        self.scheduler.add_job(
            self._generate_daily_report,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour=16,
                minute=0,
                timezone='Asia/Kolkata'
            ),
            id='daily_report',
            name='Daily Report Generation',
            replace_existing=True
        )
        
        # Portfolio rebalancing check (weekly - Sunday 10 PM)
        self.scheduler.add_job(
            self._check_portfolio_rebalancing,
            trigger=CronTrigger(
                day_of_week='sun',
                hour=22,
                minute=0,
                timezone='Asia/Kolkata'
            ),
            id='rebalancing_check',
            name='Portfolio Rebalancing Check',
            replace_existing=True
        )
        
        # System health check (every 10 minutes)
        self.scheduler.add_job(
            self._system_health_check,
            trigger=IntervalTrigger(minutes=10),
            id='health_check',
            name='System Health Check',
            replace_existing=True
        )
        
    async def _check_market_hours(self):
        """Check if market is open and update status"""
        try:
            now = datetime.now()
            market_open = time(9, 15)
            market_close = time(15, 30)
            
            is_market_open = (
                now.weekday() < 5 and  # Monday to Friday
                market_open <= now.time() <= market_close
            )
            
            if hasattr(self.app_state, 'market_open') and self.app_state.market_open != is_market_open:
                self.app_state.market_open = is_market_open
                
                await websocket_broadcaster.broadcast_system_status({
                    "market_open": is_market_open,
                    "timestamp": now.isoformat()
                })
                
                logger.info(f"Market status changed: {'OPEN' if is_market_open else 'CLOSED'}")
                
        except Exception as e:
            logger.error(f"Error in market hours check: {str(e)}")
    
    async def _pre_market_analysis(self):
        """Run pre-market analysis"""
        try:
            if not self.app_state or not self.app_state.trading_enabled:
                return
                
            logger.info("Starting pre-market analysis...")
            
            crew_manager = self.app_state.crew_manager
            watchlist = self.app_state.watchlist or ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']
            
            # Run analysis for each symbol in watchlist
            for symbol in watchlist:
                try:
                    analysis = await crew_manager.analyze_trade_opportunity(symbol)
                    logger.info(f"Pre-market analysis for {symbol}: {analysis.get('recommendation', 'N/A')}")
                except Exception as e:
                    logger.error(f"Failed to analyze {symbol}: {str(e)}")
            
            logger.info("Pre-market analysis completed")
            
        except Exception as e:
            logger.error(f"Error in pre-market analysis: {str(e)}")
    
    async def _market_open_tasks(self):
        """Tasks to run at market open"""
        try:
            logger.info("Market opened - running startup tasks")
            
            # Update system status
            self.app_state.market_open = True
            self.app_state.daily_trades = 0
            
            # Subscribe to market data feeds
            kite_client = self.app_state.kite_client
            watchlist = self.app_state.watchlist or ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']
            
            for symbol in watchlist:
                try:
                    await kite_client.subscribe_ticker(f"NSE:{symbol}")
                except Exception as e:
                    logger.error(f"Failed to subscribe to {symbol}: {str(e)}")
            
            # Broadcast market open
            await websocket_broadcaster.broadcast_alert({
                "type": "market_open",
                "message": "Market is now open for trading",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in market open tasks: {str(e)}")
    
    async def _monitor_positions(self):
        """Monitor open positions and check for exit conditions"""
        try:
            if not self.app_state or not self.app_state.trading_enabled:
                return
            
            kite_client = self.app_state.kite_client
            crew_manager = self.app_state.crew_manager
            
            # Get current positions
            positions = await kite_client.get_positions()
            
            for position in positions.get("net", []):
                if position["quantity"] != 0:
                    # Check if position needs to be closed
                    exit_analysis = await crew_manager.analyze_exit_conditions(
                        symbol=position["tradingsymbol"],
                        entry_price=position["average_price"],
                        current_price=position.get("last_price", 0),
                        quantity=position["quantity"],
                        pnl=position.get("pnl", 0)
                    )
                    
                    if exit_analysis.get("should_exit", False):
                        logger.warning(f"Exit signal for {position['tradingsymbol']}: {exit_analysis.get('reason')}")
                        # In production, this would place an exit order
            
        except Exception as e:
            logger.error(f"Error in position monitoring: {str(e)}")
    
    async def _check_risk_limits(self):
        """Check if risk limits are being adhered to"""
        try:
            if not self.app_state:
                return
            
            kite_client = self.app_state.kite_client
            
            # Get portfolio and positions
            portfolio = await kite_client.get_portfolio()
            positions = await kite_client.get_positions()
            
            # Calculate current risk metrics
            total_value = portfolio.get("equity", 0)
            day_pnl = sum(pos.get("pnl", 0) for pos in positions.get("net", []))
            day_pnl_percent = (day_pnl / total_value * 100) if total_value > 0 else 0
            
            # Check daily loss limit
            max_daily_loss = getattr(self.app_state, "max_daily_loss", 5.0)
            if day_pnl_percent < -max_daily_loss:
                logger.critical(f"Daily loss limit breached: {day_pnl_percent:.2f}%")
                
                # Disable trading
                self.app_state.trading_enabled = False
                
                # Broadcast alert
                await websocket_broadcaster.broadcast_alert({
                    "type": "risk_limit_breach",
                    "severity": "critical",
                    "message": f"Daily loss limit breached: {day_pnl_percent:.2f}%",
                    "action": "Trading disabled",
                    "timestamp": datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error in risk limit check: {str(e)}")
    
    async def _market_close_tasks(self):
        """Tasks to run at market close"""
        try:
            logger.info("Market closed - running end of day tasks")
            
            # Update system status
            self.app_state.market_open = False
            
            # Unsubscribe from market data feeds
            kite_client = self.app_state.kite_client
            await kite_client.unsubscribe_all_tickers()
            
            # Broadcast market close
            await websocket_broadcaster.broadcast_alert({
                "type": "market_close",
                "message": "Market is now closed",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in market close tasks: {str(e)}")
    
    async def _generate_daily_report(self):
        """Generate daily trading report"""
        try:
            logger.info("Generating daily report...")
            
            kite_client = self.app_state.kite_client
            crew_manager = self.app_state.crew_manager
            
            # Get daily statistics
            positions = await kite_client.get_positions()
            orders = await kite_client.get_orders()
            
            # Calculate metrics
            total_trades = len(orders)
            successful_trades = len([o for o in orders if o.get("status") == "COMPLETE"])
            day_pnl = sum(pos.get("pnl", 0) for pos in positions.get("day", []))
            
            report = {
                "date": datetime.now().date().isoformat(),
                "total_trades": total_trades,
                "successful_trades": successful_trades,
                "day_pnl": day_pnl,
                "open_positions": len([p for p in positions.get("net", []) if p["quantity"] != 0]),
                "agent_performance": await crew_manager.get_daily_agent_performance()
            }
            
            logger.info(f"Daily report generated: {report}")
            
            # In production, this would save the report and send notifications
            
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
    
    async def _check_portfolio_rebalancing(self):
        """Check if portfolio needs rebalancing"""
        try:
            logger.info("Checking portfolio rebalancing requirements...")
            
            # This would implement portfolio rebalancing logic
            # For now, just log the check
            
        except Exception as e:
            logger.error(f"Error in rebalancing check: {str(e)}")
    
    async def _system_health_check(self):
        """Perform system health check"""
        try:
            # Check various system components
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "components": {}
            }
            
            # Check database connection
            health_status["components"]["database"] = "healthy"
            
            # Check Kite API connection
            if hasattr(self.app_state, "kite_client"):
                health_status["components"]["kite_api"] = (
                    "healthy" if self.app_state.kite_client.is_connected() else "unhealthy"
                )
            
            # Check agent status
            if hasattr(self.app_state, "crew_manager"):
                agents_status = await self.app_state.crew_manager.get_all_agents_status()
                health_status["components"]["agents"] = (
                    "healthy" if len(agents_status) > 0 else "unhealthy"
                )
            
            # Log any unhealthy components
            unhealthy = [k for k, v in health_status["components"].items() if v != "healthy"]
            if unhealthy:
                logger.warning(f"Unhealthy components: {unhealthy}")
            
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
    
    def start(self):
        """Start the scheduler"""
        if not self.running:
            self.scheduler.start()
            self.running = True
            logger.info("Trading scheduler started")
    
    def shutdown(self):
        """Shutdown the scheduler"""
        if self.running:
            self.scheduler.shutdown(wait=True)
            self.running = False
            logger.info("Trading scheduler stopped")
    
    def pause(self):
        """Pause all jobs"""
        self.scheduler.pause()
        logger.info("Trading scheduler paused")
    
    def resume(self):
        """Resume all jobs"""
        self.scheduler.resume()
        logger.info("Trading scheduler resumed")
    
    def get_jobs(self):
        """Get list of scheduled jobs"""
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            }
            for job in self.scheduler.get_jobs()
        ]


# Global scheduler instance
trading_scheduler = TradingScheduler()