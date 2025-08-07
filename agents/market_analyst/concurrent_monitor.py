"""Multi-stock monitoring with concurrent processing"""

import asyncio
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from .data_processor import RealTimeDataProcessor
from .pattern_recognition import PatternRecognitionEngine
from .statistical_analyzer import StatisticalAnalyzer
from .volume_analyzer import VolumeSignalGenerator


@dataclass
class MonitoringTask:
    """Monitoring task for a symbol"""

    symbol: str
    task_type: str  # 'full_analysis', 'quick_scan', 'pattern_check', 'volume_check'
    priority: int  # 1-5, 5 being highest
    created_at: datetime
    timeout: int = 30  # seconds

    def __lt__(self, other):
        # For priority queue ordering (higher priority first)
        return self.priority > other.priority


@dataclass
class MonitoringResult:
    """Result from monitoring task"""

    symbol: str
    task_type: str
    success: bool
    data: Any
    processing_time: float
    timestamp: datetime
    error: str | None = None


class SymbolMonitor:
    """Monitors a single symbol with all analysis types"""

    def __init__(self, symbol: str, analyzers: dict[str, Any]):
        self.symbol = symbol
        self.analyzers = analyzers
        self.last_analysis: dict[str, datetime] = {}
        self.analysis_intervals = {
            "full_analysis": 300,  # 5 minutes
            "quick_scan": 60,  # 1 minute
            "pattern_check": 180,  # 3 minutes
            "volume_check": 30,  # 30 seconds
        }

    async def run_analysis(self, task_type: str) -> MonitoringResult:
        """Run specific analysis type"""
        start_time = time.time()

        try:
            if task_type == "full_analysis":
                result = await self._full_analysis()
            elif task_type == "quick_scan":
                result = await self._quick_scan()
            elif task_type == "pattern_check":
                result = await self._pattern_check()
            elif task_type == "volume_check":
                result = await self._volume_check()
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            processing_time = time.time() - start_time
            self.last_analysis[task_type] = datetime.now()

            return MonitoringResult(
                symbol=self.symbol,
                task_type=task_type,
                success=True,
                data=result,
                processing_time=processing_time,
                timestamp=datetime.now(),
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in {task_type} for {self.symbol}: {str(e)}")

            return MonitoringResult(
                symbol=self.symbol,
                task_type=task_type,
                success=False,
                data=None,
                processing_time=processing_time,
                timestamp=datetime.now(),
                error=str(e),
            )

    async def _full_analysis(self) -> dict[str, Any]:
        """Complete analysis of the symbol"""
        try:
            # Run all analyzers
            statistical_analysis = self.analyzers["statistical"].analyze_symbol(
                self.symbol
            )
            pattern_analysis = self.analyzers["pattern"].analyze_patterns(self.symbol)
            volume_signals = self.analyzers["volume"].generate_volume_signals(
                self.symbol
            )
            volume_anomalies = self.analyzers[
                "volume"
            ].anomaly_detector.detect_anomalies(self.symbol)

            return {
                "symbol": self.symbol,
                "statistical_analysis": (
                    statistical_analysis.to_dict() if statistical_analysis else None
                ),
                "pattern_analysis": pattern_analysis,
                "volume_signals": [
                    signal.to_dict() if hasattr(signal, "to_dict") else signal
                    for signal in volume_signals
                ],
                "volume_anomalies": [anomaly.to_dict() for anomaly in volume_anomalies],
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error in full analysis for {self.symbol}: {str(e)}")
            raise

    async def _quick_scan(self) -> dict[str, Any]:
        """Quick price and volume scan"""
        try:
            data_processor = self.analyzers["data_processor"]
            latest_tick = data_processor.get_latest_tick(self.symbol)

            if not latest_tick:
                return {"symbol": self.symbol, "status": "no_data"}

            # Get recent tick history for quick analysis
            recent_ticks = data_processor.get_tick_history(self.symbol, 10)

            if len(recent_ticks) < 2:
                return {"symbol": self.symbol, "status": "insufficient_data"}

            # Quick metrics
            prices = [tick.last_price for tick in recent_ticks]
            volumes = [tick.volume for tick in recent_ticks]

            price_change = (prices[-1] - prices[0]) / prices[0] * 100
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            current_volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1

            # Quick alerts
            alerts = []
            if abs(price_change) > 2.0:  # 2% price change
                alerts.append(f"Significant price change: {price_change:.2f}%")

            if current_volume_ratio > 2.0:  # 2x volume
                alerts.append(f"High volume: {current_volume_ratio:.1f}x average")

            return {
                "symbol": self.symbol,
                "current_price": latest_tick.last_price,
                "price_change": price_change,
                "volume_ratio": current_volume_ratio,
                "alerts": alerts,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error in quick scan for {self.symbol}: {str(e)}")
            raise

    async def _pattern_check(self) -> dict[str, Any]:
        """Check for pattern formations"""
        try:
            pattern_engine = self.analyzers["pattern"]
            pattern_analysis = pattern_engine.analyze_patterns(self.symbol, ["5min"])

            # Extract key pattern information
            active_patterns = []

            # Check breakouts
            if "breakouts" in pattern_analysis:
                for breakout in pattern_analysis["breakouts"]:
                    if breakout.get("strength", 0) > 2.0:  # Significant breakout
                        active_patterns.append(
                            {
                                "type": "breakout",
                                "direction": breakout["direction"],
                                "strength": breakout["strength"],
                                "level": breakout["level"],
                            }
                        )

            # Check chart patterns
            for _timeframe, patterns in pattern_analysis.get(
                "chart_patterns", {}
            ).items():
                for pattern in patterns:
                    if pattern.confidence > 0.7:  # High confidence patterns
                        active_patterns.append(
                            {
                                "type": "chart_pattern",
                                "name": pattern.pattern_name,
                                "direction": pattern.direction,
                                "confidence": pattern.confidence,
                            }
                        )

            return {
                "symbol": self.symbol,
                "active_patterns": active_patterns,
                "pattern_count": len(active_patterns),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error in pattern check for {self.symbol}: {str(e)}")
            raise

    async def _volume_check(self) -> dict[str, Any]:
        """Check for volume anomalies"""
        try:
            volume_analyzer = self.analyzers["volume"]
            anomalies = volume_analyzer.anomaly_detector.detect_anomalies(
                self.symbol, ["1min"]
            )

            # Filter significant anomalies
            significant_anomalies = [
                anomaly
                for anomaly in anomalies
                if anomaly.severity > 0.7 or anomaly.volume_ratio > 2.0
            ]

            return {
                "symbol": self.symbol,
                "anomalies": [anomaly.to_dict() for anomaly in significant_anomalies],
                "anomaly_count": len(significant_anomalies),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error in volume check for {self.symbol}: {str(e)}")
            raise

    def needs_analysis(self, task_type: str) -> bool:
        """Check if analysis is needed based on interval"""
        if task_type not in self.last_analysis:
            return True

        last_run = self.last_analysis[task_type]
        interval = self.analysis_intervals[task_type]

        return (datetime.now() - last_run).total_seconds() > interval


class ConcurrentMonitor:
    """Concurrent monitoring system for multiple stocks"""

    def __init__(self, data_processor: RealTimeDataProcessor, max_workers: int = 10):
        self.data_processor = data_processor
        self.max_workers = max_workers

        # Initialize analyzers
        self.analyzers = {
            "data_processor": data_processor,
            "statistical": StatisticalAnalyzer(data_processor),
            "pattern": PatternRecognitionEngine(data_processor),
            "volume": VolumeSignalGenerator(data_processor),
        }

        # Monitoring state
        self.monitors: dict[str, SymbolMonitor] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.result_queue: asyncio.Queue = asyncio.Queue()

        # Worker management
        self.workers: list[asyncio.Task] = []
        self.is_running = False
        self.worker_stats = defaultdict(int)

        # Callbacks for results
        self.result_callbacks: dict[str, list[Callable]] = defaultdict(list)

        # Performance tracking
        self.performance_stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_processing_time": 0,
            "symbols_monitored": 0,
            "last_update": datetime.now(),
        }

    def add_symbol(self, symbol: str) -> bool:
        """Add symbol to monitoring"""
        try:
            if symbol not in self.monitors:
                self.monitors[symbol] = SymbolMonitor(symbol, self.analyzers)
                self.performance_stats["symbols_monitored"] = len(self.monitors)
                logger.info(f"Added {symbol} to concurrent monitoring")
                return True
            return False

        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {str(e)}")
            return False

    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol from monitoring"""
        if symbol in self.monitors:
            del self.monitors[symbol]
            self.performance_stats["symbols_monitored"] = len(self.monitors)
            logger.info(f"Removed {symbol} from concurrent monitoring")
            return True
        return False

    async def start_monitoring(self):
        """Start the concurrent monitoring system"""
        if self.is_running:
            logger.warning("Monitoring already running")
            return

        self.is_running = True
        logger.info(f"Starting concurrent monitoring with {self.max_workers} workers")

        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)

        # Start task scheduler
        asyncio.create_task(self._scheduler())

        # Start result processor
        asyncio.create_task(self._result_processor())

        logger.info("Concurrent monitoring started")

    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)

        self.workers.clear()
        logger.info("Concurrent monitoring stopped")

    async def _scheduler(self):
        """Task scheduler that queues analysis tasks"""
        while self.is_running:
            try:
                current_time = datetime.now()

                for symbol, monitor in self.monitors.items():
                    # Check each analysis type
                    for task_type in [
                        "volume_check",
                        "quick_scan",
                        "pattern_check",
                        "full_analysis",
                    ]:
                        if monitor.needs_analysis(task_type):
                            # Determine priority based on task type and market conditions
                            priority = self._calculate_task_priority(symbol, task_type)

                            task = MonitoringTask(
                                symbol=symbol,
                                task_type=task_type,
                                priority=priority,
                                created_at=current_time,
                            )

                            await self.task_queue.put((priority, task))

                # Schedule next run
                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in scheduler: {str(e)}")
                await asyncio.sleep(10)

    async def _worker(self, worker_id: str):
        """Worker that processes monitoring tasks"""
        logger.info(f"Worker {worker_id} started")

        while self.is_running:
            try:
                # Get task from queue with timeout
                try:
                    priority, task = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                except TimeoutError:
                    continue

                # Process the task
                if task.symbol in self.monitors:
                    monitor = self.monitors[task.symbol]
                    result = await monitor.run_analysis(task.task_type)

                    # Put result in result queue
                    await self.result_queue.put(result)

                    # Update worker stats
                    self.worker_stats[worker_id] += 1

                # Mark task as done
                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {str(e)}")
                await asyncio.sleep(1)

        logger.info(f"Worker {worker_id} stopped")

    async def _result_processor(self):
        """Process analysis results and trigger callbacks"""
        while self.is_running:
            try:
                # Get result from queue
                try:
                    result = await asyncio.wait_for(
                        self.result_queue.get(), timeout=1.0
                    )
                except TimeoutError:
                    continue

                # Update performance stats
                self._update_performance_stats(result)

                # Trigger callbacks for this result type
                callbacks = self.result_callbacks.get(result.task_type, [])
                for callback in callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                    except Exception as e:
                        logger.error(f"Error in result callback: {str(e)}")

                # Log significant results
                if result.success and result.task_type == "full_analysis":
                    logger.debug(
                        f"Completed full analysis for {result.symbol} in {result.processing_time:.2f}s"
                    )
                elif not result.success:
                    logger.warning(
                        f"Failed {result.task_type} for {result.symbol}: {result.error}"
                    )

            except Exception as e:
                logger.error(f"Error in result processor: {str(e)}")
                await asyncio.sleep(1)

    def _calculate_task_priority(self, symbol: str, task_type: str) -> int:
        """Calculate task priority based on various factors"""
        base_priorities = {
            "volume_check": 5,  # Highest priority for volume anomalies
            "quick_scan": 4,  # Quick checks
            "pattern_check": 3,  # Pattern analysis
            "full_analysis": 2,  # Comprehensive analysis
        }

        priority = base_priorities.get(task_type, 1)

        # Boost priority for symbols with recent activity
        latest_tick = self.data_processor.get_latest_tick(symbol)
        if latest_tick:
            # High volume or significant price change gets priority boost
            if abs(latest_tick.change_percent) > 2.0:  # 2% price change
                priority += 1

            # Check volume ratio if available
            recent_ticks = self.data_processor.get_tick_history(symbol, 5)
            if len(recent_ticks) > 1:
                volumes = [tick.volume for tick in recent_ticks]
                if volumes[-1] > sum(volumes[:-1]) / len(volumes[:-1]) * 2:  # 2x volume
                    priority += 1

        return min(priority, 5)  # Cap at 5

    def _update_performance_stats(self, result: MonitoringResult):
        """Update performance statistics"""
        if result.success:
            self.performance_stats["tasks_completed"] += 1

            # Update average processing time
            current_avg = self.performance_stats["avg_processing_time"]
            total_tasks = self.performance_stats["tasks_completed"]
            new_avg = (
                (current_avg * (total_tasks - 1)) + result.processing_time
            ) / total_tasks
            self.performance_stats["avg_processing_time"] = new_avg
        else:
            self.performance_stats["tasks_failed"] += 1

        self.performance_stats["last_update"] = datetime.now()

    def add_result_callback(self, task_type: str, callback: Callable):
        """Add callback for specific result type"""
        self.result_callbacks[task_type].append(callback)

    def remove_result_callback(self, task_type: str, callback: Callable):
        """Remove result callback"""
        if callback in self.result_callbacks[task_type]:
            self.result_callbacks[task_type].remove(callback)

    async def force_analysis(
        self, symbol: str, task_type: str = "full_analysis"
    ) -> MonitoringResult | None:
        """Force immediate analysis of a symbol"""
        if symbol not in self.monitors:
            logger.warning(f"Symbol {symbol} not being monitored")
            return None

        try:
            monitor = self.monitors[symbol]
            result = await monitor.run_analysis(task_type)

            # Process result immediately
            await self.result_queue.put(result)

            return result

        except Exception as e:
            logger.error(f"Error forcing analysis for {symbol}: {str(e)}")
            return None

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get current monitoring status"""
        queue_size = self.task_queue.qsize() if hasattr(self.task_queue, "qsize") else 0

        return {
            "is_running": self.is_running,
            "symbols_monitored": len(self.monitors),
            "active_workers": len(self.workers),
            "queue_size": queue_size,
            "worker_stats": dict(self.worker_stats),
            "performance_stats": self.performance_stats.copy(),
            "symbols": list(self.monitors.keys()),
        }

    def get_symbol_status(self, symbol: str) -> dict[str, Any] | None:
        """Get status for a specific symbol"""
        if symbol not in self.monitors:
            return None

        monitor = self.monitors[symbol]

        return {
            "symbol": symbol,
            "last_analysis": {
                task_type: last_time.isoformat() if last_time else None
                for task_type, last_time in monitor.last_analysis.items()
            },
            "analysis_intervals": monitor.analysis_intervals,
            "needs_analysis": {
                task_type: monitor.needs_analysis(task_type)
                for task_type in monitor.analysis_intervals.keys()
            },
        }
