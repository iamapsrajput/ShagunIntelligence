import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributionMode(Enum):
    """Data distribution modes"""

    BROADCAST = "broadcast"  # Send to all subscribers
    ROUND_ROBIN = "round_robin"  # Distribute evenly
    SMART_ROUTING = "smart"  # Route based on subscriber performance
    PRIORITY = "priority"  # Route to high-priority subscribers first


@dataclass
class Subscriber:
    """Represents a data subscriber"""

    id: str
    callback: Callable
    symbol_filter: set[str] | None = None
    priority: int = 5  # 1-10, higher is more important
    batch_size: int = 1
    max_latency_ms: float = 100
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SubscriberMetrics:
    """Performance metrics for a subscriber"""

    messages_sent: int = 0
    messages_failed: int = 0
    total_latency_ms: float = 0
    last_update: datetime | None = None
    avg_processing_time_ms: float = 0
    health_score: float = 100.0  # 0-100


class DataDistributor:
    """Low-latency data distribution to subscribers"""

    def __init__(self, mode: DistributionMode = DistributionMode.BROADCAST):
        self.mode = mode

        # Subscriber management
        self.subscribers: dict[str, Subscriber] = {}
        self.subscriber_metrics: dict[str, SubscriberMetrics] = defaultdict(
            SubscriberMetrics
        )

        # Symbol-based routing
        self.symbol_subscribers: dict[str, set[str]] = defaultdict(set)

        # Batch processing
        self.batch_queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.batch_tasks: dict[str, asyncio.Task] = {}

        # Performance optimization
        self.distribution_pool = asyncio.Queue(maxsize=10000)
        self.worker_tasks: list[asyncio.Task] = []
        self.num_workers = 4

        # Statistics
        self.stats = {
            "total_distributed": 0,
            "distribution_errors": 0,
            "avg_latency_ms": 0,
            "peak_latency_ms": 0,
        }

        # Round-robin state
        self.round_robin_indices: dict[str, int] = defaultdict(int)

        logger.info(f"DataDistributor initialized with mode: {mode.value}")

    async def start(self) -> None:
        """Start the distributor workers"""
        # Start distribution workers
        for i in range(self.num_workers):
            task = asyncio.create_task(self._distribution_worker(i))
            self.worker_tasks.append(task)

        logger.info(f"Started {self.num_workers} distribution workers")

    async def stop(self) -> None:
        """Stop the distributor"""
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        # Cancel batch tasks
        for task in self.batch_tasks.values():
            task.cancel()

        logger.info("DataDistributor stopped")

    def add_subscriber(
        self,
        symbol: str,
        callback: Callable,
        subscriber_id: str | None = None,
        priority: int = 5,
        batch_size: int = 1,
    ) -> str:
        """Add a subscriber for a symbol"""
        if subscriber_id is None:
            subscriber_id = f"sub_{len(self.subscribers)}_{int(time.time())}"

        subscriber = Subscriber(
            id=subscriber_id,
            callback=callback,
            symbol_filter={symbol} if symbol != "*" else None,
            priority=priority,
            batch_size=batch_size,
        )

        self.subscribers[subscriber_id] = subscriber

        # Update symbol routing
        self.symbol_subscribers[symbol].add(subscriber_id)

        # Start batch processor if needed
        if batch_size > 1:
            self._start_batch_processor(subscriber_id)

        logger.info(f"Added subscriber {subscriber_id} for symbol {symbol}")

        return subscriber_id

    def remove_subscriber(self, symbol: str, callback: Callable) -> None:
        """Remove a subscriber"""
        # Find subscriber by callback
        subscriber_id = None
        for sub_id, subscriber in self.subscribers.items():
            if subscriber.callback == callback:
                subscriber_id = sub_id
                break

        if subscriber_id:
            # Remove from subscribers
            del self.subscribers[subscriber_id]

            # Remove from symbol routing
            self.symbol_subscribers[symbol].discard(subscriber_id)

            # Stop batch processor if exists
            if subscriber_id in self.batch_tasks:
                self.batch_tasks[subscriber_id].cancel()
                del self.batch_tasks[subscriber_id]

            logger.info(f"Removed subscriber {subscriber_id}")

    async def distribute_tick(self, symbol: str, tick_data: dict[str, Any]) -> None:
        """Distribute tick data to subscribers"""
        # Add to distribution queue
        await self.distribution_pool.put(
            {"symbol": symbol, "data": tick_data, "timestamp": time.time()}
        )

    async def _distribution_worker(self, worker_id: int) -> None:
        """Worker process for distributing data"""
        logger.info(f"Distribution worker {worker_id} started")

        while True:
            try:
                # Get item from queue
                item = await self.distribution_pool.get()

                # Distribute based on mode
                if self.mode == DistributionMode.BROADCAST:
                    await self._broadcast_distribution(item)
                elif self.mode == DistributionMode.ROUND_ROBIN:
                    await self._round_robin_distribution(item)
                elif self.mode == DistributionMode.SMART_ROUTING:
                    await self._smart_distribution(item)
                elif self.mode == DistributionMode.PRIORITY:
                    await self._priority_distribution(item)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in distribution worker {worker_id}: {str(e)}")
                self.stats["distribution_errors"] += 1

    async def _broadcast_distribution(self, item: dict[str, Any]) -> None:
        """Broadcast to all relevant subscribers"""
        symbol = item["symbol"]
        data = item["data"]

        # Get subscribers for this symbol
        subscriber_ids = self._get_subscribers_for_symbol(symbol)

        # Distribute to all subscribers concurrently
        tasks = []
        for subscriber_id in subscriber_ids:
            subscriber = self.subscribers.get(subscriber_id)
            if subscriber:
                task = asyncio.create_task(
                    self._send_to_subscriber(subscriber, data, item["timestamp"])
                )
                tasks.append(task)

        # Wait for all distributions
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _round_robin_distribution(self, item: dict[str, Any]) -> None:
        """Distribute using round-robin algorithm"""
        symbol = item["symbol"]
        data = item["data"]

        # Get subscribers for this symbol
        subscriber_ids = list(self._get_subscribers_for_symbol(symbol))

        if not subscriber_ids:
            return

        # Get current index for this symbol
        current_index = self.round_robin_indices[symbol]

        # Select subscriber
        selected_id = subscriber_ids[current_index % len(subscriber_ids)]
        subscriber = self.subscribers.get(selected_id)

        if subscriber:
            await self._send_to_subscriber(subscriber, data, item["timestamp"])

        # Update index
        self.round_robin_indices[symbol] = (current_index + 1) % len(subscriber_ids)

    async def _smart_distribution(self, item: dict[str, Any]) -> None:
        """Smart routing based on subscriber performance"""
        symbol = item["symbol"]
        data = item["data"]

        # Get subscribers and their metrics
        candidates = []
        for subscriber_id in self._get_subscribers_for_symbol(symbol):
            subscriber = self.subscribers.get(subscriber_id)
            if subscriber:
                metrics = self.subscriber_metrics[subscriber_id]
                candidates.append((subscriber, metrics))

        if not candidates:
            return

        # Sort by health score (best first)
        candidates.sort(key=lambda x: x[1].health_score, reverse=True)

        # Distribute to top performers
        top_count = max(1, len(candidates) // 3)  # Top third
        tasks = []

        for subscriber, _ in candidates[:top_count]:
            task = asyncio.create_task(
                self._send_to_subscriber(subscriber, data, item["timestamp"])
            )
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _priority_distribution(self, item: dict[str, Any]) -> None:
        """Distribute based on subscriber priority"""
        symbol = item["symbol"]
        data = item["data"]

        # Get subscribers sorted by priority
        subscribers = []
        for subscriber_id in self._get_subscribers_for_symbol(symbol):
            subscriber = self.subscribers.get(subscriber_id)
            if subscriber:
                subscribers.append(subscriber)

        # Sort by priority (highest first)
        subscribers.sort(key=lambda x: x.priority, reverse=True)

        # Distribute to high-priority subscribers first
        tasks = []
        for subscriber in subscribers:
            # Add delay for lower priority subscribers
            delay = (10 - subscriber.priority) * 0.001  # 0-9ms delay

            task = asyncio.create_task(
                self._send_with_delay(subscriber, data, item["timestamp"], delay)
            )
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_with_delay(
        self,
        subscriber: Subscriber,
        data: dict[str, Any],
        timestamp: float,
        delay: float,
    ) -> None:
        """Send data with optional delay"""
        if delay > 0:
            await asyncio.sleep(delay)

        await self._send_to_subscriber(subscriber, data, timestamp)

    async def _send_to_subscriber(
        self, subscriber: Subscriber, data: dict[str, Any], enqueue_timestamp: float
    ) -> None:
        """Send data to a single subscriber"""
        start_time = time.time()

        try:
            # Calculate latency
            latency_ms = (start_time - enqueue_timestamp) * 1000

            # Check if subscriber uses batching
            if subscriber.batch_size > 1:
                # Add to batch queue
                self.batch_queues[subscriber.id].append(data)
            else:
                # Send immediately
                if asyncio.iscoroutinefunction(subscriber.callback):
                    await subscriber.callback(data)
                else:
                    # Run sync callback in executor
                    await asyncio.get_event_loop().run_in_executor(
                        None, subscriber.callback, data
                    )

            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_subscriber_metrics(
                subscriber.id, True, latency_ms, processing_time_ms
            )

            # Update global stats
            self.stats["total_distributed"] += 1
            self._update_latency_stats(latency_ms)

        except Exception as e:
            logger.error(f"Error sending to subscriber {subscriber.id}: {str(e)}")
            self._update_subscriber_metrics(subscriber.id, False, 0, 0)
            self.stats["distribution_errors"] += 1

    def _get_subscribers_for_symbol(self, symbol: str) -> set[str]:
        """Get all subscribers for a symbol"""
        # Direct symbol subscribers
        subscribers = self.symbol_subscribers.get(symbol, set()).copy()

        # Add wildcard subscribers
        subscribers.update(self.symbol_subscribers.get("*", set()))

        return subscribers

    def _update_subscriber_metrics(
        self,
        subscriber_id: str,
        success: bool,
        latency_ms: float,
        processing_time_ms: float,
    ) -> None:
        """Update metrics for a subscriber"""
        metrics = self.subscriber_metrics[subscriber_id]

        if success:
            metrics.messages_sent += 1
            metrics.total_latency_ms += latency_ms

            # Update average processing time
            total = metrics.messages_sent
            metrics.avg_processing_time_ms = (
                metrics.avg_processing_time_ms * (total - 1) + processing_time_ms
            ) / total
        else:
            metrics.messages_failed += 1

        metrics.last_update = datetime.now()

        # Update health score
        total_messages = metrics.messages_sent + metrics.messages_failed
        if total_messages > 0:
            success_rate = metrics.messages_sent / total_messages
            avg_latency = metrics.total_latency_ms / max(1, metrics.messages_sent)

            # Health score based on success rate and latency
            health = success_rate * 70  # 70% weight on success

            # Latency penalty (assumes 100ms is acceptable)
            latency_penalty = min(30, (avg_latency / 100) * 30)
            health += 30 - latency_penalty

            metrics.health_score = max(0, min(100, health))

    def _update_latency_stats(self, latency_ms: float) -> None:
        """Update global latency statistics"""
        # Update peak latency
        self.stats["peak_latency_ms"] = max(self.stats["peak_latency_ms"], latency_ms)

        # Update average latency
        total = self.stats["total_distributed"]
        if total > 0:
            current_avg = self.stats["avg_latency_ms"]
            self.stats["avg_latency_ms"] = (
                current_avg * (total - 1) + latency_ms
            ) / total

    def _start_batch_processor(self, subscriber_id: str) -> None:
        """Start batch processor for a subscriber"""
        if subscriber_id not in self.batch_tasks:
            task = asyncio.create_task(self._batch_processor(subscriber_id))
            self.batch_tasks[subscriber_id] = task

    async def _batch_processor(self, subscriber_id: str) -> None:
        """Process batched messages for a subscriber"""
        subscriber = self.subscribers[subscriber_id]

        while subscriber_id in self.subscribers:
            try:
                # Wait for batch to fill or timeout
                await asyncio.sleep(0.1)  # 100ms batch window

                # Get pending messages
                messages = self.batch_queues[subscriber_id]
                if messages:
                    # Send batch
                    batch = messages[: subscriber.batch_size]
                    self.batch_queues[subscriber_id] = messages[subscriber.batch_size :]

                    # Send to subscriber
                    if asyncio.iscoroutinefunction(subscriber.callback):
                        await subscriber.callback(batch)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None, subscriber.callback, batch
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor for {subscriber_id}: {str(e)}")

    def get_distribution_stats(self) -> dict[str, Any]:
        """Get distribution statistics"""
        subscriber_stats = {}

        for sub_id, metrics in self.subscriber_metrics.items():
            if sub_id in self.subscribers:
                subscriber_stats[sub_id] = {
                    "messages_sent": metrics.messages_sent,
                    "messages_failed": metrics.messages_failed,
                    "avg_latency_ms": (
                        metrics.total_latency_ms / max(1, metrics.messages_sent)
                    ),
                    "avg_processing_time_ms": metrics.avg_processing_time_ms,
                    "health_score": metrics.health_score,
                    "last_update": (
                        metrics.last_update.isoformat() if metrics.last_update else None
                    ),
                }

        return {
            **self.stats,
            "mode": self.mode.value,
            "active_subscribers": len(self.subscribers),
            "queue_size": self.distribution_pool.qsize(),
            "subscriber_stats": subscriber_stats,
        }

    def set_distribution_mode(self, mode: DistributionMode) -> None:
        """Change distribution mode"""
        self.mode = mode
        logger.info(f"Distribution mode changed to: {mode.value}")
