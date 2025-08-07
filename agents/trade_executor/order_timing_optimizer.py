import logging
from collections import deque
from datetime import datetime, time
from enum import Enum
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """Market phases for timing optimization"""

    PRE_OPEN = "PRE_OPEN"
    OPENING = "OPENING"
    MORNING = "MORNING"
    MIDDAY = "MIDDAY"
    AFTERNOON = "AFTERNOON"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


class OrderTimingOptimizer:
    """Optimizes order timing based on market conditions and historical patterns"""

    def __init__(self):
        # Market timing parameters (NSE timings)
        self.market_phases = {
            MarketPhase.PRE_OPEN: (time(9, 0), time(9, 15)),
            MarketPhase.OPENING: (time(9, 15), time(9, 45)),
            MarketPhase.MORNING: (time(9, 45), time(12, 0)),
            MarketPhase.MIDDAY: (time(12, 0), time(14, 0)),
            MarketPhase.AFTERNOON: (time(14, 0), time(15, 15)),
            MarketPhase.CLOSING: (time(15, 15), time(15, 30)),
        }

        # Volume patterns by market phase (historical averages)
        self.volume_patterns = {
            MarketPhase.OPENING: 1.5,  # 50% higher volume
            MarketPhase.MORNING: 1.2,  # 20% higher volume
            MarketPhase.MIDDAY: 0.8,  # 20% lower volume
            MarketPhase.AFTERNOON: 1.0,  # Normal volume
            MarketPhase.CLOSING: 1.3,  # 30% higher volume
        }

        # Spread patterns by market phase
        self.spread_patterns = {
            MarketPhase.OPENING: 1.8,  # 80% wider spreads
            MarketPhase.MORNING: 1.2,  # 20% wider spreads
            MarketPhase.MIDDAY: 1.5,  # 50% wider spreads
            MarketPhase.AFTERNOON: 1.0,  # Normal spreads
            MarketPhase.CLOSING: 1.4,  # 40% wider spreads
        }

        # Order execution strategies
        self.execution_strategies = {
            "AGGRESSIVE": {"urgency": 1.0, "slice_time": 30},  # 30 seconds
            "NORMAL": {"urgency": 0.5, "slice_time": 60},  # 1 minute
            "PASSIVE": {"urgency": 0.2, "slice_time": 180},  # 3 minutes
            "PATIENT": {"urgency": 0.1, "slice_time": 300},  # 5 minutes
        }

        # Historical execution data
        self.execution_history = deque(maxlen=1000)
        self.symbol_patterns: dict[str, dict] = {}

    def get_optimal_timing(
        self, symbol: str, action: str, quantity: int, urgency: str = "NORMAL"
    ) -> dict[str, Any]:
        """Get optimal timing parameters for order execution"""
        current_time = datetime.now().time()
        market_phase = self._get_market_phase(current_time)

        # Get execution strategy
        strategy = self.execution_strategies.get(
            urgency, self.execution_strategies["NORMAL"]
        )

        # Calculate order slicing
        slice_params = self._calculate_order_slices(quantity, market_phase, strategy)

        # Get timing recommendations
        timing_params = {
            "execute_now": self._should_execute_now(market_phase, action),
            "wait_time": self._calculate_wait_time(market_phase, current_time),
            "market_phase": market_phase.value,
            "volume_factor": self.volume_patterns.get(market_phase, 1.0),
            "spread_factor": self.spread_patterns.get(market_phase, 1.0),
            "slices": slice_params["slices"],
            "slice_interval": slice_params["interval"],
            "execution_strategy": urgency,
            "recommendations": self._get_timing_recommendations(
                symbol, action, market_phase
            ),
        }

        return timing_params

    def _get_market_phase(self, current_time: time) -> MarketPhase:
        """Determine current market phase"""
        for phase, (start, end) in self.market_phases.items():
            if start <= current_time < end:
                return phase
        return MarketPhase.CLOSED

    def _should_execute_now(self, market_phase: MarketPhase, action: str) -> bool:
        """Determine if order should be executed immediately"""
        # Avoid opening and closing volatility for normal orders
        if market_phase in [MarketPhase.OPENING, MarketPhase.CLOSING]:
            return False

        # Best times for execution
        if market_phase in [MarketPhase.MORNING, MarketPhase.AFTERNOON]:
            return True

        # Midday can be good for patient orders
        if market_phase == MarketPhase.MIDDAY:
            return action == "BUY"  # Better for accumulation

        return False

    def _calculate_wait_time(
        self, market_phase: MarketPhase, current_time: time
    ) -> int:
        """Calculate optimal wait time in seconds"""
        if market_phase == MarketPhase.OPENING:
            # Wait until opening volatility settles (30 mins after open)
            target_time = time(9, 45)
            wait_seconds = self._time_difference_seconds(current_time, target_time)
            return max(0, wait_seconds)

        elif market_phase == MarketPhase.CLOSING:
            # Execute immediately if in closing phase
            return 0

        elif market_phase == MarketPhase.MIDDAY:
            # Maybe wait for afternoon session
            target_time = time(14, 0)
            wait_seconds = self._time_difference_seconds(current_time, target_time)
            # Only wait if less than 30 minutes
            return wait_seconds if wait_seconds < 1800 else 0

        return 0

    def _time_difference_seconds(self, time1: time, time2: time) -> int:
        """Calculate difference between two times in seconds"""
        date_today = datetime.today().date()
        datetime1 = datetime.combine(date_today, time1)
        datetime2 = datetime.combine(date_today, time2)
        return int((datetime2 - datetime1).total_seconds())

    def _calculate_order_slices(
        self, quantity: int, market_phase: MarketPhase, strategy: dict
    ) -> dict[str, Any]:
        """Calculate how to slice large orders"""
        # Determine slice size based on market phase
        volume_factor = self.volume_patterns.get(market_phase, 1.0)

        # Base slice size (adjust based on your needs)
        base_slice_size = 100
        adjusted_slice_size = int(base_slice_size * volume_factor)

        # Calculate number of slices
        if quantity <= adjusted_slice_size:
            return {"slices": [quantity], "interval": 0}

        num_slices = min(10, max(2, quantity // adjusted_slice_size))
        slice_size = quantity // num_slices

        # Create slices with remainder in last slice
        slices = [slice_size] * (num_slices - 1)
        slices.append(quantity - sum(slices))

        # Calculate interval between slices
        interval = strategy["slice_time"] // num_slices

        return {"slices": slices, "interval": interval}

    def _get_timing_recommendations(
        self, symbol: str, action: str, market_phase: MarketPhase
    ) -> list[str]:
        """Get specific timing recommendations"""
        recommendations = []

        if market_phase == MarketPhase.OPENING:
            recommendations.append("Avoid first 30 minutes due to high volatility")
            recommendations.append("Consider waiting for price discovery to complete")

        elif market_phase == MarketPhase.MIDDAY:
            recommendations.append("Lower liquidity period - use limit orders")
            recommendations.append("Consider splitting order into smaller chunks")

        elif market_phase == MarketPhase.CLOSING:
            recommendations.append("High volatility expected - monitor closely")
            recommendations.append("Ensure sufficient time for order execution")

        # Symbol-specific patterns
        if symbol in self.symbol_patterns:
            pattern = self.symbol_patterns[symbol]
            if pattern.get("high_volatility_times"):
                recommendations.append(
                    f"Historical high volatility at {pattern['high_volatility_times']}"
                )

        return recommendations

    def analyze_execution_quality(
        self,
        order_id: str,
        symbol: str,
        expected_price: float,
        executed_price: float,
        execution_time: float,
    ) -> dict[str, Any]:
        """Analyze the quality of an execution"""
        slippage = abs(executed_price - expected_price) / expected_price * 100
        market_phase = self._get_market_phase(datetime.now().time())

        # Determine execution quality
        quality_score = self._calculate_quality_score(
            slippage, execution_time, market_phase
        )

        # Store in history
        execution_data = {
            "order_id": order_id,
            "symbol": symbol,
            "timestamp": datetime.now(),
            "market_phase": market_phase.value,
            "slippage": slippage,
            "execution_time": execution_time,
            "quality_score": quality_score,
        }

        self.execution_history.append(execution_data)

        # Update symbol patterns
        self._update_symbol_patterns(symbol, execution_data)

        return {
            "quality_score": quality_score,
            "slippage_pct": slippage,
            "execution_time": execution_time,
            "market_phase": market_phase.value,
            "rating": self._get_quality_rating(quality_score),
        }

    def _calculate_quality_score(
        self, slippage: float, execution_time: float, market_phase: MarketPhase
    ) -> float:
        """Calculate execution quality score (0-100)"""
        # Base score
        score = 100.0

        # Deduct for slippage (up to 50 points)
        slippage_penalty = min(50, slippage * 10)
        score -= slippage_penalty

        # Deduct for execution time (up to 20 points)
        time_penalty = min(20, execution_time / 10)
        score -= time_penalty

        # Adjust for market phase
        phase_factor = {
            MarketPhase.OPENING: 0.8,  # Lower expectations
            MarketPhase.MORNING: 1.0,  # Normal
            MarketPhase.MIDDAY: 0.9,  # Slightly lower
            MarketPhase.AFTERNOON: 1.0,  # Normal
            MarketPhase.CLOSING: 0.85,  # Lower expectations
        }.get(market_phase, 1.0)

        score *= phase_factor

        return max(0, min(100, score))

    def _get_quality_rating(self, score: float) -> str:
        """Get quality rating based on score"""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 75:
            return "GOOD"
        elif score >= 60:
            return "AVERAGE"
        elif score >= 40:
            return "BELOW_AVERAGE"
        else:
            return "POOR"

    def _update_symbol_patterns(self, symbol: str, execution_data: dict) -> None:
        """Update symbol-specific patterns"""
        if symbol not in self.symbol_patterns:
            self.symbol_patterns[symbol] = {
                "executions": [],
                "avg_slippage_by_phase": {},
                "high_volatility_times": [],
            }

        pattern = self.symbol_patterns[symbol]
        pattern["executions"].append(execution_data)

        # Keep only recent executions
        if len(pattern["executions"]) > 100:
            pattern["executions"] = pattern["executions"][-100:]

        # Update average slippage by phase
        phase = execution_data["market_phase"]
        if phase not in pattern["avg_slippage_by_phase"]:
            pattern["avg_slippage_by_phase"][phase] = []

        pattern["avg_slippage_by_phase"][phase].append(execution_data["slippage"])

    def get_best_execution_times(self, symbol: str) -> list[tuple[time, time]]:
        """Get recommended execution time windows for a symbol"""
        # Default best times based on general market patterns
        best_times = [
            (time(10, 0), time(11, 30)),  # Morning after opening volatility
            (time(14, 0), time(15, 0)),  # Afternoon before closing
        ]

        # Adjust based on symbol-specific patterns if available
        if symbol in self.symbol_patterns:
            pattern = self.symbol_patterns[symbol]

            # Find phases with lowest average slippage
            phase_slippages = {}
            for phase, slippages in pattern["avg_slippage_by_phase"].items():
                if slippages:
                    phase_slippages[phase] = np.mean(slippages)

            # Get best phases
            if phase_slippages:
                sorted_phases = sorted(phase_slippages.items(), key=lambda x: x[1])
                best_phases = [phase for phase, _ in sorted_phases[:2]]

                # Convert phases to time windows
                custom_times = []
                for phase in best_phases:
                    if phase in self.market_phases:
                        custom_times.append(self.market_phases[phase])

                if custom_times:
                    best_times = custom_times

        return best_times

    def get_execution_statistics(self) -> dict[str, Any]:
        """Get overall execution statistics"""
        if not self.execution_history:
            return {"message": "No execution history available"}

        # Calculate statistics
        slippages = [e["slippage"] for e in self.execution_history]
        execution_times = [e["execution_time"] for e in self.execution_history]
        quality_scores = [e["quality_score"] for e in self.execution_history]

        # Group by market phase
        phase_stats = {}
        for phase in MarketPhase:
            phase_data = [
                e for e in self.execution_history if e["market_phase"] == phase.value
            ]

            if phase_data:
                phase_stats[phase.value] = {
                    "count": len(phase_data),
                    "avg_slippage": np.mean([e["slippage"] for e in phase_data]),
                    "avg_quality": np.mean([e["quality_score"] for e in phase_data]),
                }

        return {
            "total_executions": len(self.execution_history),
            "average_slippage": np.mean(slippages),
            "average_execution_time": np.mean(execution_times),
            "average_quality_score": np.mean(quality_scores),
            "best_quality_score": max(quality_scores),
            "worst_quality_score": min(quality_scores),
            "phase_statistics": phase_stats,
        }
