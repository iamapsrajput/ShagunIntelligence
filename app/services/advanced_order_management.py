"""
Advanced Order Management System
Provides sophisticated order types, smart routing, and execution algorithms
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
from loguru import logger

from app.core.resilience import with_circuit_breaker, with_retry


class AdvancedOrderType(Enum):
    """Advanced order types supported by the system"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "SL"
    STOP_LOSS_MARKET = "SL-M"
    BRACKET = "BO"
    COVER = "CO"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"  # Time Weighted Average Price
    VWAP = "VWAP"  # Volume Weighted Average Price
    IMPLEMENTATION_SHORTFALL = "IS"
    ARRIVAL_PRICE = "AP"


class ExecutionStrategy(Enum):
    """Execution strategy types"""

    AGGRESSIVE = "AGGRESSIVE"  # Prioritize speed
    PASSIVE = "PASSIVE"  # Prioritize price
    BALANCED = "BALANCED"  # Balance speed and price
    STEALTH = "STEALTH"  # Minimize market impact


@dataclass
class MarketImpactModel:
    """Market impact estimation model"""

    permanent_impact_factor: float = 0.1
    temporary_impact_factor: float = 0.05
    volatility_adjustment: float = 1.0
    liquidity_adjustment: float = 1.0

    def estimate_impact(
        self, quantity: int, avg_volume: int, volatility: float
    ) -> dict[str, float]:
        """Estimate market impact for an order"""
        participation_rate = quantity / max(avg_volume, 1)

        permanent_impact = (
            self.permanent_impact_factor
            * participation_rate
            * volatility
            * self.volatility_adjustment
        )

        temporary_impact = (
            self.temporary_impact_factor
            * participation_rate
            * self.liquidity_adjustment
        )

        return {
            "permanent_impact": permanent_impact,
            "temporary_impact": temporary_impact,
            "total_impact": permanent_impact + temporary_impact,
            "participation_rate": participation_rate,
        }


@dataclass
class SlippageModel:
    """Slippage estimation and control model"""

    base_slippage: float = 0.001  # 0.1%
    volume_factor: float = 0.5
    volatility_factor: float = 0.3
    spread_factor: float = 0.2

    def estimate_slippage(
        self, order_size: int, avg_volume: int, volatility: float, spread: float
    ) -> float:
        """Estimate expected slippage"""
        volume_impact = (order_size / max(avg_volume, 1)) * self.volume_factor
        volatility_impact = volatility * self.volatility_factor
        spread_impact = spread * self.spread_factor

        return self.base_slippage + volume_impact + volatility_impact + spread_impact


@dataclass
class AdvancedOrderRequest:
    """Advanced order request with sophisticated parameters"""

    symbol: str
    exchange: str
    transaction_type: str  # BUY/SELL
    quantity: int
    order_type: AdvancedOrderType
    execution_strategy: ExecutionStrategy = ExecutionStrategy.BALANCED

    # Price parameters
    price: float | None = None
    trigger_price: float | None = None
    limit_price: float | None = None

    # Advanced parameters
    max_participation_rate: float = 0.1  # Max 10% of volume
    max_slippage: float = 0.005  # Max 0.5% slippage
    time_horizon: int = 300  # 5 minutes default
    urgency_factor: float = 0.5  # 0 = patient, 1 = urgent

    # Iceberg parameters
    iceberg_visible_quantity: int | None = None
    iceberg_variance: float = 0.1  # 10% variance in slice sizes

    # TWAP/VWAP parameters
    twap_intervals: int = 10
    vwap_lookback_periods: int = 20

    # Risk controls
    max_order_value: float | None = None
    position_limit_check: bool = True

    # Metadata
    strategy_id: str | None = None
    parent_order_id: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)


class SmartOrderRouter:
    """Intelligent order routing system"""

    def __init__(self):
        self.venue_configs = {
            "NSE": {"latency": 10, "fees": 0.0003, "liquidity_score": 0.9},
            "BSE": {"latency": 15, "fees": 0.0004, "liquidity_score": 0.7},
        }
        self.routing_history = {}

    def select_optimal_venue(self, symbol: str, order_size: int, urgency: float) -> str:
        """Select optimal execution venue"""
        # For Indian markets, NSE is typically the primary venue
        # This can be enhanced with real-time venue analysis

        scores = {}
        for venue, config in self.venue_configs.items():
            # Calculate venue score based on multiple factors
            latency_score = 1.0 / (config["latency"] / 10.0)
            fee_score = 1.0 / (config["fees"] * 1000)
            liquidity_score = config["liquidity_score"]

            # Weight factors based on urgency
            if urgency > 0.7:  # High urgency - prioritize speed
                total_score = latency_score * 0.6 + liquidity_score * 0.4
            else:  # Lower urgency - consider costs
                total_score = (
                    latency_score * 0.3 + fee_score * 0.3 + liquidity_score * 0.4
                )

            scores[venue] = total_score

        return max(scores, key=scores.get)


class ExecutionAlgorithm:
    """Base class for execution algorithms"""

    def __init__(self, name: str):
        self.name = name
        self.market_impact_model = MarketImpactModel()
        self.slippage_model = SlippageModel()

    async def execute(
        self, order_request: AdvancedOrderRequest, market_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute order using specific algorithm"""
        raise NotImplementedError


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time Weighted Average Price algorithm"""

    def __init__(self):
        super().__init__("TWAP")

    async def execute(
        self, order_request: AdvancedOrderRequest, market_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute TWAP order"""
        total_quantity = order_request.quantity
        intervals = order_request.twap_intervals
        time_per_interval = order_request.time_horizon / intervals

        # Calculate slice sizes
        base_slice_size = total_quantity // intervals
        remaining_quantity = total_quantity

        execution_plan = []
        for i in range(intervals):
            if i == intervals - 1:  # Last slice gets remainder
                slice_size = remaining_quantity
            else:
                # Add some randomization to avoid predictability
                variance = int(base_slice_size * 0.1)
                slice_size = base_slice_size + np.random.randint(
                    -variance, variance + 1
                )
                slice_size = min(slice_size, remaining_quantity)

            execution_plan.append(
                {
                    "slice_number": i + 1,
                    "quantity": slice_size,
                    "scheduled_time": datetime.now()
                    + timedelta(seconds=i * time_per_interval),
                    "status": "pending",
                }
            )

            remaining_quantity -= slice_size
            if remaining_quantity <= 0:
                break

        return {
            "algorithm": "TWAP",
            "execution_plan": execution_plan,
            "total_slices": len(execution_plan),
            "estimated_completion": datetime.now()
            + timedelta(seconds=order_request.time_horizon),
        }


class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume Weighted Average Price algorithm"""

    def __init__(self):
        super().__init__("VWAP")

    async def execute(
        self, order_request: AdvancedOrderRequest, market_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute VWAP order"""
        # Get historical volume profile
        volume_profile = market_data.get("volume_profile", [])
        if not volume_profile:
            # Fallback to TWAP if no volume data
            twap_algo = TWAPAlgorithm()
            return await twap_algo.execute(order_request, market_data)

        total_quantity = order_request.quantity
        total_expected_volume = sum(volume_profile)

        execution_plan = []
        remaining_quantity = total_quantity

        for i, expected_volume in enumerate(volume_profile):
            if remaining_quantity <= 0:
                break

            # Calculate slice size based on volume proportion
            volume_proportion = expected_volume / total_expected_volume
            target_participation = min(order_request.max_participation_rate, 0.2)

            slice_size = min(
                int(total_quantity * volume_proportion),
                int(expected_volume * target_participation),
                remaining_quantity,
            )

            if slice_size > 0:
                execution_plan.append(
                    {
                        "slice_number": i + 1,
                        "quantity": slice_size,
                        "expected_volume": expected_volume,
                        "participation_rate": slice_size / expected_volume,
                        "scheduled_time": datetime.now() + timedelta(minutes=i * 5),
                        "status": "pending",
                    }
                )

                remaining_quantity -= slice_size

        return {
            "algorithm": "VWAP",
            "execution_plan": execution_plan,
            "total_slices": len(execution_plan),
            "target_participation_rate": order_request.max_participation_rate,
        }


class IcebergAlgorithm(ExecutionAlgorithm):
    """Iceberg order algorithm"""

    def __init__(self):
        super().__init__("ICEBERG")

    async def execute(
        self, order_request: AdvancedOrderRequest, market_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute Iceberg order"""
        total_quantity = order_request.quantity
        visible_quantity = order_request.iceberg_visible_quantity or (
            total_quantity // 10
        )
        variance = order_request.iceberg_variance

        execution_plan = []
        remaining_quantity = total_quantity
        slice_number = 1

        while remaining_quantity > 0:
            # Calculate slice size with variance
            base_size = min(visible_quantity, remaining_quantity)
            variance_amount = int(base_size * variance)

            if variance_amount > 0:
                slice_size = base_size + np.random.randint(
                    -variance_amount, variance_amount + 1
                )
            else:
                slice_size = base_size

            slice_size = max(1, min(slice_size, remaining_quantity))

            execution_plan.append(
                {
                    "slice_number": slice_number,
                    "quantity": slice_size,
                    "visible_quantity": slice_size,
                    "hidden_quantity": remaining_quantity - slice_size,
                    "status": "pending",
                }
            )

            remaining_quantity -= slice_size
            slice_number += 1

        return {
            "algorithm": "ICEBERG",
            "execution_plan": execution_plan,
            "total_slices": len(execution_plan),
            "average_visible_size": visible_quantity,
        }


class AdvancedOrderManager:
    """Advanced Order Management System with sophisticated execution algorithms"""

    def __init__(self, kite_client=None, market_data_service=None):
        self.kite_client = kite_client
        self.market_data_service = market_data_service
        self.smart_router = SmartOrderRouter()

        # Initialize execution algorithms
        self.algorithms = {
            AdvancedOrderType.TWAP: TWAPAlgorithm(),
            AdvancedOrderType.VWAP: VWAPAlgorithm(),
            AdvancedOrderType.ICEBERG: IcebergAlgorithm(),
        }

        # Order tracking
        self.active_orders = {}
        self.execution_history = []

        # Performance metrics
        self.execution_metrics = {
            "total_orders": 0,
            "successful_orders": 0,
            "average_slippage": 0.0,
            "average_execution_time": 0.0,
            "market_impact_savings": 0.0,
        }

    @with_circuit_breaker("advanced_order_management")
    @with_retry(max_retries=2, delay=1.0)
    async def place_advanced_order(
        self, order_request: AdvancedOrderRequest
    ) -> dict[str, Any]:
        """Place an advanced order with sophisticated execution"""
        try:
            start_time = time.time()

            # Pre-execution validation
            validation_result = await self._validate_advanced_order(order_request)
            if not validation_result["valid"]:
                return {
                    "status": "rejected",
                    "reason": validation_result["reason"],
                    "order_id": None,
                }

            # Get market data for execution
            market_data = await self._get_market_data(order_request.symbol)

            # Estimate market impact and slippage
            impact_analysis = await self._analyze_market_impact(
                order_request, market_data
            )

            # Select optimal execution venue
            optimal_venue = self.smart_router.select_optimal_venue(
                order_request.symbol,
                order_request.quantity,
                order_request.urgency_factor,
            )

            # Choose and execute algorithm
            if order_request.order_type in self.algorithms:
                algorithm = self.algorithms[order_request.order_type]
                execution_plan = await algorithm.execute(order_request, market_data)

                # Execute the plan
                execution_result = await self._execute_plan(
                    order_request, execution_plan, optimal_venue
                )
            else:
                # Fallback to simple execution
                execution_result = await self._execute_simple_order(
                    order_request, optimal_venue
                )

            # Update metrics
            execution_time = time.time() - start_time
            await self._update_execution_metrics(execution_result, execution_time)

            return execution_result

        except Exception as e:
            logger.error(f"Advanced order execution failed: {str(e)}")
            return {"status": "error", "reason": str(e), "order_id": None}

    async def _validate_advanced_order(
        self, order_request: AdvancedOrderRequest
    ) -> dict[str, Any]:
        """Validate advanced order parameters"""
        try:
            # Basic validation
            if order_request.quantity <= 0:
                return {"valid": False, "reason": "Invalid quantity"}

            if order_request.max_participation_rate > 0.5:
                return {"valid": False, "reason": "Participation rate too high"}

            # Market hours validation
            current_time = datetime.now().time()
            market_open = datetime.strptime("09:15", "%H:%M").time()
            market_close = datetime.strptime("15:30", "%H:%M").time()

            if not (market_open <= current_time <= market_close):
                return {"valid": False, "reason": "Market is closed"}

            # Position limit validation if enabled
            if order_request.position_limit_check:
                position_check = await self._check_position_limits(order_request)
                if not position_check["valid"]:
                    return position_check

            return {"valid": True, "reason": "Order validated successfully"}

        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {str(e)}"}

    async def _get_market_data(self, symbol: str) -> dict[str, Any]:
        """Get comprehensive market data for execution"""
        try:
            if self.market_data_service:
                # Get real-time data
                quote = await self.market_data_service.get_quote(symbol)
                depth = await self.market_data_service.get_market_depth(symbol)
                volume_profile = await self.market_data_service.get_volume_profile(
                    symbol
                )

                return {
                    "quote": quote,
                    "market_depth": depth,
                    "volume_profile": volume_profile,
                    "timestamp": datetime.now(),
                }
            else:
                # Fallback to basic data
                return {
                    "quote": {"price": 100.0, "volume": 10000},
                    "market_depth": {"bid": 99.95, "ask": 100.05, "spread": 0.10},
                    "volume_profile": [1000] * 20,  # Mock volume profile
                    "timestamp": datetime.now(),
                }

        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return {}

    async def _analyze_market_impact(
        self, order_request: AdvancedOrderRequest, market_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze potential market impact of the order"""
        try:
            quote = market_data.get("quote", {})
            avg_volume = quote.get("volume", 10000)
            current_price = quote.get("price", 100.0)

            # Estimate volatility (simplified)
            volatility = 0.02  # 2% default volatility

            # Calculate market impact
            impact_model = MarketImpactModel()
            impact_estimate = impact_model.estimate_impact(
                order_request.quantity, avg_volume, volatility
            )

            # Calculate slippage estimate
            slippage_model = SlippageModel()
            depth = market_data.get("market_depth", {})
            spread = depth.get("spread", 0.1)

            slippage_estimate = slippage_model.estimate_slippage(
                order_request.quantity, avg_volume, volatility, spread
            )

            return {
                "market_impact": impact_estimate,
                "slippage_estimate": slippage_estimate,
                "recommendation": self._get_execution_recommendation(
                    impact_estimate, slippage_estimate, order_request
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing market impact: {str(e)}")
            return {"market_impact": {}, "slippage_estimate": 0.0}

    def _get_execution_recommendation(
        self,
        impact_estimate: dict[str, float],
        slippage_estimate: float,
        order_request: AdvancedOrderRequest,
    ) -> dict[str, Any]:
        """Get execution strategy recommendation"""
        total_cost = impact_estimate.get("total_impact", 0.0) + slippage_estimate

        if total_cost > order_request.max_slippage:
            return {
                "action": "split_order",
                "reason": "High market impact detected",
                "suggested_algorithm": "TWAP",
                "suggested_time_horizon": order_request.time_horizon * 2,
            }
        elif impact_estimate.get("participation_rate", 0.0) > 0.2:
            return {
                "action": "use_iceberg",
                "reason": "Large order size relative to volume",
                "suggested_algorithm": "ICEBERG",
                "suggested_visible_size": order_request.quantity // 20,
            }
        else:
            return {
                "action": "proceed",
                "reason": "Acceptable market impact",
                "suggested_algorithm": order_request.order_type.value,
            }

    async def _execute_plan(
        self,
        order_request: AdvancedOrderRequest,
        execution_plan: dict[str, Any],
        venue: str,
    ) -> dict[str, Any]:
        """Execute the algorithmic execution plan"""
        try:
            plan_slices = execution_plan.get("execution_plan", [])
            executed_slices = []
            total_filled = 0
            total_value = 0.0

            for slice_info in plan_slices:
                # Execute individual slice
                slice_result = await self._execute_slice(
                    order_request, slice_info, venue
                )

                executed_slices.append(slice_result)

                if slice_result.get("status") == "filled":
                    total_filled += slice_result.get("filled_quantity", 0)
                    total_value += slice_result.get("filled_value", 0.0)

                # Wait between slices if specified
                if "wait_time" in slice_info:
                    await asyncio.sleep(slice_info["wait_time"])

            average_price = total_value / total_filled if total_filled > 0 else 0.0

            return {
                "status": "completed",
                "algorithm": execution_plan.get("algorithm"),
                "total_filled": total_filled,
                "average_price": average_price,
                "executed_slices": executed_slices,
                "execution_summary": {
                    "total_slices": len(plan_slices),
                    "successful_slices": len(
                        [s for s in executed_slices if s.get("status") == "filled"]
                    ),
                    "fill_rate": total_filled / order_request.quantity,
                },
            }

        except Exception as e:
            logger.error(f"Error executing plan: {str(e)}")
            return {"status": "error", "reason": str(e)}

    async def _execute_slice(
        self,
        order_request: AdvancedOrderRequest,
        slice_info: dict[str, Any],
        venue: str,
    ) -> dict[str, Any]:
        """Execute individual order slice"""
        try:
            # This would integrate with the actual order execution system
            # For now, simulate execution

            slice_quantity = slice_info.get("quantity", 0)

            # Simulate execution with some randomness
            fill_probability = 0.95  # 95% fill rate
            if np.random.random() < fill_probability:
                # Simulate price with small slippage
                base_price = 100.0  # Would get from market data
                slippage = np.random.normal(0, 0.001)  # 0.1% std dev
                execution_price = base_price * (1 + slippage)

                return {
                    "status": "filled",
                    "slice_number": slice_info.get("slice_number"),
                    "filled_quantity": slice_quantity,
                    "execution_price": execution_price,
                    "filled_value": slice_quantity * execution_price,
                    "venue": venue,
                    "timestamp": datetime.now(),
                }
            else:
                return {
                    "status": "partial",
                    "slice_number": slice_info.get("slice_number"),
                    "filled_quantity": int(slice_quantity * 0.7),  # Partial fill
                    "venue": venue,
                    "timestamp": datetime.now(),
                }

        except Exception as e:
            return {
                "status": "failed",
                "slice_number": slice_info.get("slice_number"),
                "error": str(e),
                "timestamp": datetime.now(),
            }

    async def _execute_simple_order(
        self, order_request: AdvancedOrderRequest, venue: str
    ) -> dict[str, Any]:
        """Execute simple order types (MARKET, LIMIT, etc.)"""
        try:
            # This would integrate with the existing order management system
            # For now, simulate execution

            execution_price = 100.0  # Would get from market data
            slippage = np.random.normal(0, 0.002)  # 0.2% std dev for simple orders
            final_price = execution_price * (1 + slippage)

            return {
                "status": "filled",
                "order_type": order_request.order_type.value,
                "filled_quantity": order_request.quantity,
                "execution_price": final_price,
                "filled_value": order_request.quantity * final_price,
                "venue": venue,
                "slippage": abs(slippage),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            return {"status": "failed", "error": str(e), "timestamp": datetime.now()}

    async def _check_position_limits(
        self, order_request: AdvancedOrderRequest
    ) -> dict[str, Any]:
        """Check position limits and risk controls"""
        try:
            # This would integrate with the risk management system
            # For now, basic checks

            max_position_value = (
                order_request.max_order_value or 100000
            )  # ₹1 lakh default
            estimated_value = order_request.quantity * 100.0  # Rough estimate

            if estimated_value > max_position_value:
                return {
                    "valid": False,
                    "reason": f"Order value {estimated_value} exceeds limit {max_position_value}",
                }

            return {"valid": True, "reason": "Position limits OK"}

        except Exception as e:
            return {"valid": False, "reason": f"Position check error: {str(e)}"}

    async def _update_execution_metrics(
        self, execution_result: dict[str, Any], execution_time: float
    ):
        """Update execution performance metrics"""
        try:
            self.execution_metrics["total_orders"] += 1

            if execution_result.get("status") in ["filled", "completed"]:
                self.execution_metrics["successful_orders"] += 1

                # Update average execution time
                current_avg = self.execution_metrics["average_execution_time"]
                total_orders = self.execution_metrics["total_orders"]
                self.execution_metrics["average_execution_time"] = (
                    current_avg * (total_orders - 1) + execution_time
                ) / total_orders

                # Update slippage if available
                slippage = execution_result.get("slippage", 0.0)
                if slippage > 0:
                    current_slippage = self.execution_metrics["average_slippage"]
                    self.execution_metrics["average_slippage"] = (
                        current_slippage * (total_orders - 1) + slippage
                    ) / total_orders

            # Store execution history
            self.execution_history.append(
                {
                    "timestamp": datetime.now(),
                    "execution_time": execution_time,
                    "result": execution_result,
                }
            )

            # Keep only last 1000 executions
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]

        except Exception as e:
            logger.error(f"Error updating execution metrics: {str(e)}")

    def get_execution_metrics(self) -> dict[str, Any]:
        """Get current execution performance metrics"""
        success_rate = 0.0
        if self.execution_metrics["total_orders"] > 0:
            success_rate = (
                self.execution_metrics["successful_orders"]
                / self.execution_metrics["total_orders"]
                * 100
            )

        return {
            **self.execution_metrics,
            "success_rate": success_rate,
            "last_updated": datetime.now(),
        }

    async def cancel_advanced_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an advanced order and all its child orders"""
        try:
            if order_id in self.active_orders:
                order_info = self.active_orders[order_id]

                # Cancel all child orders if it's an algorithmic order
                if "execution_plan" in order_info:
                    cancel_results = []
                    for slice_info in order_info["execution_plan"]:
                        if "order_id" in slice_info:
                            cancel_result = await self._cancel_slice_order(
                                slice_info["order_id"]
                            )
                            cancel_results.append(cancel_result)

                    return {
                        "status": "cancelled",
                        "order_id": order_id,
                        "cancelled_slices": len(cancel_results),
                        "cancel_results": cancel_results,
                    }
                else:
                    # Cancel simple order
                    return await self._cancel_simple_order(order_id)
            else:
                return {
                    "status": "not_found",
                    "order_id": order_id,
                    "message": "Order not found in active orders",
                }

        except Exception as e:
            return {"status": "error", "order_id": order_id, "error": str(e)}

    async def _cancel_slice_order(self, slice_order_id: str) -> dict[str, Any]:
        """Cancel individual slice order"""
        # This would integrate with the actual order management system
        return {
            "slice_order_id": slice_order_id,
            "status": "cancelled",
            "timestamp": datetime.now(),
        }

    async def _cancel_simple_order(self, order_id: str) -> dict[str, Any]:
        """Cancel simple order"""
        # This would integrate with the actual order management system
        return {
            "order_id": order_id,
            "status": "cancelled",
            "timestamp": datetime.now(),
        }

    def get_active_orders(self) -> dict[str, Any]:
        """Get all active advanced orders"""
        return {
            "active_orders": list(self.active_orders.keys()),
            "total_active": len(self.active_orders),
            "timestamp": datetime.now(),
        }

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get detailed status of an advanced order"""
        if order_id in self.active_orders:
            return {
                "order_id": order_id,
                "status": "active",
                "details": self.active_orders[order_id],
                "timestamp": datetime.now(),
            }
        else:
            # Check execution history
            for execution in reversed(self.execution_history):
                if execution.get("result", {}).get("order_id") == order_id:
                    return {
                        "order_id": order_id,
                        "status": "completed",
                        "details": execution,
                        "timestamp": datetime.now(),
                    }

            return {
                "order_id": order_id,
                "status": "not_found",
                "timestamp": datetime.now(),
            }
