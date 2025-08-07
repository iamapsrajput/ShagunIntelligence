import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptationType(Enum):
    """Types of strategy adaptations"""

    PARAMETER_UPDATE = "parameter_update"
    WEIGHT_ADJUSTMENT = "weight_adjustment"
    RULE_MODIFICATION = "rule_modification"
    STRATEGY_SWITCH = "strategy_switch"
    AGENT_REBALANCE = "agent_rebalance"


@dataclass
class LearningEvent:
    """Represents a learning event from trading activity"""

    timestamp: datetime
    event_type: str  # decision, execution, outcome
    symbol: str
    decision_data: dict[str, Any]
    outcome_data: dict[str, Any]
    performance_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Adaptation:
    """Represents a strategy adaptation"""

    adaptation_type: AdaptationType
    parameter: str
    old_value: Any
    new_value: Any
    reason: str
    confidence: float
    expected_improvement: float


class LearningManager:
    """Manages learning and adaptation mechanisms for the trading system"""

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate

        # Learning history
        self.learning_events = deque(maxlen=10000)
        self.decision_outcomes: dict[str, list[dict]] = defaultdict(list)

        # Performance tracking
        self.performance_window = 100  # Number of trades to consider
        self.performance_history = deque(maxlen=1000)

        # Strategy performance
        self.strategy_performance = {
            "trend_following": {"wins": 0, "losses": 0, "avg_return": 0},
            "mean_reversion": {"wins": 0, "losses": 0, "avg_return": 0},
            "momentum": {"wins": 0, "losses": 0, "avg_return": 0},
            "sentiment_based": {"wins": 0, "losses": 0, "avg_return": 0},
        }

        # Agent performance tracking
        self.agent_contributions = defaultdict(
            lambda: {
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "contribution_score": 0.5,
            }
        )

        # Adaptation history
        self.adaptation_history = []

        # Learning parameters
        self.adaptation_thresholds = {
            "min_samples": 20,  # Minimum samples before adaptation
            "performance_drop": 0.1,  # 10% performance drop triggers adaptation
            "confidence_threshold": 0.7,  # Minimum confidence for adaptation
            "cooldown_period": 300,  # 5 minutes between adaptations
        }

        self.last_adaptation_time = None

    def record_decision_outcome(
        self,
        decision: dict[str, Any],
        success: bool,
        outcome_data: dict | None = None,
    ) -> None:
        """Record the outcome of a trading decision"""
        event = LearningEvent(
            timestamp=datetime.now(),
            event_type="decision_outcome",
            symbol=decision["symbol"],
            decision_data=decision,
            outcome_data=outcome_data or {"success": success},
        )

        self.learning_events.append(event)

        # Track by opportunity ID
        opp_id = decision.get("opportunity_id", "unknown")
        self.decision_outcomes[opp_id].append(
            {
                "decision": decision,
                "success": success,
                "timestamp": event.timestamp,
                "outcome": outcome_data,
            }
        )

        # Update strategy performance
        self._update_strategy_performance(decision, success, outcome_data)

        # Update agent contributions
        self._update_agent_contributions(decision, success)

        logger.info(
            f"Recorded outcome for {decision['symbol']}: "
            f"{'Success' if success else 'Failure'}"
        )

    def _update_strategy_performance(
        self, decision: dict[str, Any], success: bool, outcome_data: dict | None
    ) -> None:
        """Update performance metrics for strategies"""
        # Determine strategy type from decision metadata
        strategy = self._identify_strategy(decision)

        if strategy in self.strategy_performance:
            perf = self.strategy_performance[strategy]

            if success:
                perf["wins"] += 1
            else:
                perf["losses"] += 1

            # Update average return
            if outcome_data and "return" in outcome_data:
                total_trades = perf["wins"] + perf["losses"]
                current_avg = perf["avg_return"]
                new_return = outcome_data["return"]

                perf["avg_return"] = (
                    current_avg * (total_trades - 1) + new_return
                ) / total_trades

    def _identify_strategy(self, decision: dict[str, Any]) -> str:
        """Identify the primary strategy used in a decision"""
        # Analyze decision factors to determine strategy
        factors = decision.get("source_agents", [])

        if "sentiment" in str(factors).lower():
            return "sentiment_based"
        elif decision.get("risk_metrics", {}).get("trend_strength", 0) > 0.7:
            return "trend_following"
        elif decision.get("expected_return", 0) < 0.01:
            return "mean_reversion"
        else:
            return "momentum"

    def _update_agent_contributions(
        self, decision: dict[str, Any], success: bool
    ) -> None:
        """Update agent contribution scores"""
        source_agents = decision.get("source_agents", [])

        for agent in source_agents:
            contrib = self.agent_contributions[agent]

            if success:
                contrib["correct_predictions"] += 1
            else:
                contrib["incorrect_predictions"] += 1

            # Update contribution score (exponential moving average)
            total = contrib["correct_predictions"] + contrib["incorrect_predictions"]
            if total > 0:
                win_rate = contrib["correct_predictions"] / total
                contrib["contribution_score"] = (
                    contrib["contribution_score"] * (1 - self.learning_rate)
                    + win_rate * self.learning_rate
                )

    def get_adaptation_recommendations(
        self,
        performance: dict[str, Any],
        opportunities: list[Any],
        decisions: list[dict],
    ) -> list[Adaptation]:
        """Get recommendations for strategy adaptations"""
        adaptations = []

        # Check if adaptation is needed
        if not self._should_adapt(performance):
            return adaptations

        # Analyze recent performance
        analysis = self._analyze_performance_patterns()

        # 1. Parameter adaptations
        param_adaptations = self._recommend_parameter_updates(performance, analysis)
        adaptations.extend(param_adaptations)

        # 2. Weight adjustments
        weight_adaptations = self._recommend_weight_adjustments(analysis)
        adaptations.extend(weight_adaptations)

        # 3. Strategy switches
        strategy_adaptations = self._recommend_strategy_changes(analysis)
        adaptations.extend(strategy_adaptations)

        # 4. Agent rebalancing
        agent_adaptations = self._recommend_agent_rebalancing()
        adaptations.extend(agent_adaptations)

        # Filter and prioritize adaptations
        filtered_adaptations = self._filter_adaptations(adaptations)

        # Record adaptations
        for adaptation in filtered_adaptations:
            self.adaptation_history.append(
                {
                    "timestamp": datetime.now(),
                    "adaptation": adaptation,
                    "performance_before": performance,
                }
            )

        self.last_adaptation_time = datetime.now()

        return filtered_adaptations

    def _should_adapt(self, performance: dict[str, Any]) -> bool:
        """Check if adaptation should be triggered"""
        # Check cooldown period
        if self.last_adaptation_time:
            time_since_last = (datetime.now() - self.last_adaptation_time).seconds
            if time_since_last < self.adaptation_thresholds["cooldown_period"]:
                return False

        # Check minimum samples
        if len(self.learning_events) < self.adaptation_thresholds["min_samples"]:
            return False

        # Check performance triggers
        recent_performance = self._calculate_recent_performance()

        # Trigger if significant performance drop
        if "win_rate" in recent_performance and "historical_win_rate" in performance:
            performance_drop = (
                performance["historical_win_rate"] - recent_performance["win_rate"]
            )
            if performance_drop > self.adaptation_thresholds["performance_drop"]:
                return True

        # Other trigger conditions can be added here

        return False

    def _analyze_performance_patterns(self) -> dict[str, Any]:
        """Analyze performance patterns from learning events"""
        recent_events = list(self.learning_events)[-self.performance_window :]

        if not recent_events:
            return {}

        analysis = {
            "total_events": len(recent_events),
            "success_by_hour": defaultdict(lambda: {"success": 0, "total": 0}),
            "success_by_strategy": defaultdict(lambda: {"success": 0, "total": 0}),
            "success_by_confidence": defaultdict(lambda: {"success": 0, "total": 0}),
            "agent_performance": {},
            "failure_patterns": [],
        }

        # Analyze by time of day
        for event in recent_events:
            hour = event.timestamp.hour
            success = event.outcome_data.get("success", False)

            analysis["success_by_hour"][hour]["total"] += 1
            if success:
                analysis["success_by_hour"][hour]["success"] += 1

        # Analyze by confidence level
        for event in recent_events:
            confidence = event.decision_data.get("confidence", 0)
            confidence_bucket = round(confidence, 1)  # Round to nearest 0.1
            success = event.outcome_data.get("success", False)

            analysis["success_by_confidence"][confidence_bucket]["total"] += 1
            if success:
                analysis["success_by_confidence"][confidence_bucket]["success"] += 1

        # Identify failure patterns
        failures = [
            e for e in recent_events if not e.outcome_data.get("success", False)
        ]
        if failures:
            # Common failure characteristics
            failure_patterns = self._identify_failure_patterns(failures)
            analysis["failure_patterns"] = failure_patterns

        return analysis

    def _identify_failure_patterns(self, failures: list[LearningEvent]) -> list[dict]:
        """Identify common patterns in failures"""
        patterns = []

        # Group by symbol
        symbol_failures = defaultdict(list)
        for failure in failures:
            symbol_failures[failure.symbol].append(failure)

        # Identify symbols with high failure rates
        for symbol, events in symbol_failures.items():
            if len(events) >= 3:  # At least 3 failures
                patterns.append(
                    {
                        "type": "symbol_specific",
                        "symbol": symbol,
                        "failure_count": len(events),
                        "recommendation": f"Reduce trading on {symbol}",
                    }
                )

        return patterns

    def _recommend_parameter_updates(
        self, performance: dict[str, Any], analysis: dict[str, Any]
    ) -> list[Adaptation]:
        """Recommend parameter updates based on analysis"""
        adaptations = []

        # Adjust confidence threshold based on confidence-success correlation
        confidence_analysis = analysis.get("success_by_confidence", {})
        if confidence_analysis:
            # Find optimal confidence threshold
            best_confidence = self._find_optimal_confidence(confidence_analysis)

            if best_confidence > 0:
                adaptations.append(
                    Adaptation(
                        adaptation_type=AdaptationType.PARAMETER_UPDATE,
                        parameter="min_confidence",
                        old_value=0.65,  # Current default
                        new_value=best_confidence,
                        reason="Optimize based on confidence-success correlation",
                        confidence=0.8,
                        expected_improvement=0.05,
                    )
                )

        # Adjust position sizing based on recent volatility
        if (
            performance.get("recent_volatility", 0)
            > performance.get("avg_volatility", 0) * 1.5
        ):
            adaptations.append(
                Adaptation(
                    adaptation_type=AdaptationType.PARAMETER_UPDATE,
                    parameter="max_position_size",
                    old_value=0.1,
                    new_value=0.05,  # Reduce position size
                    reason="High market volatility detected",
                    confidence=0.9,
                    expected_improvement=0.03,
                )
            )

        return adaptations

    def _find_optimal_confidence(self, confidence_analysis: dict[float, dict]) -> float:
        """Find optimal confidence threshold"""
        best_confidence = 0
        best_success_rate = 0

        for confidence, stats in confidence_analysis.items():
            if stats["total"] >= 5:  # Minimum samples
                success_rate = stats["success"] / stats["total"]

                # Prefer higher confidence with good success rate
                weighted_score = success_rate * confidence

                if weighted_score > best_success_rate * best_confidence:
                    best_confidence = confidence
                    best_success_rate = success_rate

        return best_confidence

    def _recommend_weight_adjustments(
        self, analysis: dict[str, Any]
    ) -> list[Adaptation]:
        """Recommend adjustments to decision fusion weights"""
        adaptations = []

        # Get agent performance scores
        agent_scores = {}
        for agent, contrib in self.agent_contributions.items():
            agent_scores[agent] = contrib["contribution_score"]

        # Normalize scores to weights
        if agent_scores:
            total_score = sum(agent_scores.values())
            if total_score > 0:
                new_weights = {
                    agent: score / total_score for agent, score in agent_scores.items()
                }

                adaptations.append(
                    Adaptation(
                        adaptation_type=AdaptationType.WEIGHT_ADJUSTMENT,
                        parameter="decision_weights",
                        old_value={"market": 0.3, "technical": 0.4, "sentiment": 0.2},
                        new_value=new_weights,
                        reason="Rebalance based on agent performance",
                        confidence=0.75,
                        expected_improvement=0.04,
                    )
                )

        return adaptations

    def _recommend_strategy_changes(self, analysis: dict[str, Any]) -> list[Adaptation]:
        """Recommend strategy changes based on performance"""
        adaptations = []

        # Analyze strategy performance
        best_strategy = None
        best_performance = -float("inf")

        for strategy, perf in self.strategy_performance.items():
            total_trades = perf["wins"] + perf["losses"]
            if total_trades >= 10:  # Minimum trades
                score = perf["avg_return"] * (perf["wins"] / total_trades)

                if score > best_performance:
                    best_strategy = strategy
                    best_performance = score

        if best_strategy:
            adaptations.append(
                Adaptation(
                    adaptation_type=AdaptationType.STRATEGY_SWITCH,
                    parameter="primary_strategy",
                    old_value="balanced",
                    new_value=best_strategy,
                    reason="Switch to best performing strategy",
                    confidence=0.7,
                    expected_improvement=0.06,
                )
            )

        return adaptations

    def _recommend_agent_rebalancing(self) -> list[Adaptation]:
        """Recommend agent rebalancing based on performance"""
        adaptations = []

        # Identify underperforming agents
        for agent, contrib in self.agent_contributions.items():
            if contrib["contribution_score"] < 0.3:  # Poor performance
                adaptations.append(
                    Adaptation(
                        adaptation_type=AdaptationType.AGENT_REBALANCE,
                        parameter=f"agent_weight_{agent}",
                        old_value=1.0,
                        new_value=0.5,  # Reduce influence
                        reason="Reduce influence of underperforming agent",
                        confidence=0.8,
                        expected_improvement=0.02,
                    )
                )

        return adaptations

    def _filter_adaptations(self, adaptations: list[Adaptation]) -> list[Adaptation]:
        """Filter and prioritize adaptations"""
        # Filter by confidence
        filtered = [
            a
            for a in adaptations
            if a.confidence >= self.adaptation_thresholds["confidence_threshold"]
        ]

        # Sort by expected improvement
        filtered.sort(key=lambda x: x.expected_improvement, reverse=True)

        # Limit number of simultaneous adaptations
        return filtered[:3]

    def _calculate_recent_performance(self) -> dict[str, Any]:
        """Calculate recent performance metrics"""
        recent_events = list(self.learning_events)[-self.performance_window :]

        if not recent_events:
            return {}

        successes = sum(
            1 for e in recent_events if e.outcome_data.get("success", False)
        )
        total = len(recent_events)

        return {
            "win_rate": successes / total if total > 0 else 0,
            "total_trades": total,
            "recent_successes": successes,
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get learning metrics and statistics"""
        return {
            "total_learning_events": len(self.learning_events),
            "adaptations_made": len(self.adaptation_history),
            "strategy_performance": self.strategy_performance,
            "agent_contributions": dict(self.agent_contributions),
            "recent_performance": self._calculate_recent_performance(),
            "last_adaptation": (
                self.last_adaptation_time.isoformat()
                if self.last_adaptation_time
                else None
            ),
        }

    def export_learning_data(self, filepath: str) -> bool:
        """Export learning data for analysis"""
        try:
            data = {
                "learning_events": [
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "event_type": e.event_type,
                        "symbol": e.symbol,
                        "decision_data": e.decision_data,
                        "outcome_data": e.outcome_data,
                    }
                    for e in list(self.learning_events)[-1000:]  # Last 1000 events
                ],
                "adaptation_history": self.adaptation_history,
                "metrics": self.get_metrics(),
                "timestamp": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Learning data exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting learning data: {str(e)}")
            return False
