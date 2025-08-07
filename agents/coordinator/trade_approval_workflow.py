import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from simpleeval import NameNotDefined, simple_eval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Approval status for trading decisions"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    REVIEW_REQUIRED = "review_required"


class RejectionReason(Enum):
    """Reasons for trade rejection"""

    INSUFFICIENT_CAPITAL = "insufficient_capital"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    MARKET_CONDITIONS = "unfavorable_market_conditions"
    LOW_CONFIDENCE = "confidence_below_threshold"
    CORRELATION_RISK = "high_correlation_risk"
    DRAWDOWN_LIMIT = "drawdown_limit_exceeded"
    VOLATILITY_SPIKE = "volatility_spike_detected"
    REGULATORY = "regulatory_restriction"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class ApprovalRule:
    """Represents an approval rule"""

    name: str
    condition: str  # Python expression to evaluate
    priority: int  # Higher priority rules are checked first
    action: ApprovalStatus
    reason: RejectionReason | None = None
    message: str | None = None


class TradeApprovalWorkflow:
    """Manages trade approval workflow with risk validation"""

    def __init__(self):
        # Risk parameters
        self.risk_parameters = {
            "max_position_size": 0.1,  # 10% of capital
            "max_daily_loss": 0.05,  # 5% daily loss limit
            "max_drawdown": 0.15,  # 15% maximum drawdown
            "min_confidence": 0.65,  # 65% minimum confidence
            "max_correlation": 0.7,  # 70% maximum correlation
            "volatility_multiplier": 2.5,  # Max volatility vs average
            "max_concurrent_positions": 10,
            "position_concentration": 0.3,  # Max 30% in single position
        }

        # Approval rules
        self.approval_rules = self._initialize_approval_rules()

        # Approval history
        self.approval_history = []

        # Statistics
        self.approval_stats = {
            "total_decisions": 0,
            "approved": 0,
            "rejected": 0,
            "conditional": 0,
            "rejection_reasons": {},
        }

    def _initialize_approval_rules(self) -> list[ApprovalRule]:
        """Initialize default approval rules"""
        rules = [
            # Critical risk rules (highest priority)
            ApprovalRule(
                name="drawdown_limit",
                condition="context.get('current_drawdown', 0) >= self.risk_parameters['max_drawdown']",
                priority=100,
                action=ApprovalStatus.REJECTED,
                reason=RejectionReason.DRAWDOWN_LIMIT,
            ),
            ApprovalRule(
                name="daily_loss_limit",
                condition="context.get('daily_loss', 0) >= self.risk_parameters['max_daily_loss']",
                priority=95,
                action=ApprovalStatus.REJECTED,
                reason=RejectionReason.RISK_LIMIT_EXCEEDED,
            ),
            # Position and capital rules
            ApprovalRule(
                name="position_size_limit",
                condition="decision['position_size'] > self.risk_parameters['max_position_size']",
                priority=90,
                action=ApprovalStatus.REJECTED,
                reason=RejectionReason.POSITION_LIMIT_EXCEEDED,
            ),
            ApprovalRule(
                name="insufficient_capital",
                condition="decision['allocated_capital'] > context.get('available_capital', 0)",
                priority=85,
                action=ApprovalStatus.REJECTED,
                reason=RejectionReason.INSUFFICIENT_CAPITAL,
            ),
            ApprovalRule(
                name="position_concentration",
                condition="self._check_position_concentration(decision, context)",
                priority=80,
                action=ApprovalStatus.REJECTED,
                reason=RejectionReason.POSITION_LIMIT_EXCEEDED,
                message="Position concentration limit exceeded",
            ),
            # Market condition rules
            ApprovalRule(
                name="volatility_spike",
                condition="context.get('current_volatility', 0) > context.get('avg_volatility', 1) * self.risk_parameters['volatility_multiplier']",
                priority=75,
                action=ApprovalStatus.CONDITIONAL,
                reason=RejectionReason.VOLATILITY_SPIKE,
                message="High volatility detected - reduce position size",
            ),
            # Confidence rules
            ApprovalRule(
                name="low_confidence",
                condition="decision['confidence'] < self.risk_parameters['min_confidence']",
                priority=70,
                action=ApprovalStatus.REJECTED,
                reason=RejectionReason.LOW_CONFIDENCE,
            ),
            # Correlation rules
            ApprovalRule(
                name="high_correlation",
                condition="self._check_correlation_risk(decision, context)",
                priority=65,
                action=ApprovalStatus.CONDITIONAL,
                reason=RejectionReason.CORRELATION_RISK,
                message="High correlation with existing positions",
            ),
            # Default approval rule (lowest priority)
            ApprovalRule(
                name="default_approval",
                condition="True",  # Always true if reached
                priority=0,
                action=ApprovalStatus.APPROVED,
            ),
        ]

        # Sort by priority (descending)
        return sorted(rules, key=lambda r: r.priority, reverse=True)

    def process_decision(
        self, decision: dict[str, Any], risk_manager_agent: Any | None = None
    ) -> dict[str, Any]:
        """Process a trading decision through approval workflow"""
        logger.info(f"Processing trade approval for {decision['symbol']}")

        # Get context for decision
        context = self._get_decision_context(decision, risk_manager_agent)

        # Apply approval rules
        approval_result = self._apply_approval_rules(decision, context)

        # Handle conditional approvals
        if approval_result["status"] == ApprovalStatus.CONDITIONAL:
            approval_result = self._handle_conditional_approval(
                decision, context, approval_result
            )

        # Update decision with approval
        approved_decision = decision.copy()
        approved_decision.update(
            {
                "status": approval_result["status"].value,
                "approval_timestamp": datetime.now().isoformat(),
                "approval_details": approval_result,
            }
        )

        # Add rejection reason if rejected
        if approval_result["status"] == ApprovalStatus.REJECTED:
            approved_decision["rejection_reason"] = approval_result.get(
                "reason", ""
            ).value
            approved_decision["rejection_message"] = approval_result.get("message", "")

        # Record approval
        self._record_approval(approved_decision, approval_result)

        return approved_decision

    def _get_decision_context(
        self, decision: dict[str, Any], risk_manager_agent: Any | None
    ) -> dict[str, Any]:
        """Get context for decision evaluation"""
        context = {
            "timestamp": datetime.now(),
            "available_capital": 100000,  # Would get from portfolio
            "current_positions": [],  # Would get from position monitor
            "daily_pnl": 0,
            "daily_loss": 0,
            "current_drawdown": 0,
            "avg_volatility": 0.02,
            "current_volatility": 0.025,
        }

        # Get risk metrics from risk manager if available
        if risk_manager_agent:
            try:
                risk_metrics = risk_manager_agent.get_portfolio_risk_metrics()
                context.update(
                    {
                        "current_positions": risk_metrics.get("positions", []),
                        "daily_loss": risk_metrics.get("daily_loss_pct", 0),
                        "current_drawdown": risk_metrics.get("current_drawdown", 0),
                        "correlation_matrix": risk_metrics.get(
                            "correlation_matrix", {}
                        ),
                    }
                )
            except Exception as e:
                logger.error(f"Error getting risk metrics: {str(e)}")

        return context

    def _apply_approval_rules(
        self, decision: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply approval rules to decision"""
        for rule in self.approval_rules:
            try:
                # Evaluate rule condition
                # Create safe evaluation context
                eval_context = {"decision": decision, "context": context, "self": self}

                # Evaluate condition using a safe expression evaluator
                try:
                    condition_result = simple_eval(
                        rule.condition,
                        names=eval_context,
                        functions={},
                    )
                except NameNotDefined as e:
                    logger.error(f"Undefined name in rule '{rule.name}': {str(e)}")
                    continue

                if condition_result:
                    logger.info(
                        f"Rule '{rule.name}' triggered with action {rule.action}"
                    )

                    return {
                        "status": rule.action,
                        "rule": rule.name,
                        "reason": rule.reason,
                        "message": rule.message or f"Rule '{rule.name}' triggered",
                    }

            except Exception as e:
                logger.error(f"Error evaluating rule '{rule.name}': {str(e)}")
                continue

        # Should not reach here due to default rule
        return {
            "status": ApprovalStatus.REJECTED,
            "reason": RejectionReason.MANUAL_OVERRIDE,
            "message": "No approval rules matched",
        }

    def _handle_conditional_approval(
        self,
        decision: dict[str, Any],
        context: dict[str, Any],
        approval_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle conditional approvals with adjustments"""
        reason = approval_result.get("reason")

        if reason == RejectionReason.VOLATILITY_SPIKE:
            # Reduce position size during high volatility
            original_size = decision["position_size"]
            decision["position_size"] *= 0.5  # Reduce by 50%
            decision["allocated_capital"] *= 0.5

            approval_result["status"] = ApprovalStatus.APPROVED
            approval_result["adjustments"] = {
                "position_size": {
                    "original": original_size,
                    "adjusted": decision["position_size"],
                },
                "reason": "Reduced due to high volatility",
            }

        elif reason == RejectionReason.CORRELATION_RISK:
            # Reduce position size for correlated positions
            original_size = decision["position_size"]
            decision["position_size"] *= 0.7  # Reduce by 30%
            decision["allocated_capital"] *= 0.7

            approval_result["status"] = ApprovalStatus.APPROVED
            approval_result["adjustments"] = {
                "position_size": {
                    "original": original_size,
                    "adjusted": decision["position_size"],
                },
                "reason": "Reduced due to correlation risk",
            }

        return approval_result

    def _check_position_concentration(
        self, decision: dict[str, Any], context: dict[str, Any]
    ) -> bool:
        """Check if position concentration limit would be exceeded"""
        current_positions = context.get("current_positions", [])
        total_capital = context.get("available_capital", 100000)

        # Calculate current concentration for the symbol
        symbol_exposure = sum(
            pos.get("value", 0)
            for pos in current_positions
            if pos.get("symbol") == decision["symbol"]
        )

        # Add new position
        new_total_exposure = symbol_exposure + decision["allocated_capital"]
        concentration = new_total_exposure / total_capital

        return concentration > self.risk_parameters["position_concentration"]

    def _check_correlation_risk(
        self, decision: dict[str, Any], context: dict[str, Any]
    ) -> bool:
        """Check correlation risk with existing positions"""
        correlation_matrix = context.get("correlation_matrix", {})
        current_positions = context.get("current_positions", [])

        if not correlation_matrix or not current_positions:
            return False

        # Check correlation with each existing position
        for position in current_positions:
            correlation = correlation_matrix.get(decision["symbol"], {}).get(
                position["symbol"], 0
            )

            if abs(correlation) > self.risk_parameters["max_correlation"]:
                return True

        return False

    def _record_approval(
        self, decision: dict[str, Any], approval_result: dict[str, Any]
    ) -> None:
        """Record approval decision for audit"""
        self.approval_history.append(
            {
                "timestamp": datetime.now(),
                "symbol": decision["symbol"],
                "action": decision["action"],
                "confidence": decision["confidence"],
                "status": approval_result["status"].value,
                "rule": approval_result.get("rule"),
                "reason": (
                    approval_result.get("reason", "").value
                    if approval_result.get("reason")
                    else None
                ),
                "adjustments": approval_result.get("adjustments"),
            }
        )

        # Update statistics
        self.approval_stats["total_decisions"] += 1

        status = approval_result["status"]
        if status == ApprovalStatus.APPROVED:
            self.approval_stats["approved"] += 1
        elif status == ApprovalStatus.REJECTED:
            self.approval_stats["rejected"] += 1

            # Track rejection reasons
            reason = approval_result.get("reason")
            if reason:
                reason_str = reason.value
                self.approval_stats["rejection_reasons"][reason_str] = (
                    self.approval_stats["rejection_reasons"].get(reason_str, 0) + 1
                )
        elif status == ApprovalStatus.CONDITIONAL:
            self.approval_stats["conditional"] += 1

        # Keep only recent history
        if len(self.approval_history) > 1000:
            self.approval_history = self.approval_history[-1000:]

    def add_custom_rule(self, rule: ApprovalRule) -> None:
        """Add a custom approval rule"""
        self.approval_rules.append(rule)
        # Re-sort by priority
        self.approval_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added custom rule: {rule.name}")

    def update_risk_parameters(self, parameters: dict[str, Any]) -> None:
        """Update risk parameters"""
        self.risk_parameters.update(parameters)
        logger.info(f"Updated risk parameters: {parameters}")

    def get_risk_parameters(self) -> dict[str, Any]:
        """Get current risk parameters"""
        return self.risk_parameters.copy()

    def get_approval_statistics(self) -> dict[str, Any]:
        """Get approval workflow statistics"""
        stats = self.approval_stats.copy()

        if stats["total_decisions"] > 0:
            stats["approval_rate"] = (
                stats["approved"] / stats["total_decisions"]
            ) * 100
            stats["rejection_rate"] = (
                stats["rejected"] / stats["total_decisions"]
            ) * 100

        return stats

    def get_recent_approvals(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent approval decisions"""
        return self.approval_history[-limit:]

    def export_approval_log(self, filepath: str) -> bool:
        """Export approval log to file"""
        try:
            with open(filepath, "w") as f:
                json.dump(
                    {
                        "approval_history": self.approval_history,
                        "statistics": self.get_approval_statistics(),
                        "risk_parameters": self.risk_parameters,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    default=str,
                )

            logger.info(f"Approval log exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting approval log: {str(e)}")
            return False
