from crewai import Agent, Task, Crew, Process
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import json

from .decision_fusion_engine import DecisionFusionEngine
from .task_delegator import TaskDelegator
from .trade_approval_workflow import TradeApprovalWorkflow
from .learning_manager import LearningManager
from .performance_monitor import PerformanceMonitor
from .communication_hub import CommunicationHub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of specialist agents"""
    MARKET_ANALYST = "market_analyst"
    TECHNICAL_INDICATOR = "technical_indicator"
    SENTIMENT_ANALYST = "sentiment_analyst"
    RISK_MANAGER = "risk_manager"
    TRADE_EXECUTOR = "trade_executor"
    DATA_PROCESSOR = "data_processor"


@dataclass
class TradingOpportunity:
    """Represents a potential trading opportunity"""
    id: str
    symbol: str
    action: str  # BUY or SELL
    confidence: float
    expected_return: float
    risk_score: float
    priority: float
    source_agents: List[str]
    analysis: Dict[str, Any]
    timestamp: datetime


class CoordinatorAgent(Agent):
    """Master coordinator agent that orchestrates all other agents"""
    
    def __init__(self, agents: Dict[AgentType, Agent], config: Dict[str, Any] = None):
        super().__init__(
            name="Coordinator",
            role="Master orchestrator and decision maker",
            goal="Coordinate all agents to make optimal trading decisions while managing risk",
            backstory="""You are the chief trading strategist with decades of experience
            in financial markets. You excel at synthesizing diverse information sources,
            managing teams of specialists, and making decisive trading decisions. You
            balance aggressive profit-seeking with prudent risk management.""",
            verbose=True,
            allow_delegation=True,
            max_iter=10
        )
        
        self.agents = agents
        self.config = config or {}
        
        # Initialize components
        self.decision_engine = DecisionFusionEngine()
        self.task_delegator = TaskDelegator(agents)
        self.approval_workflow = TradeApprovalWorkflow()
        self.learning_manager = LearningManager()
        self.performance_monitor = PerformanceMonitor()
        self.communication_hub = CommunicationHub()
        
        # Trading state
        self.active_opportunities: List[TradingOpportunity] = []
        self.pending_decisions: List[Dict[str, Any]] = []
        self.execution_queue: List[Dict[str, Any]] = []
        
        # Configuration
        self.max_concurrent_trades = config.get("max_concurrent_trades", 5)
        self.min_confidence_threshold = config.get("min_confidence", 0.7)
        self.decision_interval = config.get("decision_interval", 60)  # seconds
        
    def analyze_market(self, symbols: List[str]) -> List[TradingOpportunity]:
        """Orchestrate market analysis across all agents"""
        logger.info(f"Starting market analysis for {len(symbols)} symbols")
        
        # Create analysis tasks for different agents
        analysis_tasks = []
        
        # Market analysis task
        if AgentType.MARKET_ANALYST in self.agents:
            task = self.task_delegator.create_task(
                AgentType.MARKET_ANALYST,
                "analyze_market_conditions",
                {"symbols": symbols}
            )
            analysis_tasks.append(task)
        
        # Technical analysis task
        if AgentType.TECHNICAL_INDICATOR in self.agents:
            task = self.task_delegator.create_task(
                AgentType.TECHNICAL_INDICATOR,
                "analyze_technical_indicators",
                {"symbols": symbols, "timeframes": ["5min", "15min", "1hour"]}
            )
            analysis_tasks.append(task)
        
        # Sentiment analysis task
        if AgentType.SENTIMENT_ANALYST in self.agents:
            task = self.task_delegator.create_task(
                AgentType.SENTIMENT_ANALYST,
                "analyze_market_sentiment",
                {"symbols": symbols}
            )
            analysis_tasks.append(task)
        
        # Execute tasks in parallel
        results = self.task_delegator.execute_parallel_tasks(analysis_tasks)
        
        # Aggregate and process results
        opportunities = self._process_analysis_results(results, symbols)
        
        # Rank opportunities by priority
        self.active_opportunities = self._rank_opportunities(opportunities)
        
        return self.active_opportunities
    
    def _process_analysis_results(self, results: List[Dict[str, Any]], 
                                 symbols: List[str]) -> List[TradingOpportunity]:
        """Process analysis results from different agents"""
        opportunities = []
        
        # Group results by symbol
        symbol_analysis = {symbol: {} for symbol in symbols}
        
        for result in results:
            if result["status"] == "success":
                agent_type = result["agent_type"]
                data = result["data"]
                
                # Process based on agent type
                if agent_type == AgentType.MARKET_ANALYST:
                    for symbol, analysis in data.get("symbol_analysis", {}).items():
                        symbol_analysis[symbol]["market"] = analysis
                        
                elif agent_type == AgentType.TECHNICAL_INDICATOR:
                    for symbol, indicators in data.get("indicators", {}).items():
                        symbol_analysis[symbol]["technical"] = indicators
                        
                elif agent_type == AgentType.SENTIMENT_ANALYST:
                    for symbol, sentiment in data.get("sentiment", {}).items():
                        symbol_analysis[symbol]["sentiment"] = sentiment
        
        # Use decision fusion engine to create opportunities
        for symbol, analysis in symbol_analysis.items():
            if self._has_sufficient_data(analysis):
                opportunity = self.decision_engine.fuse_insights(
                    symbol=symbol,
                    market_analysis=analysis.get("market", {}),
                    technical_analysis=analysis.get("technical", {}),
                    sentiment_analysis=analysis.get("sentiment", {})
                )
                
                if opportunity and opportunity.confidence >= self.min_confidence_threshold:
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _has_sufficient_data(self, analysis: Dict[str, Any]) -> bool:
        """Check if we have sufficient data for decision making"""
        required_components = ["market", "technical"]
        return all(comp in analysis and analysis[comp] for comp in required_components)
    
    def _rank_opportunities(self, opportunities: List[TradingOpportunity]) -> List[TradingOpportunity]:
        """Rank opportunities by priority"""
        # Calculate priority score for each opportunity
        for opp in opportunities:
            # Priority factors:
            # 1. Confidence (40%)
            # 2. Expected return (30%)
            # 3. Risk score (20%)
            # 4. Number of confirming agents (10%)
            
            confidence_score = opp.confidence * 0.4
            return_score = min(opp.expected_return / 10, 1.0) * 0.3  # Cap at 10%
            risk_score = (1 - opp.risk_score) * 0.2  # Lower risk is better
            agent_score = (len(opp.source_agents) / len(self.agents)) * 0.1
            
            opp.priority = confidence_score + return_score + risk_score + agent_score
        
        # Sort by priority (descending)
        return sorted(opportunities, key=lambda x: x.priority, reverse=True)
    
    def make_trading_decisions(self) -> List[Dict[str, Any]]:
        """Make final trading decisions based on opportunities"""
        decisions = []
        
        # Get current positions from risk manager
        current_positions = self._get_current_positions()
        available_capital = self._get_available_capital()
        
        for opportunity in self.active_opportunities:
            # Skip if we've reached max concurrent trades
            if len(current_positions) >= self.max_concurrent_trades:
                logger.info(f"Max concurrent trades reached, skipping {opportunity.symbol}")
                break
            
            # Validate with risk manager
            risk_validation = self._validate_with_risk_manager(opportunity)
            
            if risk_validation["approved"]:
                # Create trading decision
                decision = self._create_trading_decision(
                    opportunity,
                    risk_validation,
                    available_capital
                )
                
                # Run through approval workflow
                approved_decision = self.approval_workflow.process_decision(
                    decision,
                    self.agents.get(AgentType.RISK_MANAGER)
                )
                
                if approved_decision["status"] == "approved":
                    decisions.append(approved_decision)
                    
                    # Update available capital
                    available_capital -= approved_decision["allocated_capital"]
                    
                    # Log decision
                    self.communication_hub.broadcast_decision(approved_decision)
                else:
                    logger.warning(f"Decision rejected for {opportunity.symbol}: "
                                 f"{approved_decision.get('rejection_reason')}")
            else:
                logger.info(f"Risk validation failed for {opportunity.symbol}: "
                           f"{risk_validation.get('reason')}")
        
        self.pending_decisions = decisions
        return decisions
    
    def _validate_with_risk_manager(self, opportunity: TradingOpportunity) -> Dict[str, Any]:
        """Validate opportunity with risk manager"""
        if AgentType.RISK_MANAGER not in self.agents:
            return {"approved": True, "risk_metrics": {}}
        
        task = self.task_delegator.create_task(
            AgentType.RISK_MANAGER,
            "evaluate_trade_risk",
            {
                "symbol": opportunity.symbol,
                "action": opportunity.action,
                "expected_return": opportunity.expected_return,
                "analysis": opportunity.analysis
            }
        )
        
        result = self.task_delegator.execute_task(task)
        
        if result["status"] == "success":
            risk_data = result["data"]
            return {
                "approved": risk_data.get("risk_acceptable", False),
                "risk_metrics": risk_data.get("metrics", {}),
                "position_size": risk_data.get("recommended_position_size"),
                "stop_loss": risk_data.get("stop_loss"),
                "reason": risk_data.get("reason", "")
            }
        
        return {"approved": False, "reason": "Risk evaluation failed"}
    
    def _create_trading_decision(self, opportunity: TradingOpportunity,
                               risk_validation: Dict[str, Any],
                               available_capital: float) -> Dict[str, Any]:
        """Create a complete trading decision"""
        position_size = risk_validation.get("position_size", 0.02)  # Default 2%
        allocated_capital = available_capital * position_size
        
        return {
            "opportunity_id": opportunity.id,
            "symbol": opportunity.symbol,
            "action": opportunity.action,
            "confidence": opportunity.confidence,
            "expected_return": opportunity.expected_return,
            "priority": opportunity.priority,
            "allocated_capital": allocated_capital,
            "position_size": position_size,
            "stop_loss": risk_validation.get("stop_loss"),
            "target": opportunity.expected_return * 1.5,  # 1.5x expected return
            "risk_metrics": risk_validation.get("risk_metrics", {}),
            "source_agents": opportunity.source_agents,
            "timestamp": datetime.now(),
            "status": "pending"
        }
    
    def execute_decisions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute approved trading decisions"""
        execution_results = []
        
        if AgentType.TRADE_EXECUTOR not in self.agents:
            logger.error("Trade executor agent not available")
            return []
        
        for decision in decisions:
            # Create execution task
            task = self.task_delegator.create_task(
                AgentType.TRADE_EXECUTOR,
                "execute_trade",
                {
                    "symbol": decision["symbol"],
                    "action": decision["action"],
                    "quantity": self._calculate_quantity(
                        decision["symbol"],
                        decision["allocated_capital"]
                    ),
                    "stop_loss": decision["stop_loss"],
                    "target": decision["target"],
                    "order_type": "LIMIT",  # Can be made configurable
                    "metadata": {
                        "opportunity_id": decision["opportunity_id"],
                        "confidence": decision["confidence"]
                    }
                }
            )
            
            # Execute trade
            result = self.task_delegator.execute_task(task)
            
            # Process result
            execution_result = {
                "decision": decision,
                "execution": result,
                "timestamp": datetime.now()
            }
            
            execution_results.append(execution_result)
            
            # Update learning manager with result
            self.learning_manager.record_decision_outcome(
                decision,
                result["status"] == "success"
            )
            
            # Broadcast execution result
            self.communication_hub.broadcast_execution(execution_result)
        
        return execution_results
    
    def monitor_and_adapt(self) -> Dict[str, Any]:
        """Monitor performance and adapt strategies"""
        # Get performance metrics
        performance = self.performance_monitor.get_current_performance()
        
        # Check if adaptation is needed
        if self._should_adapt(performance):
            # Get adaptation recommendations
            adaptations = self.learning_manager.get_adaptation_recommendations(
                performance,
                self.active_opportunities,
                self.pending_decisions
            )
            
            # Apply adaptations
            self._apply_adaptations(adaptations)
            
            logger.info(f"Applied {len(adaptations)} strategy adaptations")
            
            return {
                "adapted": True,
                "adaptations": adaptations,
                "new_parameters": self._get_current_parameters()
            }
        
        return {"adapted": False, "reason": "Performance within acceptable range"}
    
    def _should_adapt(self, performance: Dict[str, Any]) -> bool:
        """Determine if strategy adaptation is needed"""
        # Adapt if:
        # 1. Win rate drops below 40%
        # 2. Consecutive losses exceed 5
        # 3. Drawdown exceeds 10%
        # 4. Sharpe ratio below 0.5
        
        win_rate = performance.get("win_rate", 100)
        consecutive_losses = performance.get("consecutive_losses", 0)
        drawdown = performance.get("max_drawdown", 0)
        sharpe_ratio = performance.get("sharpe_ratio", 1.0)
        
        return (win_rate < 40 or 
                consecutive_losses > 5 or 
                drawdown > 0.1 or 
                sharpe_ratio < 0.5)
    
    def _apply_adaptations(self, adaptations: List[Dict[str, Any]]) -> None:
        """Apply strategy adaptations"""
        for adaptation in adaptations:
            param = adaptation["parameter"]
            new_value = adaptation["new_value"]
            
            if param == "min_confidence":
                self.min_confidence_threshold = new_value
            elif param == "max_concurrent_trades":
                self.max_concurrent_trades = new_value
            elif param == "risk_parameters":
                self._update_risk_parameters(new_value)
            elif param == "decision_weights":
                self.decision_engine.update_weights(new_value)
    
    def _get_current_positions(self) -> List[Dict[str, Any]]:
        """Get current open positions"""
        if AgentType.TRADE_EXECUTOR in self.agents:
            # This would call the trade executor's position monitor
            return []
        return []
    
    def _get_available_capital(self) -> float:
        """Get available capital for trading"""
        # This would integrate with portfolio management
        return self.config.get("trading_capital", 100000)
    
    def _calculate_quantity(self, symbol: str, capital: float) -> int:
        """Calculate order quantity based on allocated capital"""
        # This would use real-time price data
        # For now, simple calculation
        assumed_price = 100  # Would fetch real price
        return int(capital / assumed_price)
    
    def _update_risk_parameters(self, new_params: Dict[str, Any]) -> None:
        """Update risk management parameters"""
        if AgentType.RISK_MANAGER in self.agents:
            # This would update the risk manager's parameters
            pass
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters"""
        return {
            "min_confidence_threshold": self.min_confidence_threshold,
            "max_concurrent_trades": self.max_concurrent_trades,
            "decision_weights": self.decision_engine.get_weights(),
            "risk_parameters": self.approval_workflow.get_risk_parameters()
        }
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        return {
            "active_opportunities": len(self.active_opportunities),
            "pending_decisions": len(self.pending_decisions),
            "current_positions": len(self._get_current_positions()),
            "performance": self.performance_monitor.get_current_performance(),
            "agent_status": self.task_delegator.get_agent_status(),
            "learning_metrics": self.learning_manager.get_metrics(),
            "parameters": self._get_current_parameters(),
            "timestamp": datetime.now().isoformat()
        }