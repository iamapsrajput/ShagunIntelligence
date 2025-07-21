from crewai import Agent, Task, Crew, Process
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import json
from loguru import logger as loguru_logger
import asyncio

from .decision_fusion_engine import DecisionFusionEngine
from .task_delegator import TaskDelegator
from .trade_approval_workflow import TradeApprovalWorkflow
from .learning_manager import LearningManager
from .performance_monitor import PerformanceMonitor
from .communication_hub import CommunicationHub
from ..base_quality_aware_agent import BaseQualityAwareAgent, DataQualityLevel, TradingMode
from backend.data_sources.integration import get_data_source_integration

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
    """Represents a potential trading opportunity with data quality metrics"""
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
    data_quality_score: float = 1.0
    quality_level: str = "high"
    multi_source_consensus: float = 0.0


class CoordinatorAgent(BaseQualityAwareAgent, Agent):
    """Master coordinator agent that orchestrates multi-source data collection and quality-aware decisions"""
    
    def __init__(self, agents: Dict[AgentType, Agent], config: Dict[str, Any] = None):
        BaseQualityAwareAgent.__init__(self)
        Agent.__init__(self,
            name="Quality-Aware Coordinator",
            role="Orchestrate multi-source data collection and quality-based agent coordination",
            goal="Ensure all agents work with high-quality data and make coordinated decisions",
            backstory="""You are the master coordinator who understands that successful
            trading depends on data quality. You orchestrate:
            1. Multi-source data collection for each trading decision
            2. Quality assessment before delegating tasks to specialist agents
            3. Consensus building from multiple data sources
            4. Priority adjustment based on data reliability
            You ensure all agents receive quality-validated data and adjust their
            operations based on current data quality levels.""",
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
        
        # Quality-aware settings
        self.min_data_quality_for_analysis = 0.6  # Minimum quality to proceed
        self.quality_confidence_weights = {
            DataQualityLevel.HIGH: 1.0,
            DataQualityLevel.MEDIUM: 0.7,
            DataQualityLevel.LOW: 0.3,
            DataQualityLevel.CRITICAL: 0.0
        }
        
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
        # Calculate quality distribution of opportunities
        quality_distribution = {}
        for opp in self.active_opportunities:
            level = getattr(opp, 'quality_level', 'unknown')
            quality_distribution[level] = quality_distribution.get(level, 0) + 1
        
        return {
            "active_opportunities": len(self.active_opportunities),
            "pending_decisions": len(self.pending_decisions),
            "current_positions": len(self._get_current_positions()),
            "performance": self.performance_monitor.get_current_performance(),
            "agent_status": self.task_delegator.get_agent_status(),
            "learning_metrics": self.learning_manager.get_metrics(),
            "parameters": self._get_current_parameters(),
            "quality_metrics": {
                "min_quality_threshold": self.min_data_quality_for_analysis,
                "opportunity_quality_distribution": quality_distribution,
                "quality_weights": {
                    k.value: v for k, v in self.quality_confidence_weights.items()
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def analyze_market_quality_aware(self, symbols: List[str]) -> List[TradingOpportunity]:
        """Orchestrate quality-aware market analysis with multi-source data collection."""
        logger.info(f"Starting quality-aware market analysis for {len(symbols)} symbols")
        
        # First, assess data quality for all symbols
        quality_assessments = {}
        for symbol in symbols:
            try:
                # Get quality assessment and consensus data
                quote_data, quality_score, quality_level = await self.get_quality_weighted_data(
                    symbol, "quote"
                )
                consensus_data, consensus_confidence = await self.get_multi_source_consensus(symbol)
                
                quality_assessments[symbol] = {
                    'quality_score': quality_score,
                    'quality_level': quality_level,
                    'consensus_confidence': consensus_confidence,
                    'trading_mode': self.get_trading_mode(quality_level),
                    'has_consensus': consensus_confidence > 0.5
                }
                
                logger.info(
                    f"{symbol}: Quality {quality_score:.1%}, Level: {quality_level.value}, "
                    f"Consensus: {consensus_confidence:.1%}"
                )
            except Exception as e:
                logger.error(f"Error assessing quality for {symbol}: {e}")
                quality_assessments[symbol] = {
                    'quality_score': 0.0,
                    'quality_level': DataQualityLevel.CRITICAL,
                    'consensus_confidence': 0.0,
                    'trading_mode': TradingMode.EXIT_ONLY,
                    'has_consensus': False
                }
        
        # Filter symbols based on minimum quality requirements
        tradeable_symbols = [
            symbol for symbol in symbols
            if quality_assessments[symbol]['quality_score'] >= self.min_data_quality_for_analysis
        ]
        
        if not tradeable_symbols:
            logger.warning("No symbols meet minimum quality requirements for analysis")
            return []
        
        # Create quality-aware analysis tasks
        analysis_tasks = self._create_quality_aware_tasks(tradeable_symbols, quality_assessments)
        
        # Execute tasks in parallel
        results = self.task_delegator.execute_parallel_tasks(analysis_tasks)
        
        # Process results with quality weighting
        opportunities = self._process_quality_aware_results(results, quality_assessments)
        
        # Rank opportunities with quality considerations
        self.active_opportunities = self._rank_quality_aware_opportunities(opportunities)
        
        logger.info(
            f"Quality-aware analysis complete: {len(self.active_opportunities)} opportunities found "
            f"from {len(tradeable_symbols)} tradeable symbols"
        )
        
        return self.active_opportunities
    
    def _create_quality_aware_tasks(
        self, 
        symbols: List[str], 
        quality_assessments: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create analysis tasks with quality-based parameters."""
        tasks = []
        
        # Group symbols by quality level for efficient processing
        quality_groups = {}
        for symbol in symbols:
            level = quality_assessments[symbol]['quality_level']
            if level not in quality_groups:
                quality_groups[level] = []
            quality_groups[level].append(symbol)
        
        # Create tasks based on quality levels
        for quality_level, group_symbols in quality_groups.items():
            # Market analysis task
            if AgentType.MARKET_ANALYST in self.agents:
                task = self.task_delegator.create_task(
                    AgentType.MARKET_ANALYST,
                    "analyze_with_quality",  # Quality-aware method
                    {
                        "symbols": group_symbols,
                        "quality_level": quality_level.value,
                        "min_confidence": self.quality_confidence_weights[quality_level]
                    }
                )
                tasks.append(task)
            
            # Technical analysis with quality-appropriate indicators
            if AgentType.TECHNICAL_INDICATOR in self.agents:
                # Adjust timeframes based on quality
                if quality_level == DataQualityLevel.HIGH:
                    timeframes = ["1min", "5min", "15min", "1hour"]
                elif quality_level == DataQualityLevel.MEDIUM:
                    timeframes = ["5min", "15min", "1hour"]  # Skip 1min for noise
                else:
                    timeframes = ["15min", "1hour"]  # Only longer timeframes
                
                task = self.task_delegator.create_task(
                    AgentType.TECHNICAL_INDICATOR,
                    "analyze_symbol",
                    {
                        "symbols": group_symbols,
                        "timeframes": timeframes,
                        "quality_aware": True
                    }
                )
                tasks.append(task)
            
            # Sentiment analysis only for medium/high quality
            if (AgentType.SENTIMENT_ANALYST in self.agents and 
                quality_level in [DataQualityLevel.HIGH, DataQualityLevel.MEDIUM]):
                task = self.task_delegator.create_task(
                    AgentType.SENTIMENT_ANALYST,
                    "analyze_sentiment_multi_source",
                    {
                        "symbols": group_symbols,
                        "require_consensus": quality_level == DataQualityLevel.MEDIUM
                    }
                )
                tasks.append(task)
        
        return tasks
    
    def _process_quality_aware_results(
        self,
        results: List[Dict[str, Any]],
        quality_assessments: Dict[str, Dict[str, Any]]
    ) -> List[TradingOpportunity]:
        """Process analysis results with quality weighting."""
        opportunities = []
        symbol_analyses = {}
        
        # Aggregate results by symbol
        for result in results:
            if result["status"] == "success":
                agent_type = result["agent_type"]
                data = result["data"]
                
                # Process based on agent type
                if isinstance(data, dict) and "symbol_analysis" in data:
                    for symbol, analysis in data["symbol_analysis"].items():
                        if symbol not in symbol_analyses:
                            symbol_analyses[symbol] = {}
                        symbol_analyses[symbol][agent_type.value] = analysis
        
        # Create opportunities with quality adjustment
        for symbol, analyses in symbol_analyses.items():
            if symbol not in quality_assessments:
                continue
                
            quality_info = quality_assessments[symbol]
            
            # Use decision fusion with quality weighting
            opportunity = self._create_quality_weighted_opportunity(
                symbol=symbol,
                analyses=analyses,
                quality_info=quality_info
            )
            
            if opportunity:
                opportunities.append(opportunity)
        
        return opportunities
    
    def _create_quality_weighted_opportunity(
        self,
        symbol: str,
        analyses: Dict[str, Any],
        quality_info: Dict[str, Any]
    ) -> Optional[TradingOpportunity]:
        """Create trading opportunity with quality-weighted confidence."""
        try:
            # Extract insights from different agents
            market_analysis = analyses.get(AgentType.MARKET_ANALYST.value, {})
            technical_analysis = analyses.get(AgentType.TECHNICAL_INDICATOR.value, {})
            sentiment_analysis = analyses.get(AgentType.SENTIMENT_ANALYST.value, {})
            
            # Base confidence from analyses
            market_confidence = market_analysis.get('confidence', 0.5)
            technical_confidence = technical_analysis.get('confidence', 0.5)
            sentiment_confidence = sentiment_analysis.get('confidence', 0.5)
            
            # Weight by data quality
            quality_weight = self.quality_confidence_weights[quality_info['quality_level']]
            
            # Calculate weighted confidence
            if sentiment_analysis:  # All three analyses available
                weighted_confidence = (
                    market_confidence * 0.4 + 
                    technical_confidence * 0.4 + 
                    sentiment_confidence * 0.2
                ) * quality_weight
            else:  # Only market and technical
                weighted_confidence = (
                    market_confidence * 0.6 + 
                    technical_confidence * 0.4
                ) * quality_weight
            
            # Determine action
            action = self._determine_action(market_analysis, technical_analysis)
            if not action:
                return None
            
            # Create opportunity
            opportunity = TradingOpportunity(
                id=f"{symbol}_{datetime.now().timestamp()}",
                symbol=symbol,
                action=action,
                confidence=weighted_confidence,
                expected_return=market_analysis.get('expected_return', 0.0),
                risk_score=market_analysis.get('risk_score', 0.5),
                priority=0.0,  # Will be set in ranking
                source_agents=list(analyses.keys()),
                analysis={
                    'market': market_analysis,
                    'technical': technical_analysis,
                    'sentiment': sentiment_analysis
                },
                timestamp=datetime.now(),
                data_quality_score=quality_info['quality_score'],
                quality_level=quality_info['quality_level'].value,
                multi_source_consensus=quality_info['consensus_confidence']
            )
            
            # Apply minimum confidence threshold with quality adjustment
            adjusted_threshold = self.min_confidence_threshold * max(0.8, quality_weight)
            
            if opportunity.confidence >= adjusted_threshold:
                return opportunity
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating quality-weighted opportunity for {symbol}: {e}")
            return None
    
    def _determine_action(self, market_analysis: Dict, technical_analysis: Dict) -> Optional[str]:
        """Determine trading action from analyses."""
        market_signal = market_analysis.get('signal', 'NEUTRAL')
        technical_signal = technical_analysis.get('signal', 'NEUTRAL')
        
        # Both must agree for action
        if market_signal == technical_signal and market_signal in ['BUY', 'SELL']:
            return market_signal
        
        return None
    
    def _rank_quality_aware_opportunities(
        self, 
        opportunities: List[TradingOpportunity]
    ) -> List[TradingOpportunity]:
        """Rank opportunities with quality as a primary factor."""
        for opp in opportunities:
            # Priority calculation with quality emphasis:
            # 1. Data quality (30%)
            # 2. Confidence (25%)
            # 3. Expected return (20%)
            # 4. Risk score (15%)
            # 5. Multi-source consensus (10%)
            
            quality_score = opp.data_quality_score * 0.3
            confidence_score = opp.confidence * 0.25
            return_score = min(opp.expected_return / 10, 1.0) * 0.2
            risk_score = (1 - opp.risk_score) * 0.15
            consensus_score = opp.multi_source_consensus * 0.1
            
            opp.priority = (
                quality_score + confidence_score + return_score + 
                risk_score + consensus_score
            )
        
        # Sort by priority (descending)
        return sorted(opportunities, key=lambda x: x.priority, reverse=True)
    
    async def make_quality_aware_trading_decisions(self) -> List[Dict[str, Any]]:
        """Make trading decisions with data quality as primary consideration."""
        decisions = []
        
        # Get current positions and capital
        current_positions = self._get_current_positions()
        available_capital = self._get_available_capital()
        
        for opportunity in self.active_opportunities:
            # Skip if quality is too low
            if opportunity.data_quality_score < self.min_data_quality_for_analysis:
                logger.info(
                    f"Skipping {opportunity.symbol} - data quality {opportunity.data_quality_score:.1%} "
                    f"below minimum {self.min_data_quality_for_analysis:.1%}"
                )
                continue
            
            # Skip if we've reached max concurrent trades
            if len(current_positions) >= self.max_concurrent_trades:
                logger.info(f"Max concurrent trades reached, skipping {opportunity.symbol}")
                break
            
            # Quality-aware risk validation
            risk_validation = await self._validate_with_quality_aware_risk_manager(opportunity)
            
            if risk_validation["approved"]:
                # Create trading decision with quality metadata
                decision = self._create_quality_aware_decision(
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
                    available_capital -= approved_decision["allocated_capital"]
                    
                    # Log quality-aware decision
                    logger.info(
                        f"Approved trade for {opportunity.symbol} with quality {opportunity.quality_level} "
                        f"({opportunity.data_quality_score:.1%})"
                    )
                    
                    self.communication_hub.broadcast_decision(approved_decision)
                else:
                    logger.warning(
                        f"Decision rejected for {opportunity.symbol}: "
                        f"{approved_decision.get('rejection_reason')}"
                    )
            else:
                logger.info(
                    f"Risk validation failed for {opportunity.symbol}: "
                    f"{risk_validation.get('reason')}"
                )
        
        self.pending_decisions = decisions
        return decisions
    
    async def _validate_with_quality_aware_risk_manager(
        self, 
        opportunity: TradingOpportunity
    ) -> Dict[str, Any]:
        """Validate opportunity with quality-aware risk manager."""
        if AgentType.RISK_MANAGER not in self.agents:
            return {"approved": True, "risk_metrics": {}}
        
        # Include quality information in risk evaluation
        task = self.task_delegator.create_task(
            AgentType.RISK_MANAGER,
            "evaluate_trade_risk",
            {
                "symbol": opportunity.symbol,
                "action": opportunity.action,
                "expected_return": opportunity.expected_return,
                "analysis": opportunity.analysis,
                "data_quality_score": opportunity.data_quality_score,
                "quality_level": opportunity.quality_level,
                "confidence": opportunity.confidence
            }
        )
        
        result = self.task_delegator.execute_task(task)
        
        if result["status"] == "success":
            risk_data = result["data"]
            return {
                "approved": risk_data.get("risk_acceptable", False),
                "risk_metrics": risk_data.get("metrics", {}),
                "position_size": risk_data.get("recommended_position_size"),
                "quality_adjusted_size": risk_data.get("quality_adjusted_position_size"),
                "stop_loss": risk_data.get("stop_loss"),
                "reason": risk_data.get("reason", "")
            }
        
        return {"approved": False, "reason": "Risk evaluation failed"}
    
    def _create_quality_aware_decision(
        self,
        opportunity: TradingOpportunity,
        risk_validation: Dict[str, Any],
        available_capital: float
    ) -> Dict[str, Any]:
        """Create trading decision with quality adjustments."""
        # Use quality-adjusted position size if available
        position_size = risk_validation.get(
            "quality_adjusted_size", 
            risk_validation.get("position_size", 0.02)
        )
        
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
            "target": opportunity.expected_return * 1.5,
            "risk_metrics": risk_validation.get("risk_metrics", {}),
            "source_agents": opportunity.source_agents,
            "data_quality": {
                "score": opportunity.data_quality_score,
                "level": opportunity.quality_level,
                "consensus": opportunity.multi_source_consensus,
                "trading_mode": self.get_trading_mode(
                    DataQualityLevel[opportunity.quality_level.upper()]
                ).value
            },
            "timestamp": datetime.now(),
            "status": "pending"
        }