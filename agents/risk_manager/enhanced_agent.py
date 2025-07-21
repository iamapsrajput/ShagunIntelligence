"""Enhanced Risk Management Agent with comprehensive risk controls."""

from crewai import Agent
from langchain.llms.base import BaseLLM
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from loguru import logger
import pandas as pd
import numpy as np

from .position_sizing import PositionSizer
from .stop_loss_manager import StopLossManager
from .portfolio_analyzer import PortfolioAnalyzer
from .circuit_breaker import CircuitBreaker
from .risk_metrics import RiskMetricsCalculator


class EnhancedRiskManagerAgent(Agent):
    """Enhanced agent for comprehensive risk management in trading."""
    
    def __init__(
        self,
        llm: BaseLLM,
        capital: float = 100000,
        max_risk_per_trade: float = 0.02,
        max_portfolio_risk: float = 0.06,
        max_correlation: float = 0.7
    ):
        """
        Initialize the Enhanced Risk Manager Agent.
        
        Args:
            llm: Language model for agent
            capital: Total trading capital
            max_risk_per_trade: Maximum risk per trade (2% default)
            max_portfolio_risk: Maximum portfolio risk (6% default)
            max_correlation: Maximum allowed correlation between positions
        """
        self.llm = llm
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        
        # Initialize components
        self.position_sizer = PositionSizer(capital, max_risk_per_trade)
        self.stop_loss_manager = StopLossManager()
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.circuit_breaker = CircuitBreaker()
        self.risk_metrics = RiskMetricsCalculator()
        
        # Risk tracking
        self.active_positions = {}
        self.risk_history = []
        self.blocked_trades = []
        
        # Call parent constructor
        super().__init__(
            role='Enhanced Risk Manager',
            goal='Comprehensively assess and manage trading risks with advanced position sizing, dynamic stops, and portfolio monitoring',
            backstory="""You are an elite risk management specialist with decades of experience 
            in quantitative finance and portfolio management. You utilize advanced mathematical 
            models including Value at Risk (VaR), volatility-based position sizing, and correlation 
            analysis to protect capital while optimizing returns. You implement strict risk controls 
            including circuit breakers and dynamic stop-loss management. Your expertise spans 
            market microstructure, extreme event modeling, and systematic risk management.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        
        logger.info("Enhanced Risk Manager Agent initialized")

    async def evaluate_trade_risk(
        self,
        symbol: str,
        entry_price: float,
        target_price: float,
        market_data: pd.DataFrame,
        confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Evaluate risk for a potential trade.
        
        Args:
            symbol: Trading symbol
            entry_price: Proposed entry price
            target_price: Target price
            market_data: Historical price data
            confidence: Trade confidence level
            
        Returns:
            Comprehensive risk evaluation
        """
        try:
            # Calculate ATR for volatility
            atr = self.stop_loss_manager.calculate_atr(market_data)
            
            # Find support/resistance levels
            support, resistance = self.stop_loss_manager.find_support_resistance(
                market_data, entry_price
            )
            
            # Calculate optimal stop loss
            stop_loss = self.stop_loss_manager.calculate_dynamic_stop(
                entry_price=entry_price,
                atr=atr,
                support_level=support,
                trade_direction='long' if target_price > entry_price else 'short'
            )
            
            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                entry_price=entry_price,
                stop_loss=stop_loss,
                volatility=atr / entry_price,
                confidence=confidence
            )
            
            # Check portfolio impact
            portfolio_impact = await self._assess_portfolio_impact(
                symbol, position_size['shares'], entry_price
            )
            
            # Circuit breaker check
            circuit_status = self.circuit_breaker.check_conditions(
                market_data, self.active_positions
            )
            
            # Risk score calculation
            risk_score = self._calculate_risk_score(
                risk_reward_ratio, confidence, portfolio_impact, circuit_status
            )
            
            evaluation = {
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'position_size': position_size,
                'risk_reward_ratio': risk_reward_ratio,
                'risk_amount': position_size['risk_amount'],
                'risk_percentage': position_size['risk_percentage'],
                'atr': atr,
                'support_level': support,
                'resistance_level': resistance,
                'portfolio_impact': portfolio_impact,
                'circuit_breaker_status': circuit_status,
                'risk_score': risk_score,
                'recommendation': self._generate_recommendation(risk_score, risk_reward_ratio),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log evaluation
            self.risk_history.append(evaluation)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating trade risk: {str(e)}")
            raise

    async def calculate_optimal_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        market_data: pd.DataFrame,
        strategy: str = 'fixed_percentage'
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size using various algorithms.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            market_data: Historical price data
            strategy: Position sizing strategy
            
        Returns:
            Position sizing details
        """
        try:
            # Calculate volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Get position size based on strategy
            if strategy == 'fixed_percentage':
                position = self.position_sizer.calculate_position_size(
                    entry_price, stop_loss, volatility
                )
            elif strategy == 'kelly_criterion':
                position = self.position_sizer.kelly_criterion_size(
                    win_rate=0.6,  # Would be calculated from historical data
                    avg_win=0.02,
                    avg_loss=0.01,
                    entry_price=entry_price,
                    stop_loss=stop_loss
                )
            elif strategy == 'volatility_based':
                position = self.position_sizer.volatility_based_size(
                    entry_price, volatility
                )
            else:
                position = self.position_sizer.calculate_position_size(
                    entry_price, stop_loss, volatility
                )
            
            # Adjust for portfolio constraints
            adjusted_position = await self._adjust_for_portfolio_limits(
                symbol, position
            )
            
            return adjusted_position
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            raise

    async def set_dynamic_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        market_data: pd.DataFrame,
        position_type: str = 'long'
    ) -> Dict[str, Any]:
        """
        Set and update dynamic stop loss levels.
        
        Args:
            symbol: Trading symbol
            entry_price: Original entry price
            current_price: Current market price
            market_data: Recent price data
            position_type: 'long' or 'short'
            
        Returns:
            Stop loss information
        """
        try:
            # Calculate current ATR
            atr = self.stop_loss_manager.calculate_atr(market_data)
            
            # Find current support/resistance
            support, resistance = self.stop_loss_manager.find_support_resistance(
                market_data, current_price
            )
            
            # Calculate trailing stop
            trailing_stop = self.stop_loss_manager.calculate_trailing_stop(
                entry_price=entry_price,
                current_price=current_price,
                atr=atr,
                position_type=position_type
            )
            
            # Get breakeven stop if applicable
            breakeven_stop = self.stop_loss_manager.breakeven_stop(
                entry_price, current_price, position_type
            )
            
            # Choose optimal stop
            if position_type == 'long':
                optimal_stop = max(trailing_stop, breakeven_stop or 0, support * 0.98)
            else:
                optimal_stop = min(trailing_stop, breakeven_stop or float('inf'), resistance * 1.02)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'trailing_stop': trailing_stop,
                'breakeven_stop': breakeven_stop,
                'support_stop': support * 0.98 if position_type == 'long' else None,
                'resistance_stop': resistance * 1.02 if position_type == 'short' else None,
                'recommended_stop': optimal_stop,
                'stop_distance': abs(current_price - optimal_stop),
                'stop_percentage': abs(current_price - optimal_stop) / current_price,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error setting dynamic stop loss: {str(e)}")
            raise

    async def monitor_portfolio_risk(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Monitor overall portfolio risk metrics.
        
        Args:
            positions: Current portfolio positions
            market_data: Market data for all positions
            
        Returns:
            Portfolio risk analysis
        """
        try:
            # Update active positions
            self.active_positions = {p['symbol']: p for p in positions}
            
            # Calculate portfolio metrics
            total_exposure = self.portfolio_analyzer.calculate_total_exposure(
                positions, self.capital
            )
            
            # Sector concentration
            sector_concentration = self.portfolio_analyzer.analyze_sector_concentration(
                positions
            )
            
            # Correlation analysis
            correlation_matrix = self.portfolio_analyzer.calculate_correlation_matrix(
                positions, market_data
            )
            correlation_risk = self.portfolio_analyzer.assess_correlation_risk(
                correlation_matrix, self.max_correlation
            )
            
            # Calculate VaR
            portfolio_var = self.risk_metrics.calculate_portfolio_var(
                positions, market_data, confidence_level=0.95
            )
            
            # Calculate max drawdown
            drawdown_info = self.risk_metrics.calculate_max_drawdown(
                self.capital, positions, market_data
            )
            
            # Check circuit breaker conditions
            circuit_status = self.circuit_breaker.check_portfolio_circuit(
                total_exposure,
                portfolio_var,
                drawdown_info['current_drawdown']
            )
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'total_positions': len(positions),
                'total_exposure': total_exposure,
                'exposure_percentage': total_exposure['percentage'],
                'sector_concentration': sector_concentration,
                'correlation_risk': correlation_risk,
                'portfolio_var': portfolio_var,
                'max_drawdown': drawdown_info,
                'circuit_breaker': circuit_status,
                'risk_level': self._assess_portfolio_risk_level(
                    total_exposure['percentage'],
                    portfolio_var['var_percentage'],
                    correlation_risk['max_correlation']
                ),
                'recommendations': self._generate_portfolio_recommendations(
                    total_exposure, sector_concentration, correlation_risk, circuit_status
                )
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error monitoring portfolio risk: {str(e)}")
            raise

    async def should_block_trade(
        self,
        symbol: str,
        proposed_size: int,
        entry_price: float,
        current_portfolio: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Determine if a trade should be blocked based on risk criteria.
        
        Args:
            symbol: Trading symbol
            proposed_size: Proposed position size
            entry_price: Entry price
            current_portfolio: Current positions
            
        Returns:
            Tuple of (should_block, reason)
        """
        try:
            # Check circuit breaker
            if self.circuit_breaker.is_triggered():
                return True, "Circuit breaker triggered - extreme market conditions"
            
            # Check position size limits
            position_value = proposed_size * entry_price
            if position_value > self.capital * 0.1:  # Max 10% in single position
                return True, f"Position size exceeds 10% of capital limit"
            
            # Check total exposure
            current_exposure = sum(p['shares'] * p['current_price'] for p in current_portfolio)
            new_exposure = current_exposure + position_value
            
            if new_exposure > self.capital * 0.8:  # Max 80% exposure
                return True, "Total exposure would exceed 80% of capital"
            
            # Check correlation with existing positions
            if self._check_high_correlation(symbol, current_portfolio):
                return True, "High correlation with existing positions"
            
            # Check recent losses
            recent_losses = self._calculate_recent_losses()
            if recent_losses > self.capital * 0.05:  # 5% recent loss limit
                return True, "Recent losses exceed 5% of capital"
            
            return False, "Trade approved"
            
        except Exception as e:
            logger.error(f"Error checking trade block: {str(e)}")
            return True, f"Error in risk check: {str(e)}"

    def _calculate_risk_score(
        self,
        risk_reward_ratio: float,
        confidence: float,
        portfolio_impact: Dict[str, Any],
        circuit_status: Dict[str, Any]
    ) -> float:
        """Calculate overall risk score for a trade."""
        # Base score from risk/reward
        if risk_reward_ratio >= 3:
            base_score = 0.2  # Low risk
        elif risk_reward_ratio >= 2:
            base_score = 0.4  # Moderate risk
        elif risk_reward_ratio >= 1:
            base_score = 0.6  # High risk
        else:
            base_score = 0.9  # Very high risk
        
        # Adjust for confidence
        confidence_adjustment = (1 - confidence) * 0.2
        
        # Adjust for portfolio impact
        if portfolio_impact['increases_concentration']:
            base_score += 0.1
        if portfolio_impact['high_correlation']:
            base_score += 0.15
        
        # Adjust for market conditions
        if circuit_status['volatility_spike']:
            base_score += 0.2
        
        return min(base_score + confidence_adjustment, 1.0)

    def _generate_recommendation(self, risk_score: float, risk_reward_ratio: float) -> str:
        """Generate trade recommendation based on risk assessment."""
        if risk_score < 0.3 and risk_reward_ratio >= 2:
            return "STRONGLY_RECOMMENDED"
        elif risk_score < 0.5 and risk_reward_ratio >= 1.5:
            return "RECOMMENDED"
        elif risk_score < 0.7:
            return "PROCEED_WITH_CAUTION"
        else:
            return "NOT_RECOMMENDED"

    async def _assess_portfolio_impact(
        self,
        symbol: str,
        shares: int,
        price: float
    ) -> Dict[str, Any]:
        """Assess impact of new position on portfolio."""
        # This would integrate with portfolio analyzer
        return {
            'increases_concentration': False,
            'high_correlation': False,
            'within_limits': True
        }

    async def _adjust_for_portfolio_limits(
        self,
        symbol: str,
        position: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adjust position size for portfolio constraints."""
        # Apply portfolio-level constraints
        return position

    def _assess_portfolio_risk_level(
        self,
        exposure_pct: float,
        var_pct: float,
        max_correlation: float
    ) -> str:
        """Assess overall portfolio risk level."""
        risk_points = 0
        
        # Exposure risk
        if exposure_pct > 0.7:
            risk_points += 3
        elif exposure_pct > 0.5:
            risk_points += 2
        elif exposure_pct > 0.3:
            risk_points += 1
        
        # VaR risk
        if var_pct > 0.1:
            risk_points += 3
        elif var_pct > 0.05:
            risk_points += 2
        elif var_pct > 0.02:
            risk_points += 1
        
        # Correlation risk
        if max_correlation > 0.8:
            risk_points += 2
        elif max_correlation > 0.6:
            risk_points += 1
        
        if risk_points >= 6:
            return "CRITICAL"
        elif risk_points >= 4:
            return "HIGH"
        elif risk_points >= 2:
            return "MODERATE"
        else:
            return "LOW"

    def _generate_portfolio_recommendations(
        self,
        exposure: Dict[str, Any],
        concentration: Dict[str, Any],
        correlation: Dict[str, Any],
        circuit: Dict[str, Any]
    ) -> List[str]:
        """Generate portfolio-level recommendations."""
        recommendations = []
        
        if exposure['percentage'] > 0.7:
            recommendations.append("Reduce overall exposure to manage risk")
        
        if concentration['max_sector_weight'] > 0.4:
            recommendations.append("Diversify sector exposure")
        
        if correlation['max_correlation'] > 0.7:
            recommendations.append("Reduce correlated positions")
        
        if circuit['approaching_trigger']:
            recommendations.append("Prepare for potential circuit breaker activation")
        
        return recommendations

    def _check_high_correlation(
        self,
        symbol: str,
        portfolio: List[Dict[str, Any]]
    ) -> bool:
        """Check if symbol has high correlation with existing positions."""
        # Simplified check - would use actual correlation data
        return False

    def _calculate_recent_losses(self) -> float:
        """Calculate recent trading losses."""
        # Would check recent trade history
        return 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get current status of risk manager."""
        return {
            'status': 'active',
            'capital': self.capital,
            'active_positions': len(self.active_positions),
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_portfolio_risk': self.max_portfolio_risk,
            'circuit_breaker': self.circuit_breaker.get_status(),
            'blocked_trades': len(self.blocked_trades)
        }