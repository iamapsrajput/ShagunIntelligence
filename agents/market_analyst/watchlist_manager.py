"""Watchlist management and opportunity scoring system"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from loguru import logger

from .data_processor import RealTimeDataProcessor
from .statistical_analyzer import StatisticalAnalyzer, TradingSignal
from .pattern_recognition import PatternRecognitionEngine
from .volume_analyzer import VolumeSignalGenerator, VolumeAnomaly


class OpportunityType(Enum):
    """Types of trading opportunities"""
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLUME_ANOMALY = "volume_anomaly"
    PATTERN_SETUP = "pattern_setup"


class AlertLevel(Enum):
    """Alert priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WatchlistItem:
    """Individual watchlist item"""
    symbol: str
    added_date: datetime
    priority: int  # 1-5, 5 being highest
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    alert_conditions: Dict[str, Any] = field(default_factory=dict)
    last_analysis: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'added_date': self.added_date.isoformat(),
            'priority': self.priority,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'notes': self.notes,
            'tags': self.tags,
            'alert_conditions': self.alert_conditions,
            'last_analysis': self.last_analysis.isoformat() if self.last_analysis else None
        }


@dataclass
class TradingOpportunity:
    """Trading opportunity structure"""
    symbol: str
    opportunity_type: OpportunityType
    score: float  # 0.0 to 100.0
    confidence: float  # 0.0 to 1.0
    expected_move: float  # Expected price move percentage
    time_horizon: str  # 'short', 'medium', 'long'
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    risk_reward_ratio: Optional[float]
    signals: List[Dict[str, Any]]
    reasoning: List[str]
    supporting_data: Dict[str, Any]
    timestamp: datetime
    alert_level: AlertLevel
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'opportunity_type': self.opportunity_type.value,
            'score': self.score,
            'confidence': self.confidence,
            'expected_move': self.expected_move,
            'time_horizon': self.time_horizon,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'risk_reward_ratio': self.risk_reward_ratio,
            'signals': self.signals,
            'reasoning': self.reasoning,
            'supporting_data': self.supporting_data,
            'timestamp': self.timestamp.isoformat(),
            'alert_level': self.alert_level.value
        }


class OpportunityScorer:
    """Scores trading opportunities based on multiple factors"""
    
    def __init__(self):
        # Scoring weights for different components
        self.component_weights = {
            'technical_analysis': 25.0,
            'pattern_recognition': 20.0,
            'volume_analysis': 20.0,
            'momentum': 15.0,
            'risk_reward': 10.0,
            'market_conditions': 10.0
        }
        
        # Opportunity type base scores
        self.opportunity_base_scores = {
            OpportunityType.BREAKOUT: 75.0,
            OpportunityType.REVERSAL: 65.0,
            OpportunityType.MOMENTUM: 70.0,
            OpportunityType.MEAN_REVERSION: 60.0,
            OpportunityType.VOLUME_ANOMALY: 80.0,
            OpportunityType.PATTERN_SETUP: 65.0
        }
    
    def score_opportunity(
        self,
        symbol: str,
        trading_signal: Optional[TradingSignal] = None,
        pattern_analysis: Optional[Dict[str, Any]] = None,
        volume_signals: Optional[List[Any]] = None,
        volume_anomalies: Optional[List[VolumeAnomaly]] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> TradingOpportunity:
        """Score a trading opportunity based on all available data"""
        
        try:
            # Determine opportunity type
            opportunity_type = self._determine_opportunity_type(
                trading_signal, pattern_analysis, volume_signals, volume_anomalies
            )
            
            # Calculate component scores
            scores = {}
            
            # Technical analysis score
            scores['technical_analysis'] = self._score_technical_analysis(trading_signal)
            
            # Pattern recognition score
            scores['pattern_recognition'] = self._score_pattern_analysis(pattern_analysis)
            
            # Volume analysis score
            scores['volume_analysis'] = self._score_volume_analysis(volume_signals, volume_anomalies)
            
            # Momentum score
            scores['momentum'] = self._score_momentum(trading_signal, market_data)
            
            # Risk-reward score
            scores['risk_reward'] = self._score_risk_reward(trading_signal)
            
            # Market conditions score
            scores['market_conditions'] = self._score_market_conditions(market_data)
            
            # Calculate weighted total score
            total_score = 0
            total_weight = 0
            
            for component, score in scores.items():
                if score is not None:
                    weight = self.component_weights[component]
                    total_score += score * weight
                    total_weight += weight
            
            final_score = (total_score / total_weight) if total_weight > 0 else 0
            
            # Apply opportunity type base score
            base_score = self.opportunity_base_scores.get(opportunity_type, 50.0)
            final_score = (final_score + base_score) / 2
            
            # Calculate other metrics
            confidence = self._calculate_confidence(scores, trading_signal)
            expected_move = self._estimate_expected_move(trading_signal, pattern_analysis)
            time_horizon = self._determine_time_horizon(opportunity_type, pattern_analysis)
            
            # Determine alert level
            alert_level = self._determine_alert_level(final_score, confidence, opportunity_type)
            
            # Get entry/target/stop prices
            entry_price = market_data.get('current_price', 0) if market_data else 0
            target_price, stop_loss = self._calculate_levels(trading_signal, entry_price, expected_move)
            risk_reward_ratio = self._calculate_risk_reward_ratio(entry_price, target_price, stop_loss)
            
            # Compile signals and reasoning
            signals = self._compile_signals(trading_signal, pattern_analysis, volume_signals)
            reasoning = self._compile_reasoning(scores, opportunity_type, trading_signal)
            
            return TradingOpportunity(
                symbol=symbol,
                opportunity_type=opportunity_type,
                score=min(max(final_score, 0), 100),
                confidence=confidence,
                expected_move=expected_move,
                time_horizon=time_horizon,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward_ratio,
                signals=signals,
                reasoning=reasoning,
                supporting_data={
                    'component_scores': scores,
                    'base_score': base_score,
                    'raw_score': final_score
                },
                timestamp=datetime.now(),
                alert_level=alert_level
            )
            
        except Exception as e:
            logger.error(f"Error scoring opportunity for {symbol}: {str(e)}")
            return TradingOpportunity(
                symbol=symbol,
                opportunity_type=OpportunityType.PATTERN_SETUP,
                score=0.0,
                confidence=0.0,
                expected_move=0.0,
                time_horizon='medium',
                entry_price=0.0,
                target_price=None,
                stop_loss=None,
                risk_reward_ratio=None,
                signals=[],
                reasoning=['Error in scoring'],
                supporting_data={'error': str(e)},
                timestamp=datetime.now(),
                alert_level=AlertLevel.LOW
            )
    
    def _determine_opportunity_type(self, trading_signal, pattern_analysis, volume_signals, volume_anomalies) -> OpportunityType:
        """Determine the type of opportunity"""
        # Check for volume anomalies first (highest priority)
        if volume_anomalies and len(volume_anomalies) > 0:
            return OpportunityType.VOLUME_ANOMALY
        
        # Check for breakout patterns
        if pattern_analysis and 'breakouts' in pattern_analysis:
            breakouts = pattern_analysis['breakouts']
            if breakouts and len(breakouts) > 0:
                return OpportunityType.BREAKOUT
        
        # Check for pattern setups
        if pattern_analysis and 'pattern_signals' in pattern_analysis:
            pattern_signals = pattern_analysis['pattern_signals']
            if pattern_signals and len(pattern_signals) > 0:
                return OpportunityType.PATTERN_SETUP
        
        # Check trading signal for momentum/reversal
        if trading_signal:
            if trading_signal.signal_type in ['BUY', 'SELL']:
                # Check if it's a reversal based on reasoning
                reasoning_text = ' '.join(trading_signal.reasoning).lower()
                if any(word in reasoning_text for word in ['reversal', 'oversold', 'overbought', 'divergence']):
                    return OpportunityType.REVERSAL
                elif any(word in reasoning_text for word in ['momentum', 'trend', 'breakout']):
                    return OpportunityType.MOMENTUM
                else:
                    return OpportunityType.MEAN_REVERSION
        
        return OpportunityType.PATTERN_SETUP
    
    def _score_technical_analysis(self, trading_signal: Optional[TradingSignal]) -> Optional[float]:
        """Score based on technical analysis"""
        if not trading_signal:
            return None
        
        # Base score from signal strength and confidence
        base_score = trading_signal.strength * 100
        confidence_bonus = trading_signal.confidence * 20
        
        # Signal type bonus
        signal_bonus = 10 if trading_signal.signal_type in ['BUY', 'SELL'] else 0
        
        total_score = base_score + confidence_bonus + signal_bonus
        return min(total_score, 100)
    
    def _score_pattern_analysis(self, pattern_analysis: Optional[Dict[str, Any]]) -> Optional[float]:
        """Score based on pattern recognition"""
        if not pattern_analysis:
            return None
        
        score = 0
        
        # Score breakouts
        breakouts = pattern_analysis.get('breakouts', [])
        if breakouts:
            for breakout in breakouts:
                strength = breakout.get('strength', 0)
                volume_confirmed = breakout.get('volume_confirmation', False)
                score += strength * 10 + (20 if volume_confirmed else 0)
        
        # Score chart patterns
        for timeframe, patterns in pattern_analysis.get('chart_patterns', {}).items():
            if patterns:
                for pattern in patterns:
                    confidence = pattern.confidence if hasattr(pattern, 'confidence') else 0.5
                    direction_bonus = 10 if pattern.direction in ['bullish', 'bearish'] else 0
                    score += confidence * 30 + direction_bonus
        
        # Score candlestick patterns
        candlestick_patterns = pattern_analysis.get('candlestick_patterns', {})
        if candlestick_patterns:
            for timeframe, patterns in candlestick_patterns.items():
                score += len(patterns) * 5  # 5 points per pattern
        
        return min(score, 100)
    
    def _score_volume_analysis(self, volume_signals: Optional[List], volume_anomalies: Optional[List[VolumeAnomaly]]) -> Optional[float]:
        """Score based on volume analysis"""
        score = 0
        
        # Score volume anomalies
        if volume_anomalies:
            for anomaly in volume_anomalies:
                score += anomaly.severity * 30
                if anomaly.anomaly_type == 'spike':
                    score += 20  # Volume spikes are generally bullish
        
        # Score volume signals
        if volume_signals:
            for signal in volume_signals:
                signal_strength = signal.strength if hasattr(signal, 'strength') else 0.5
                score += signal_strength * 25
                
                # Bonus for specific signal types
                signal_type = signal.signal_type if hasattr(signal, 'signal_type') else ''
                if signal_type in ['accumulation', 'breakout_volume']:
                    score += 15
        
        return min(score, 100) if score > 0 else None
    
    def _score_momentum(self, trading_signal: Optional[TradingSignal], market_data: Optional[Dict]) -> Optional[float]:
        """Score based on momentum indicators"""
        if not trading_signal or not trading_signal.technical_data:
            return None
        
        score = 50  # Base score
        technical_data = trading_signal.technical_data
        
        # RSI momentum
        rsi = technical_data.get('rsi')
        if rsi:
            if 30 <= rsi <= 70:  # Good momentum range
                score += 20
            elif rsi < 30 or rsi > 70:  # Extreme values
                score += 10
        
        # MACD momentum
        macd = technical_data.get('macd')
        if macd:
            if macd > 0:
                score += 15
            else:
                score -= 5
        
        # Volume ratio
        volume_ratio = technical_data.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            score += 15
        elif volume_ratio < 0.8:
            score -= 10
        
        return min(max(score, 0), 100)
    
    def _score_risk_reward(self, trading_signal: Optional[TradingSignal]) -> Optional[float]:
        """Score based on risk-reward ratio"""
        if not trading_signal or not trading_signal.risk_reward:
            return None
        
        risk, reward = trading_signal.risk_reward
        if risk <= 0:
            return 50  # Neutral score if no risk data
        
        risk_reward_ratio = reward / risk
        
        if risk_reward_ratio >= 3.0:
            return 90
        elif risk_reward_ratio >= 2.0:
            return 80
        elif risk_reward_ratio >= 1.5:
            return 70
        elif risk_reward_ratio >= 1.0:
            return 60
        else:
            return 30  # Poor risk-reward
    
    def _score_market_conditions(self, market_data: Optional[Dict]) -> Optional[float]:
        """Score based on overall market conditions"""
        if not market_data:
            return None
        
        # This is a simplified scoring - in practice, you'd analyze broader market indicators
        score = 50  # Neutral base score
        
        # Add factors like market volatility, sector performance, etc.
        volatility = market_data.get('volatility', 0)
        if volatility < 0.02:  # Low volatility
            score += 10
        elif volatility > 0.05:  # High volatility
            score -= 10
        
        return score
    
    def _calculate_confidence(self, scores: Dict[str, Optional[float]], trading_signal: Optional[TradingSignal]) -> float:
        """Calculate overall confidence in the opportunity"""
        # Base confidence from trading signal
        base_confidence = trading_signal.confidence if trading_signal else 0.5
        
        # Boost confidence if multiple components agree
        valid_scores = [score for score in scores.values() if score is not None]
        if len(valid_scores) >= 3:
            base_confidence += 0.2
        
        # Boost if scores are consistently high
        if valid_scores and np.mean(valid_scores) > 70:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _estimate_expected_move(self, trading_signal: Optional[TradingSignal], pattern_analysis: Optional[Dict]) -> float:
        """Estimate expected price move percentage"""
        expected_move = 2.0  # Default 2% move
        
        if trading_signal and trading_signal.technical_data:
            atr_percent = trading_signal.technical_data.get('atr_percent', 2.0)
            expected_move = max(atr_percent * 1.5, 1.0)  # 1.5x ATR, minimum 1%
        
        # Adjust based on patterns
        if pattern_analysis and 'breakouts' in pattern_analysis:
            breakouts = pattern_analysis['breakouts']
            if breakouts:
                max_strength = max([b.get('strength', 0) for b in breakouts])
                expected_move = max(expected_move, max_strength)
        
        return min(expected_move, 10.0)  # Cap at 10%
    
    def _determine_time_horizon(self, opportunity_type: OpportunityType, pattern_analysis: Optional[Dict]) -> str:
        """Determine time horizon for the opportunity"""
        if opportunity_type in [OpportunityType.VOLUME_ANOMALY, OpportunityType.BREAKOUT]:
            return 'short'  # Minutes to hours
        elif opportunity_type in [OpportunityType.MOMENTUM, OpportunityType.PATTERN_SETUP]:
            return 'medium'  # Hours to days
        else:
            return 'long'  # Days to weeks
    
    def _determine_alert_level(self, score: float, confidence: float, opportunity_type: OpportunityType) -> AlertLevel:
        """Determine alert priority level"""
        if score >= 85 and confidence >= 0.8:
            return AlertLevel.CRITICAL
        elif score >= 75 and confidence >= 0.7:
            return AlertLevel.HIGH
        elif score >= 60 and confidence >= 0.6:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
    
    def _calculate_levels(self, trading_signal: Optional[TradingSignal], entry_price: float, expected_move: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate target and stop loss levels"""
        if not entry_price:
            return None, None
        
        if trading_signal and trading_signal.signal_type == 'BUY':
            target_price = entry_price * (1 + expected_move / 100)
            stop_loss = entry_price * (1 - expected_move / 200)  # Half the expected move for stop
        elif trading_signal and trading_signal.signal_type == 'SELL':
            target_price = entry_price * (1 - expected_move / 100)
            stop_loss = entry_price * (1 + expected_move / 200)
        else:
            return None, None
        
        return target_price, stop_loss
    
    def _calculate_risk_reward_ratio(self, entry_price: float, target_price: Optional[float], stop_loss: Optional[float]) -> Optional[float]:
        """Calculate risk-reward ratio"""
        if not all([entry_price, target_price, stop_loss]):
            return None
        
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        
        return reward / risk if risk > 0 else None
    
    def _compile_signals(self, trading_signal, pattern_analysis, volume_signals) -> List[Dict[str, Any]]:
        """Compile all signals into a list"""
        signals = []
        
        if trading_signal:
            signals.append({
                'type': 'technical_analysis',
                'signal': trading_signal.signal_type,
                'strength': trading_signal.strength,
                'confidence': trading_signal.confidence
            })
        
        if pattern_analysis and 'pattern_signals' in pattern_analysis:
            for signal in pattern_analysis['pattern_signals']:
                signals.append({
                    'type': 'pattern',
                    'pattern': signal.get('pattern', ''),
                    'direction': signal.get('direction', ''),
                    'strength': signal.get('strength', 0)
                })
        
        if volume_signals:
            for signal in volume_signals:
                signals.append({
                    'type': 'volume',
                    'signal_type': signal.signal_type if hasattr(signal, 'signal_type') else '',
                    'strength': signal.strength if hasattr(signal, 'strength') else 0
                })
        
        return signals
    
    def _compile_reasoning(self, scores: Dict[str, Optional[float]], opportunity_type: OpportunityType, trading_signal: Optional[TradingSignal]) -> List[str]:
        """Compile reasoning for the opportunity"""
        reasoning = []
        
        reasoning.append(f"Opportunity type: {opportunity_type.value}")
        
        # Add component score reasoning
        for component, score in scores.items():
            if score is not None and score > 60:
                reasoning.append(f"Strong {component.replace('_', ' ')}: {score:.1f}/100")
        
        # Add signal-specific reasoning
        if trading_signal and trading_signal.reasoning:
            reasoning.extend(trading_signal.reasoning[:3])  # Top 3 reasons
        
        return reasoning


class WatchlistManager:
    """Manages trading watchlists and monitors opportunities"""
    
    def __init__(self, data_processor: RealTimeDataProcessor):
        self.data_processor = data_processor
        self.statistical_analyzer = StatisticalAnalyzer(data_processor)
        self.pattern_engine = PatternRecognitionEngine(data_processor)
        self.volume_analyzer = VolumeSignalGenerator(data_processor)
        self.opportunity_scorer = OpportunityScorer()
        
        # Watchlist storage
        self.watchlists: Dict[str, List[WatchlistItem]] = {'default': []}
        self.opportunities: Dict[str, List[TradingOpportunity]] = {}
        
        # Monitoring settings
        self.scan_interval = 60  # seconds
        self.max_opportunities_per_symbol = 5
        
        # Alert callbacks
        self.alert_callbacks: List[callable] = []
        
    def create_watchlist(self, name: str) -> bool:
        """Create a new watchlist"""
        if name not in self.watchlists:
            self.watchlists[name] = []
            logger.info(f"Created watchlist: {name}")
            return True
        return False
    
    def add_symbol(self, symbol: str, watchlist_name: str = 'default', priority: int = 3, **kwargs) -> bool:
        """Add symbol to watchlist"""
        try:
            if watchlist_name not in self.watchlists:
                self.create_watchlist(watchlist_name)
            
            # Check if symbol already exists
            existing_symbols = [item.symbol for item in self.watchlists[watchlist_name]]
            if symbol in existing_symbols:
                logger.warning(f"Symbol {symbol} already in watchlist {watchlist_name}")
                return False
            
            item = WatchlistItem(
                symbol=symbol,
                added_date=datetime.now(),
                priority=priority,
                target_price=kwargs.get('target_price'),
                stop_loss=kwargs.get('stop_loss'),
                notes=kwargs.get('notes', ''),
                tags=kwargs.get('tags', []),
                alert_conditions=kwargs.get('alert_conditions', {})
            )
            
            self.watchlists[watchlist_name].append(item)
            logger.info(f"Added {symbol} to watchlist {watchlist_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {str(e)}")
            return False
    
    def remove_symbol(self, symbol: str, watchlist_name: str = 'default') -> bool:
        """Remove symbol from watchlist"""
        try:
            if watchlist_name in self.watchlists:
                self.watchlists[watchlist_name] = [
                    item for item in self.watchlists[watchlist_name] 
                    if item.symbol != symbol
                ]
                logger.info(f"Removed {symbol} from watchlist {watchlist_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {str(e)}")
            return False
    
    async def scan_opportunities(self, watchlist_name: str = 'default') -> List[TradingOpportunity]:
        """Scan watchlist for trading opportunities"""
        if watchlist_name not in self.watchlists:
            return []
        
        opportunities = []
        symbols = [item.symbol for item in self.watchlists[watchlist_name]]
        
        logger.info(f"Scanning {len(symbols)} symbols for opportunities")
        
        for symbol in symbols:
            try:
                opportunity = await self._analyze_symbol_opportunity(symbol)
                if opportunity and opportunity.score > 50:  # Minimum score threshold
                    opportunities.append(opportunity)
                    
                    # Trigger alerts for high-priority opportunities
                    if opportunity.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]:
                        await self._trigger_alert(opportunity)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Sort by score and update storage
        opportunities.sort(key=lambda x: x.score, reverse=True)
        self.opportunities[watchlist_name] = opportunities[:20]  # Keep top 20
        
        logger.info(f"Found {len(opportunities)} opportunities in {watchlist_name}")
        return opportunities
    
    async def _analyze_symbol_opportunity(self, symbol: str) -> Optional[TradingOpportunity]:
        """Analyze a single symbol for opportunities"""
        try:
            # Get current market data
            latest_tick = self.data_processor.get_latest_tick(symbol)
            if not latest_tick:
                return None
            
            market_data = {
                'current_price': latest_tick.last_price,
                'volume': latest_tick.volume,
                'change_percent': latest_tick.change_percent
            }
            
            # Perform various analyses
            trading_signal = self.statistical_analyzer.analyze_symbol(symbol)
            pattern_analysis = self.pattern_engine.analyze_patterns(symbol)
            volume_signals = self.volume_analyzer.generate_volume_signals(symbol)
            volume_anomalies = self.volume_analyzer.anomaly_detector.detect_anomalies(symbol)
            
            # Score the opportunity
            opportunity = self.opportunity_scorer.score_opportunity(
                symbol=symbol,
                trading_signal=trading_signal,
                pattern_analysis=pattern_analysis,
                volume_signals=volume_signals,
                volume_anomalies=volume_anomalies,
                market_data=market_data
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {str(e)}")
            return None
    
    def get_top_opportunities(self, watchlist_name: str = 'default', count: int = 10) -> List[TradingOpportunity]:
        """Get top trading opportunities"""
        opportunities = self.opportunities.get(watchlist_name, [])
        return opportunities[:count]
    
    def get_opportunities_by_type(self, opportunity_type: OpportunityType, watchlist_name: str = 'default') -> List[TradingOpportunity]:
        """Get opportunities filtered by type"""
        opportunities = self.opportunities.get(watchlist_name, [])
        return [opp for opp in opportunities if opp.opportunity_type == opportunity_type]
    
    def get_critical_alerts(self, watchlist_name: str = 'default') -> List[TradingOpportunity]:
        """Get critical alert opportunities"""
        opportunities = self.opportunities.get(watchlist_name, [])
        return [opp for opp in opportunities if opp.alert_level == AlertLevel.CRITICAL]
    
    def update_symbol_settings(self, symbol: str, watchlist_name: str = 'default', **kwargs) -> bool:
        """Update symbol settings"""
        try:
            if watchlist_name in self.watchlists:
                for item in self.watchlists[watchlist_name]:
                    if item.symbol == symbol:
                        for key, value in kwargs.items():
                            if hasattr(item, key):
                                setattr(item, key, value)
                        return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating symbol {symbol}: {str(e)}")
            return False
    
    def get_watchlist_summary(self, watchlist_name: str = 'default') -> Dict[str, Any]:
        """Get watchlist summary"""
        if watchlist_name not in self.watchlists:
            return {}
        
        items = self.watchlists[watchlist_name]
        opportunities = self.opportunities.get(watchlist_name, [])
        
        return {
            'name': watchlist_name,
            'total_symbols': len(items),
            'total_opportunities': len(opportunities),
            'high_priority_opportunities': len([opp for opp in opportunities if opp.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]]),
            'average_score': np.mean([opp.score for opp in opportunities]) if opportunities else 0,
            'last_scan': datetime.now(),
            'symbols': [item.symbol for item in items]
        }
    
    async def _trigger_alert(self, opportunity: TradingOpportunity):
        """Trigger alert for high-priority opportunity"""
        try:
            for callback in self.alert_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(opportunity)
                else:
                    callback(opportunity)
        except Exception as e:
            logger.error(f"Error triggering alert: {str(e)}")
    
    def add_alert_callback(self, callback: callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def save_watchlists(self, filepath: str):
        """Save watchlists to file"""
        try:
            data = {}
            for name, items in self.watchlists.items():
                data[name] = [item.to_dict() for item in items]
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved watchlists to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving watchlists: {str(e)}")
    
    def load_watchlists(self, filepath: str):
        """Load watchlists from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for name, items_data in data.items():
                items = []
                for item_data in items_data:
                    item = WatchlistItem(
                        symbol=item_data['symbol'],
                        added_date=datetime.fromisoformat(item_data['added_date']),
                        priority=item_data['priority'],
                        target_price=item_data.get('target_price'),
                        stop_loss=item_data.get('stop_loss'),
                        notes=item_data.get('notes', ''),
                        tags=item_data.get('tags', []),
                        alert_conditions=item_data.get('alert_conditions', {}),
                        last_analysis=datetime.fromisoformat(item_data['last_analysis']) if item_data.get('last_analysis') else None
                    )
                    items.append(item)
                
                self.watchlists[name] = items
            
            logger.info(f"Loaded watchlists from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading watchlists: {str(e)}")