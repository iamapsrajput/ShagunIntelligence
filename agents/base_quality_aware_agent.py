"""
Base class for quality-aware agents that make decisions based on data quality.
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from loguru import logger
from backend.data_sources.integration import get_data_source_integration
from backend.data_sources.data_quality_validator import DataQualityValidator, QualityGrade


class DataQualityLevel(Enum):
    """Data quality levels for decision making"""
    HIGH = "high"          # Quality > 0.8
    MEDIUM = "medium"      # Quality 0.6-0.8
    LOW = "low"           # Quality < 0.6
    CRITICAL = "critical"  # Quality < 0.3 or no data


class TradingMode(Enum):
    """Trading modes based on data quality"""
    NORMAL = "normal"           # Full analysis and normal trading
    CONSERVATIVE = "conservative"  # Conservative analysis with reduced positions
    DEFENSIVE = "defensive"     # Hold positions, no new trades
    EMERGENCY = "emergency"     # Emergency exit mode


class BaseQualityAwareAgent:
    """Base class for agents that incorporate data quality in decision making"""
    
    def __init__(self):
        self.data_integration = get_data_source_integration()
        self.quality_validator = DataQualityValidator()
        
        # Quality thresholds can be overridden by specific agents
        self.quality_thresholds = {
            DataQualityLevel.HIGH: 0.8,
            DataQualityLevel.MEDIUM: 0.6,
            DataQualityLevel.LOW: 0.3
        }
        
        # Trading mode mappings
        self.quality_to_mode = {
            DataQualityLevel.HIGH: TradingMode.NORMAL,
            DataQualityLevel.MEDIUM: TradingMode.CONSERVATIVE,
            DataQualityLevel.LOW: TradingMode.DEFENSIVE,
            DataQualityLevel.CRITICAL: TradingMode.EMERGENCY
        }
        
        # Position size adjustments based on quality
        self.quality_position_multipliers = {
            DataQualityLevel.HIGH: 1.0,      # Full position size
            DataQualityLevel.MEDIUM: 0.5,    # 50% position size
            DataQualityLevel.LOW: 0.0,       # No new positions
            DataQualityLevel.CRITICAL: -1.0  # Exit positions
        }
    
    async def get_quality_weighted_data(
        self,
        symbol: str,
        data_type: str = "quote"
    ) -> Tuple[Optional[Dict[str, Any]], float, DataQualityLevel]:
        """
        Get data with quality assessment.
        
        Returns:
            Tuple of (data, quality_score, quality_level)
        """
        try:
            # Fetch data based on type
            if data_type == "quote":
                data = await self.data_integration.get_quote(symbol)
            elif data_type == "quotes":
                data = await self.data_integration.get_quotes([symbol])
                data = data.get(symbol) if data else None
            elif data_type == "depth":
                data = await self.data_integration.get_market_depth(symbol)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            if not data:
                return None, 0.0, DataQualityLevel.CRITICAL
            
            # Validate data quality
            from backend.data_sources.base import DataSourceType
            from backend.data_sources.market.models import MarketData
            
            # Convert dict to MarketData if needed
            if isinstance(data, dict):
                market_data = MarketData.from_dict(data)
            else:
                market_data = data
            
            # Get reference data for validation
            reference_data = {}
            if hasattr(self.data_integration, 'market_source_manager'):
                # Get data from multiple sources for comparison
                sources_status = self.data_integration.get_market_sources_status()
                healthy_sources = [
                    name for name, status in sources_status['sources'].items()
                    if status['health'] == 'healthy'
                ]
                
                # Fetch from up to 2 other sources for reference
                for source_name in healthy_sources[:2]:
                    if source_name != data.get('data_source'):
                        try:
                            ref_quote = await self._get_reference_quote(symbol, source_name)
                            if ref_quote:
                                reference_data[source_name] = ref_quote
                        except:
                            pass
            
            # Validate quality
            quality_metrics = self.quality_validator.validate_data(
                market_data,
                DataSourceType.MARKET_DATA,
                reference_data
            )
            
            # Determine quality level
            quality_level = self._determine_quality_level(quality_metrics.overall_score)
            
            # Add quality metadata to data
            if isinstance(data, dict):
                data['_quality_score'] = quality_metrics.overall_score
                data['_quality_level'] = quality_level.value
                data['_quality_grade'] = quality_metrics.quality_grade.value
            
            logger.info(
                f"Data quality for {symbol}: {quality_metrics.overall_score:.2f} "
                f"({quality_level.value}) from {data.get('data_source', 'unknown')}"
            )
            
            return data, quality_metrics.overall_score, quality_level
            
        except Exception as e:
            logger.error(f"Error getting quality-weighted data for {symbol}: {e}")
            return None, 0.0, DataQualityLevel.CRITICAL
    
    async def get_multi_source_consensus(
        self,
        symbol: str,
        min_sources: int = 2
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Get consensus data from multiple sources.
        
        Returns:
            Tuple of (consensus_data, confidence_score)
        """
        try:
            # Get quotes from all available sources
            all_quotes = {}
            
            if hasattr(self.data_integration, 'market_source_manager'):
                sources_status = self.data_integration.get_market_sources_status()
                
                for source_name in sources_status['sources']:
                    try:
                        quote = await self._get_source_specific_quote(symbol, source_name)
                        if quote:
                            all_quotes[source_name] = quote
                    except:
                        continue
            
            if len(all_quotes) < min_sources:
                # Not enough sources, fallback to single source
                quote, quality, _ = await self.get_quality_weighted_data(symbol)
                return quote, quality if quote else 0.0
            
            # Calculate consensus
            consensus_data = self._calculate_consensus(all_quotes)
            confidence = self._calculate_confidence(all_quotes)
            
            return consensus_data, confidence
            
        except Exception as e:
            logger.error(f"Error getting multi-source consensus for {symbol}: {e}")
            return None, 0.0
    
    def _determine_quality_level(self, quality_score: float) -> DataQualityLevel:
        """Determine quality level from score"""
        if quality_score >= self.quality_thresholds[DataQualityLevel.HIGH]:
            return DataQualityLevel.HIGH
        elif quality_score >= self.quality_thresholds[DataQualityLevel.MEDIUM]:
            return DataQualityLevel.MEDIUM
        elif quality_score >= self.quality_thresholds[DataQualityLevel.LOW]:
            return DataQualityLevel.LOW
        else:
            return DataQualityLevel.CRITICAL
    
    def get_trading_mode(self, quality_level: DataQualityLevel) -> TradingMode:
        """Get trading mode based on quality level"""
        return self.quality_to_mode.get(quality_level, TradingMode.EMERGENCY)
    
    def adjust_position_size(
        self,
        base_size: float,
        quality_level: DataQualityLevel
    ) -> float:
        """Adjust position size based on data quality"""
        multiplier = self.quality_position_multipliers.get(quality_level, 0.0)
        
        if multiplier < 0:
            # Emergency exit mode
            return -abs(base_size)  # Negative indicates exit
        
        return base_size * multiplier
    
    def should_trade(self, quality_level: DataQualityLevel) -> bool:
        """Determine if trading should be allowed based on quality"""
        return quality_level in [DataQualityLevel.HIGH, DataQualityLevel.MEDIUM]
    
    def get_confidence_score(
        self,
        quality_score: float,
        additional_factors: Dict[str, float] = None
    ) -> float:
        """
        Calculate overall confidence score combining data quality and other factors.
        
        Args:
            quality_score: Base data quality score (0-1)
            additional_factors: Dict of factor_name -> score (0-1)
        
        Returns:
            Overall confidence score (0-1)
        """
        factors = {"data_quality": quality_score}
        
        if additional_factors:
            factors.update(additional_factors)
        
        # Weighted average of all factors
        weights = {
            "data_quality": 0.4,  # 40% weight on data quality
            "technical": 0.3,     # 30% on technical indicators
            "sentiment": 0.2,     # 20% on sentiment
            "market": 0.1        # 10% on market conditions
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for factor, score in factors.items():
            weight = weights.get(factor, 0.1)  # Default 10% for unknown factors
            weighted_sum += score * weight
            total_weight += weight
        
        confidence = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Apply quality penalty
        if quality_score < 0.6:
            confidence *= 0.7  # 30% penalty for low quality data
        
        return min(max(confidence, 0.0), 1.0)
    
    async def _get_reference_quote(
        self,
        symbol: str,
        source_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get quote from specific source for reference"""
        try:
            # This would need to be implemented based on source manager
            return None
        except:
            return None
    
    async def _get_source_specific_quote(
        self,
        symbol: str,
        source_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get quote from specific source"""
        try:
            # This would need to be implemented based on source manager
            return None
        except:
            return None
    
    def _calculate_consensus(
        self,
        quotes: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate consensus from multiple quotes"""
        if not quotes:
            return {}
        
        # Get all prices
        prices = [q.get('current_price', 0) for q in quotes.values() if q.get('current_price')]
        
        if not prices:
            return list(quotes.values())[0]  # Return first quote if no prices
        
        # Calculate median price (more robust than mean)
        median_price = sorted(prices)[len(prices) // 2]
        
        # Find quote closest to median
        best_quote = None
        min_diff = float('inf')
        
        for quote in quotes.values():
            price = quote.get('current_price', 0)
            if price and abs(price - median_price) < min_diff:
                min_diff = abs(price - median_price)
                best_quote = quote
        
        # Add consensus metadata
        if best_quote:
            best_quote = best_quote.copy()
            best_quote['_consensus_price'] = median_price
            best_quote['_source_count'] = len(quotes)
            best_quote['_price_variance'] = max(prices) - min(prices) if len(prices) > 1 else 0
        
        return best_quote
    
    def _calculate_confidence(
        self,
        quotes: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate confidence based on quote agreement"""
        if len(quotes) < 2:
            return 0.5  # Low confidence with single source
        
        prices = [q.get('current_price', 0) for q in quotes.values() if q.get('current_price')]
        
        if not prices or len(prices) < 2:
            return 0.5
        
        # Calculate coefficient of variation
        mean_price = sum(prices) / len(prices)
        if mean_price == 0:
            return 0.0
        
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5
        cv = std_dev / mean_price
        
        # Convert CV to confidence (lower CV = higher confidence)
        # CV of 0.01 (1%) = 0.9 confidence
        # CV of 0.05 (5%) = 0.5 confidence
        # CV > 0.1 (10%) = 0.1 confidence
        
        if cv <= 0.01:
            confidence = 0.9
        elif cv <= 0.05:
            confidence = 0.9 - ((cv - 0.01) / 0.04) * 0.4
        else:
            confidence = max(0.1, 0.5 - (cv - 0.05) * 2)
        
        # Boost confidence if we have many agreeing sources
        source_bonus = min(0.1, (len(quotes) - 2) * 0.02)
        confidence = min(1.0, confidence + source_bonus)
        
        return confidence
    
    def format_quality_message(
        self,
        quality_score: float,
        quality_level: DataQualityLevel,
        trading_mode: TradingMode
    ) -> str:
        """Format a message about data quality and trading mode"""
        return (
            f"Data Quality: {quality_score:.1%} ({quality_level.value})\n"
            f"Trading Mode: {trading_mode.value}\n"
            f"Position Sizing: {self.quality_position_multipliers[quality_level]:.0%}"
        )