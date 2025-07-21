"""Portfolio analyzer for exposure and correlation risk management."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
from loguru import logger


class PortfolioAnalyzer:
    """Analyze portfolio exposure, concentration, and correlation risks."""
    
    def __init__(self):
        """Initialize portfolio analyzer."""
        self.sector_mapping = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'META': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology',
            
            # Financial
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'GS': 'Financial', 'MS': 'Financial', 'C': 'Financial',
            
            # Healthcare
            'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
            'CVS': 'Healthcare', 'ABBV': 'Healthcare', 'MRK': 'Healthcare',
            
            # Consumer
            'AMZN': 'Consumer', 'WMT': 'Consumer', 'HD': 'Consumer',
            'MCD': 'Consumer', 'NKE': 'Consumer', 'SBUX': 'Consumer',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            'SLB': 'Energy', 'EOG': 'Energy', 'OXY': 'Energy',
            
            # Industrial
            'BA': 'Industrial', 'CAT': 'Industrial', 'GE': 'Industrial',
            'MMM': 'Industrial', 'LMT': 'Industrial', 'RTX': 'Industrial'
        }
        
        self.correlation_cache = {}
        self.analysis_history = []
        
        logger.info("PortfolioAnalyzer initialized")

    def calculate_total_exposure(
        self,
        positions: List[Dict[str, Any]],
        capital: float
    ) -> Dict[str, Any]:
        """
        Calculate total portfolio exposure.
        
        Args:
            positions: List of current positions
            capital: Total available capital
            
        Returns:
            Exposure analysis
        """
        try:
            if not positions:
                return {
                    'total_value': 0,
                    'percentage': 0,
                    'long_exposure': 0,
                    'short_exposure': 0,
                    'net_exposure': 0,
                    'gross_exposure': 0,
                    'cash_available': capital
                }
            
            # Calculate exposures
            long_exposure = sum(
                p['shares'] * p['current_price'] 
                for p in positions 
                if p.get('position_type', 'long') == 'long'
            )
            
            short_exposure = sum(
                abs(p['shares']) * p['current_price'] 
                for p in positions 
                if p.get('position_type', 'long') == 'short'
            )
            
            gross_exposure = long_exposure + short_exposure
            net_exposure = long_exposure - short_exposure
            
            return {
                'total_value': gross_exposure,
                'percentage': gross_exposure / capital if capital > 0 else 0,
                'long_exposure': long_exposure,
                'short_exposure': short_exposure,
                'net_exposure': net_exposure,
                'gross_exposure': gross_exposure,
                'long_percentage': long_exposure / capital if capital > 0 else 0,
                'short_percentage': short_exposure / capital if capital > 0 else 0,
                'net_percentage': net_exposure / capital if capital > 0 else 0,
                'cash_available': capital - gross_exposure,
                'cash_percentage': (capital - gross_exposure) / capital if capital > 0 else 0,
                'leverage': gross_exposure / capital if capital > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating exposure: {str(e)}")
            raise

    def analyze_sector_concentration(
        self,
        positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze sector concentration in portfolio.
        
        Args:
            positions: List of current positions
            
        Returns:
            Sector concentration analysis
        """
        try:
            if not positions:
                return {
                    'sector_weights': {},
                    'max_sector_weight': 0,
                    'concentration_risk': 'low',
                    'diversification_score': 1.0
                }
            
            # Calculate sector exposures
            sector_values = defaultdict(float)
            total_value = 0
            
            for position in positions:
                symbol = position['symbol']
                value = abs(position['shares'] * position['current_price'])
                total_value += value
                
                # Get sector (default to 'Unknown' if not mapped)
                sector = self.sector_mapping.get(symbol, 'Unknown')
                sector_values[sector] += value
            
            # Calculate sector weights
            sector_weights = {
                sector: value / total_value 
                for sector, value in sector_values.items()
            } if total_value > 0 else {}
            
            # Calculate concentration metrics
            max_sector_weight = max(sector_weights.values()) if sector_weights else 0
            
            # Calculate Herfindahl index for diversification
            herfindahl_index = sum(weight ** 2 for weight in sector_weights.values())
            diversification_score = 1 - herfindahl_index
            
            # Assess concentration risk
            if max_sector_weight > 0.5:
                concentration_risk = 'critical'
            elif max_sector_weight > 0.35:
                concentration_risk = 'high'
            elif max_sector_weight > 0.25:
                concentration_risk = 'moderate'
            else:
                concentration_risk = 'low'
            
            # Identify overweight sectors
            overweight_sectors = [
                sector for sector, weight in sector_weights.items()
                if weight > 0.25
            ]
            
            return {
                'sector_weights': sector_weights,
                'max_sector_weight': max_sector_weight,
                'max_sector': max(sector_weights.items(), key=lambda x: x[1])[0] if sector_weights else None,
                'concentration_risk': concentration_risk,
                'diversification_score': diversification_score,
                'herfindahl_index': herfindahl_index,
                'num_sectors': len(sector_weights),
                'overweight_sectors': overweight_sectors,
                'recommendations': self._generate_sector_recommendations(
                    sector_weights, concentration_risk
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sector concentration: {str(e)}")
            raise

    def calculate_correlation_matrix(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, pd.DataFrame],
        lookback: int = 60
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for portfolio positions.
        
        Args:
            positions: List of current positions
            market_data: Historical price data for each symbol
            lookback: Number of days to calculate correlation
            
        Returns:
            Correlation matrix
        """
        try:
            if len(positions) < 2:
                return pd.DataFrame()
            
            # Extract symbols
            symbols = [p['symbol'] for p in positions]
            
            # Create returns dataframe
            returns_data = {}
            
            for symbol in symbols:
                if symbol in market_data and len(market_data[symbol]) > lookback:
                    # Calculate returns
                    prices = market_data[symbol]['close'].tail(lookback)
                    returns = prices.pct_change().dropna()
                    returns_data[symbol] = returns
            
            if not returns_data:
                return pd.DataFrame()
            
            # Create correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            # Cache the result
            cache_key = '_'.join(sorted(symbols))
            self.correlation_cache[cache_key] = {
                'matrix': correlation_matrix,
                'timestamp': datetime.now()
            }
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()

    def assess_correlation_risk(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Assess correlation risk in portfolio.
        
        Args:
            correlation_matrix: Correlation matrix of positions
            threshold: Correlation threshold for risk assessment
            
        Returns:
            Correlation risk analysis
        """
        try:
            if correlation_matrix.empty:
                return {
                    'max_correlation': 0,
                    'high_correlation_pairs': [],
                    'correlation_risk': 'low',
                    'average_correlation': 0
                }
            
            # Find high correlations (excluding diagonal)
            high_correlations = []
            correlations = []
            
            for i in range(len(correlation_matrix)):
                for j in range(i + 1, len(correlation_matrix)):
                    corr_value = correlation_matrix.iloc[i, j]
                    correlations.append(abs(corr_value))
                    
                    if abs(corr_value) > threshold:
                        high_correlations.append({
                            'pair': (correlation_matrix.index[i], correlation_matrix.columns[j]),
                            'correlation': corr_value
                        })
            
            # Calculate metrics
            max_correlation = max(correlations) if correlations else 0
            avg_correlation = np.mean(correlations) if correlations else 0
            
            # Assess risk level
            if max_correlation > 0.9:
                risk_level = 'critical'
            elif max_correlation > 0.8:
                risk_level = 'high'
            elif max_correlation > 0.7:
                risk_level = 'moderate'
            else:
                risk_level = 'low'
            
            return {
                'max_correlation': max_correlation,
                'average_correlation': avg_correlation,
                'high_correlation_pairs': high_correlations,
                'correlation_risk': risk_level,
                'num_high_correlations': len(high_correlations),
                'recommendations': self._generate_correlation_recommendations(
                    high_correlations, risk_level
                )
            }
            
        except Exception as e:
            logger.error(f"Error assessing correlation risk: {str(e)}")
            raise

    def calculate_position_concentration(
        self,
        positions: List[Dict[str, Any]],
        capital: float
    ) -> Dict[str, Any]:
        """
        Calculate position concentration metrics.
        
        Args:
            positions: List of current positions
            capital: Total capital
            
        Returns:
            Position concentration analysis
        """
        try:
            if not positions:
                return {
                    'max_position_weight': 0,
                    'concentration_risk': 'low',
                    'top_positions': []
                }
            
            # Calculate position weights
            position_weights = []
            for position in positions:
                value = abs(position['shares'] * position['current_price'])
                weight = value / capital if capital > 0 else 0
                position_weights.append({
                    'symbol': position['symbol'],
                    'value': value,
                    'weight': weight
                })
            
            # Sort by weight
            position_weights.sort(key=lambda x: x['weight'], reverse=True)
            
            # Calculate concentration metrics
            max_weight = position_weights[0]['weight'] if position_weights else 0
            top_5_weight = sum(p['weight'] for p in position_weights[:5])
            
            # Assess concentration risk
            if max_weight > 0.15:
                concentration_risk = 'high'
            elif max_weight > 0.10:
                concentration_risk = 'moderate'
            else:
                concentration_risk = 'low'
            
            return {
                'max_position_weight': max_weight,
                'max_position': position_weights[0]['symbol'] if position_weights else None,
                'top_5_concentration': top_5_weight,
                'concentration_risk': concentration_risk,
                'top_positions': position_weights[:5],
                'position_count': len(positions),
                'average_position_size': 1 / len(positions) if positions else 0,
                'recommendations': self._generate_position_recommendations(
                    max_weight, top_5_weight
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating position concentration: {str(e)}")
            raise

    def analyze_risk_factors(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, pd.DataFrame],
        capital: float
    ) -> Dict[str, Any]:
        """
        Comprehensive risk factor analysis.
        
        Args:
            positions: Current positions
            market_data: Market data for positions
            capital: Total capital
            
        Returns:
            Comprehensive risk analysis
        """
        try:
            # Calculate all risk metrics
            exposure = self.calculate_total_exposure(positions, capital)
            sector_concentration = self.analyze_sector_concentration(positions)
            position_concentration = self.calculate_position_concentration(positions, capital)
            
            # Calculate correlation if enough positions
            if len(positions) >= 2:
                correlation_matrix = self.calculate_correlation_matrix(
                    positions, market_data
                )
                correlation_risk = self.assess_correlation_risk(correlation_matrix)
            else:
                correlation_risk = {
                    'max_correlation': 0,
                    'correlation_risk': 'low',
                    'high_correlation_pairs': []
                }
            
            # Calculate overall risk score
            risk_score = self._calculate_overall_risk_score(
                exposure,
                sector_concentration,
                position_concentration,
                correlation_risk
            )
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'exposure_analysis': exposure,
                'sector_concentration': sector_concentration,
                'position_concentration': position_concentration,
                'correlation_risk': correlation_risk,
                'overall_risk_score': risk_score,
                'risk_level': self._determine_risk_level(risk_score),
                'key_risks': self._identify_key_risks(
                    exposure, sector_concentration, position_concentration, correlation_risk
                ),
                'recommendations': self._generate_overall_recommendations(
                    exposure, sector_concentration, position_concentration, correlation_risk
                )
            }
            
            # Store in history
            self.analysis_history.append(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in risk factor analysis: {str(e)}")
            raise

    def _generate_sector_recommendations(
        self,
        sector_weights: Dict[str, float],
        risk_level: str
    ) -> List[str]:
        """Generate sector-based recommendations."""
        recommendations = []
        
        if risk_level in ['high', 'critical']:
            recommendations.append("Reduce concentration in overweight sectors")
            
        if len(sector_weights) < 4:
            recommendations.append("Consider diversifying across more sectors")
            
        # Find underweight sectors
        if sector_weights:
            avg_weight = 1 / len(sector_weights)
            underweight = [s for s, w in sector_weights.items() if w < avg_weight * 0.5]
            if underweight:
                recommendations.append(f"Consider exposure to: {', '.join(underweight[:3])}")
        
        return recommendations

    def _generate_correlation_recommendations(
        self,
        high_correlations: List[Dict[str, Any]],
        risk_level: str
    ) -> List[str]:
        """Generate correlation-based recommendations."""
        recommendations = []
        
        if risk_level in ['high', 'critical']:
            recommendations.append("Reduce positions with high correlation")
            
        if high_correlations:
            # Identify most problematic pairs
            worst_pair = max(high_correlations, key=lambda x: abs(x['correlation']))
            recommendations.append(
                f"Consider reducing exposure to {worst_pair['pair'][0]} or {worst_pair['pair'][1]}"
            )
        
        if risk_level != 'low':
            recommendations.append("Seek uncorrelated or negatively correlated assets")
        
        return recommendations

    def _generate_position_recommendations(
        self,
        max_weight: float,
        top_5_weight: float
    ) -> List[str]:
        """Generate position concentration recommendations."""
        recommendations = []
        
        if max_weight > 0.15:
            recommendations.append("Reduce largest position to below 15% of portfolio")
            
        if top_5_weight > 0.6:
            recommendations.append("Top 5 positions exceed 60% - improve diversification")
            
        if max_weight < 0.05:
            recommendations.append("Consider consolidating very small positions")
        
        return recommendations

    def _calculate_overall_risk_score(
        self,
        exposure: Dict[str, Any],
        sector: Dict[str, Any],
        position: Dict[str, Any],
        correlation: Dict[str, Any]
    ) -> float:
        """Calculate overall portfolio risk score (0-1)."""
        score = 0.0
        
        # Exposure risk (30% weight)
        if exposure['leverage'] > 1.5:
            score += 0.3
        elif exposure['leverage'] > 1.0:
            score += 0.2
        elif exposure['percentage'] > 0.8:
            score += 0.15
        elif exposure['percentage'] > 0.6:
            score += 0.1
        
        # Sector concentration risk (25% weight)
        if sector['concentration_risk'] == 'critical':
            score += 0.25
        elif sector['concentration_risk'] == 'high':
            score += 0.18
        elif sector['concentration_risk'] == 'moderate':
            score += 0.10
        
        # Position concentration risk (25% weight)
        if position['concentration_risk'] == 'high':
            score += 0.25
        elif position['concentration_risk'] == 'moderate':
            score += 0.15
        
        # Correlation risk (20% weight)
        if correlation['correlation_risk'] == 'critical':
            score += 0.20
        elif correlation['correlation_risk'] == 'high':
            score += 0.15
        elif correlation['correlation_risk'] == 'moderate':
            score += 0.10
        
        return min(score, 1.0)

    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score."""
        if risk_score > 0.7:
            return 'critical'
        elif risk_score > 0.5:
            return 'high'
        elif risk_score > 0.3:
            return 'moderate'
        else:
            return 'low'

    def _identify_key_risks(
        self,
        exposure: Dict[str, Any],
        sector: Dict[str, Any],
        position: Dict[str, Any],
        correlation: Dict[str, Any]
    ) -> List[str]:
        """Identify key portfolio risks."""
        risks = []
        
        if exposure['leverage'] > 1.0:
            risks.append(f"Leveraged {exposure['leverage']:.1f}x")
            
        if sector['concentration_risk'] in ['high', 'critical']:
            risks.append(f"High sector concentration ({sector['max_sector_weight']:.1%})")
            
        if position['max_position_weight'] > 0.15:
            risks.append(f"Large position concentration ({position['max_position_weight']:.1%})")
            
        if correlation['max_correlation'] > 0.8:
            risks.append(f"High correlation risk ({correlation['max_correlation']:.2f})")
        
        return risks

    def _generate_overall_recommendations(
        self,
        exposure: Dict[str, Any],
        sector: Dict[str, Any],
        position: Dict[str, Any],
        correlation: Dict[str, Any]
    ) -> List[str]:
        """Generate overall portfolio recommendations."""
        all_recommendations = []
        
        # Add specific recommendations
        if exposure['percentage'] > 0.8:
            all_recommendations.append("Reduce overall exposure to maintain cash buffer")
            
        all_recommendations.extend(sector.get('recommendations', []))
        all_recommendations.extend(position.get('recommendations', []))
        all_recommendations.extend(correlation.get('recommendations', []))
        
        # Remove duplicates while preserving order
        seen = set()
        recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                recommendations.append(rec)
        
        return recommendations[:5]  # Return top 5 recommendations