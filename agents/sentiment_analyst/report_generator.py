"""Report generator for daily sentiment analysis and trends."""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
import json
from collections import defaultdict
import statistics


class ReportGenerator:
    """Generate comprehensive sentiment analysis reports."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.report_templates = {
            'daily': self._daily_report_template,
            'weekly': self._weekly_report_template,
            'alert_summary': self._alert_summary_template
        }
        
        self.trend_periods = {
            'short': 1,  # 1 day
            'medium': 7,  # 1 week
            'long': 30   # 1 month
        }
        
        logger.info("ReportGenerator initialized")

    def generate_daily_report(
        self,
        symbol_analyses: Dict[str, Dict[str, Any]],
        market_analysis: Dict[str, Any],
        alerts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive daily sentiment report.
        
        Args:
            symbol_analyses: Individual symbol sentiment analyses
            market_analysis: Overall market sentiment
            alerts: Recent alerts
            
        Returns:
            Daily report data
        """
        try:
            # Calculate aggregated metrics
            aggregated_metrics = self._aggregate_symbol_metrics(symbol_analyses)
            
            # Identify top movers
            top_movers = self._identify_top_movers(symbol_analyses)
            
            # Generate trend analysis
            trend_analysis = self._analyze_trends(symbol_analyses)
            
            # Create report structure
            report = {
                'meta': {
                    'report_type': 'daily_sentiment',
                    'generated_at': datetime.now().isoformat(),
                    'period': {
                        'start': (datetime.now() - timedelta(days=1)).isoformat(),
                        'end': datetime.now().isoformat()
                    }
                },
                'executive_summary': self._generate_executive_summary(
                    aggregated_metrics, market_analysis, top_movers
                ),
                'market_overview': {
                    'overall_sentiment': market_analysis.get('market_sentiment', {}),
                    'market_mood': market_analysis.get('market_mood', 'neutral'),
                    'sector_performance': market_analysis.get('sector_sentiments', {}),
                    'trending_topics': market_analysis.get('trending_topics', [])[:5]
                },
                'symbol_analysis': {
                    'total_symbols_analyzed': len(symbol_analyses),
                    'aggregated_metrics': aggregated_metrics,
                    'top_positive': top_movers['positive'][:5],
                    'top_negative': top_movers['negative'][:5],
                    'high_confidence': top_movers['high_confidence'][:5]
                },
                'detailed_analyses': self._format_symbol_details(symbol_analyses),
                'trend_analysis': trend_analysis,
                'alerts_summary': self._summarize_alerts(alerts) if alerts else None,
                'recommendations': self._generate_recommendations(
                    aggregated_metrics, top_movers, market_analysis
                ),
                'risk_assessment': self._assess_market_risk(
                    aggregated_metrics, market_analysis
                )
            }
            
            # Add visualizations data
            report['visualizations'] = self._prepare_visualization_data(
                symbol_analyses, market_analysis
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
            raise

    def generate_intraday_update(
        self,
        current_analyses: Dict[str, Dict[str, Any]],
        previous_analyses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate intraday sentiment update comparing to previous period."""
        try:
            changes = {}
            
            for symbol in current_analyses:
                if symbol in previous_analyses:
                    current_score = current_analyses[symbol]['sentiment_scores']['overall_score']
                    previous_score = previous_analyses[symbol]['sentiment_scores']['overall_score']
                    
                    change = current_score - previous_score
                    changes[symbol] = {
                        'current': current_score,
                        'previous': previous_score,
                        'change': change,
                        'change_percent': (change / abs(previous_score) * 100) if previous_score != 0 else 0
                    }
            
            # Sort by absolute change
            significant_changes = sorted(
                changes.items(),
                key=lambda x: abs(x[1]['change']),
                reverse=True
            )[:10]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'significant_changes': significant_changes,
                'summary': self._summarize_changes(changes)
            }
            
        except Exception as e:
            logger.error(f"Error generating intraday update: {str(e)}")
            return {}

    def _aggregate_symbol_metrics(
        self,
        symbol_analyses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate metrics across all analyzed symbols."""
        if not symbol_analyses:
            return {}
        
        sentiment_scores = []
        confidence_scores = []
        news_counts = []
        social_counts = []
        
        for analysis in symbol_analyses.values():
            scores = analysis.get('sentiment_scores', {})
            sentiment_scores.append(scores.get('overall_score', 0))
            confidence_scores.append(scores.get('confidence', 0))
            
            news = analysis.get('news_analysis', {})
            news_counts.append(news.get('total_articles', 0))
            
            social = analysis.get('social_analysis', {})
            if social:
                social_counts.append(social.get('total_posts', 0))
        
        return {
            'average_sentiment': statistics.mean(sentiment_scores) if sentiment_scores else 0,
            'sentiment_std_dev': statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0,
            'average_confidence': statistics.mean(confidence_scores) if confidence_scores else 0,
            'total_news_articles': sum(news_counts),
            'total_social_posts': sum(social_counts),
            'sentiment_distribution': self._calculate_distribution(sentiment_scores)
        }

    def _identify_top_movers(
        self,
        symbol_analyses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Identify top positive, negative, and high-confidence symbols."""
        symbol_data = []
        
        for symbol, analysis in symbol_analyses.items():
            scores = analysis.get('sentiment_scores', {})
            symbol_data.append({
                'symbol': symbol,
                'sentiment_score': scores.get('overall_score', 0),
                'confidence': scores.get('confidence', 0),
                'change': analysis.get('sentiment_change', 0),
                'market_impact': analysis.get('market_impact', 'unknown')
            })
        
        # Sort by different criteria
        positive = sorted(
            [s for s in symbol_data if s['sentiment_score'] > 0],
            key=lambda x: x['sentiment_score'],
            reverse=True
        )
        
        negative = sorted(
            [s for s in symbol_data if s['sentiment_score'] < 0],
            key=lambda x: x['sentiment_score']
        )
        
        high_confidence = sorted(
            symbol_data,
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        return {
            'positive': positive,
            'negative': negative,
            'high_confidence': high_confidence
        }

    def _analyze_trends(
        self,
        symbol_analyses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze sentiment trends."""
        improving = []
        deteriorating = []
        volatile = []
        
        for symbol, analysis in symbol_analyses.items():
            change = analysis.get('sentiment_change', 0)
            
            if change > 0.2:
                improving.append({
                    'symbol': symbol,
                    'change': change,
                    'current_sentiment': analysis['sentiment_scores']['overall_score']
                })
            elif change < -0.2:
                deteriorating.append({
                    'symbol': symbol,
                    'change': change,
                    'current_sentiment': analysis['sentiment_scores']['overall_score']
                })
            
            # Check for volatility (would need historical data)
            # Placeholder for volatility detection
        
        return {
            'improving_sentiment': sorted(improving, key=lambda x: x['change'], reverse=True),
            'deteriorating_sentiment': sorted(deteriorating, key=lambda x: x['change']),
            'volatile_sentiment': volatile
        }

    def _generate_executive_summary(
        self,
        aggregated_metrics: Dict[str, Any],
        market_analysis: Dict[str, Any],
        top_movers: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Generate executive summary of the report."""
        avg_sentiment = aggregated_metrics.get('average_sentiment', 0)
        market_mood = market_analysis.get('market_mood', 'neutral')
        
        sentiment_desc = "positive" if avg_sentiment > 0.2 else "negative" if avg_sentiment < -0.2 else "neutral"
        
        summary = f"Market sentiment is {sentiment_desc} with an average score of {avg_sentiment:.2f}. "
        summary += f"The overall market mood is {market_mood}. "
        
        if top_movers['positive']:
            top_positive = top_movers['positive'][0]
            summary += f"{top_positive['symbol']} leads positive sentiment at {top_positive['sentiment_score']:.2f}. "
        
        if top_movers['negative']:
            top_negative = top_movers['negative'][0]
            summary += f"{top_negative['symbol']} shows the most negative sentiment at {top_negative['sentiment_score']:.2f}. "
        
        total_coverage = aggregated_metrics.get('total_news_articles', 0) + aggregated_metrics.get('total_social_posts', 0)
        summary += f"Analysis based on {total_coverage} total media items."
        
        return summary

    def _format_symbol_details(
        self,
        symbol_analyses: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format detailed symbol analyses for report."""
        formatted = []
        
        for symbol, analysis in symbol_analyses.items():
            scores = analysis.get('sentiment_scores', {})
            news = analysis.get('news_analysis', {})
            recommendation = analysis.get('recommendation', {})
            
            formatted.append({
                'symbol': symbol,
                'sentiment': {
                    'score': scores.get('overall_score', 0),
                    'confidence': scores.get('confidence', 0),
                    'change': analysis.get('sentiment_change', 0)
                },
                'coverage': {
                    'news_articles': news.get('total_articles', 0),
                    'social_posts': analysis.get('social_analysis', {}).get('total_posts', 0) if analysis.get('social_analysis') else 0
                },
                'key_stories': [
                    {
                        'title': article.get('title', ''),
                        'sentiment': article.get('sentiment', {}).get('score', 0),
                        'source': article.get('source', '')
                    }
                    for article in news.get('articles', [])[:3]
                ],
                'recommendation': recommendation.get('action', 'hold'),
                'risk_level': recommendation.get('risk_level', 'medium')
            })
        
        # Sort by sentiment score
        formatted.sort(key=lambda x: abs(x['sentiment']['score']), reverse=True)
        
        return formatted

    def _summarize_alerts(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize alerts for the report."""
        if not alerts:
            return {'total_alerts': 0}
        
        alert_types = defaultdict(int)
        alert_symbols = defaultdict(int)
        high_priority_alerts = []
        
        for alert in alerts:
            alert_types[alert.get('type', 'unknown')] += 1
            if 'symbol' in alert:
                alert_symbols[alert['symbol']] += 1
            
            if alert.get('priority') in ['critical', 'high']:
                high_priority_alerts.append({
                    'symbol': alert.get('symbol', 'N/A'),
                    'type': alert.get('type', 'unknown'),
                    'message': alert.get('message', ''),
                    'timestamp': alert.get('timestamp', datetime.now()).isoformat()
                })
        
        return {
            'total_alerts': len(alerts),
            'alerts_by_type': dict(alert_types),
            'most_alerted_symbols': sorted(
                alert_symbols.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'high_priority_alerts': high_priority_alerts[:10]
        }

    def _generate_recommendations(
        self,
        aggregated_metrics: Dict[str, Any],
        top_movers: Dict[str, List[Dict[str, Any]]],
        market_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Market-level recommendations
        avg_sentiment = aggregated_metrics.get('average_sentiment', 0)
        if avg_sentiment > 0.5:
            recommendations.append({
                'type': 'market',
                'action': 'consider_risk_on',
                'reasoning': 'Strong positive market sentiment',
                'confidence': 'high'
            })
        elif avg_sentiment < -0.5:
            recommendations.append({
                'type': 'market',
                'action': 'consider_risk_off',
                'reasoning': 'Strong negative market sentiment',
                'confidence': 'high'
            })
        
        # Symbol-specific recommendations
        for symbol_data in top_movers['positive'][:3]:
            if symbol_data['confidence'] > 0.7:
                recommendations.append({
                    'type': 'symbol',
                    'symbol': symbol_data['symbol'],
                    'action': 'monitor_for_entry',
                    'reasoning': f"High positive sentiment ({symbol_data['sentiment_score']:.2f})",
                    'confidence': 'medium'
                })
        
        for symbol_data in top_movers['negative'][:3]:
            if symbol_data['confidence'] > 0.7:
                recommendations.append({
                    'type': 'symbol',
                    'symbol': symbol_data['symbol'],
                    'action': 'review_positions',
                    'reasoning': f"High negative sentiment ({symbol_data['sentiment_score']:.2f})",
                    'confidence': 'medium'
                })
        
        return recommendations

    def _assess_market_risk(
        self,
        aggregated_metrics: Dict[str, Any],
        market_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall market risk based on sentiment."""
        sentiment_std = aggregated_metrics.get('sentiment_std_dev', 0)
        avg_confidence = aggregated_metrics.get('average_confidence', 0)
        market_mood = market_analysis.get('market_mood', 'neutral')
        
        # Risk factors
        risk_score = 0
        risk_factors = []
        
        # High dispersion = higher risk
        if sentiment_std > 0.5:
            risk_score += 0.3
            risk_factors.append('High sentiment dispersion')
        
        # Low confidence = higher risk
        if avg_confidence < 0.5:
            risk_score += 0.2
            risk_factors.append('Low analysis confidence')
        
        # Mixed market mood = higher risk
        if market_mood == 'mixed':
            risk_score += 0.2
            risk_factors.append('Mixed market signals')
        
        # Determine risk level
        if risk_score > 0.6:
            risk_level = 'high'
        elif risk_score > 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendation': self._get_risk_recommendation(risk_level)
        }

    def _prepare_visualization_data(
        self,
        symbol_analyses: Dict[str, Dict[str, Any]],
        market_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for visualization charts."""
        # Sentiment heatmap data
        heatmap_data = []
        for symbol, analysis in symbol_analyses.items():
            scores = analysis.get('sentiment_scores', {})
            heatmap_data.append({
                'symbol': symbol,
                'sentiment': scores.get('overall_score', 0),
                'confidence': scores.get('confidence', 0),
                'change': analysis.get('sentiment_change', 0)
            })
        
        # Sector breakdown
        sector_data = []
        for sector, data in market_analysis.get('sector_sentiments', {}).items():
            sector_data.append({
                'sector': sector,
                'sentiment': data.get('average_sentiment', 0),
                'articles': data.get('article_count', 0)
            })
        
        return {
            'sentiment_heatmap': heatmap_data,
            'sector_breakdown': sector_data,
            'trend_data': self._prepare_trend_chart_data(symbol_analyses),
            'distribution_chart': self._prepare_distribution_data(symbol_analyses)
        }

    def _calculate_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate sentiment score distribution."""
        distribution = {
            'very_positive': 0,
            'positive': 0,
            'neutral': 0,
            'negative': 0,
            'very_negative': 0
        }
        
        for score in scores:
            if score > 0.6:
                distribution['very_positive'] += 1
            elif score > 0.2:
                distribution['positive'] += 1
            elif score > -0.2:
                distribution['neutral'] += 1
            elif score > -0.6:
                distribution['negative'] += 1
            else:
                distribution['very_negative'] += 1
        
        return distribution

    def _summarize_changes(self, changes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize sentiment changes."""
        improving = sum(1 for c in changes.values() if c['change'] > 0.1)
        deteriorating = sum(1 for c in changes.values() if c['change'] < -0.1)
        stable = len(changes) - improving - deteriorating
        
        avg_change = statistics.mean([c['change'] for c in changes.values()]) if changes else 0
        
        return {
            'total_symbols': len(changes),
            'improving': improving,
            'deteriorating': deteriorating,
            'stable': stable,
            'average_change': avg_change
        }

    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level."""
        recommendations = {
            'high': 'Consider reducing positions and tightening stop losses',
            'medium': 'Monitor positions closely and maintain normal risk controls',
            'low': 'Market conditions favorable for normal trading activity'
        }
        return recommendations.get(risk_level, 'Monitor market conditions')

    def _prepare_trend_chart_data(
        self,
        symbol_analyses: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare data for trend visualization."""
        # This would include historical data in a real implementation
        return []

    def _prepare_distribution_data(
        self,
        symbol_analyses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare sentiment distribution data."""
        scores = [
            analysis['sentiment_scores']['overall_score']
            for analysis in symbol_analyses.values()
        ]
        
        return {
            'bins': [-1, -0.6, -0.2, 0.2, 0.6, 1],
            'counts': self._calculate_distribution(scores),
            'labels': ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        }

    def _daily_report_template(self, data: Dict[str, Any]) -> str:
        """Template for daily report formatting."""
        # This would generate formatted text/HTML report
        return json.dumps(data, indent=2)

    def _weekly_report_template(self, data: Dict[str, Any]) -> str:
        """Template for weekly report formatting."""
        return json.dumps(data, indent=2)

    def _alert_summary_template(self, data: Dict[str, Any]) -> str:
        """Template for alert summary formatting."""
        return json.dumps(data, indent=2)

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the report generator."""
        return {
            'status': 'active',
            'available_reports': list(self.report_templates.keys()),
            'trend_periods': self.trend_periods
        }