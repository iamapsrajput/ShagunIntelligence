"""Technical Indicator Agent for calculating real-time indicators and generating signals."""

from crewai import Agent
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger
import pandas as pd

from .indicator_calculator import IndicatorCalculator
from .signal_generator import SignalGenerator
from .rolling_data_manager import RollingDataManager
from .visualization_formatter import VisualizationFormatter


class TechnicalIndicatorAgent(Agent):
    """Agent responsible for calculating technical indicators and generating trading signals."""

    def __init__(self):
        """Initialize the Technical Indicator Agent."""
        super().__init__(
            role="Technical Indicator Analyst",
            goal="Calculate real-time technical indicators and generate accurate trading signals",
            backstory="""You are an expert technical analyst with deep knowledge of market indicators.
            You specialize in calculating various technical indicators like RSI, MACD, Bollinger Bands,
            and Moving Averages. You can identify patterns, generate buy/sell signals, and provide
            confidence levels for your predictions. You process data efficiently and provide clear
            visualization-ready data for charts.""",
            verbose=True,
            allow_delegation=False
        )
        
        self.indicator_calculator = IndicatorCalculator()
        self.signal_generator = SignalGenerator()
        self.rolling_data_manager = RollingDataManager()
        self.visualization_formatter = VisualizationFormatter()
        
        logger.info("Technical Indicator Agent initialized")

    async def analyze_symbol(
        self, 
        symbol: str, 
        data: pd.DataFrame,
        timeframe: str = "5min",
        indicators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a symbol with technical indicators.
        
        Args:
            symbol: Stock symbol to analyze
            data: Historical price data (OHLCV)
            timeframe: Time interval (1min, 5min, 15min)
            indicators: List of indicators to calculate (defaults to all)
            
        Returns:
            Analysis results with indicators and signals
        """
        try:
            # Update rolling data manager with new data
            self.rolling_data_manager.update_data(symbol, timeframe, data)
            
            # Get optimized data window for calculations
            calculation_data = self.rolling_data_manager.get_calculation_window(
                symbol, timeframe
            )
            
            # Calculate indicators
            if indicators is None:
                indicators = ["RSI", "MACD", "BB", "SMA", "EMA"]
            
            indicator_results = {}
            for indicator in indicators:
                result = await self._calculate_indicator(
                    indicator, calculation_data, timeframe
                )
                if result:
                    indicator_results[indicator] = result
            
            # Generate trading signals
            signals = self.signal_generator.generate_signals(
                calculation_data, indicator_results
            )
            
            # Format for visualization
            visualization_data = self.visualization_formatter.format_data(
                calculation_data, indicator_results, signals
            )
            
            # Compile analysis results
            analysis = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "indicators": indicator_results,
                "signals": signals,
                "visualization_data": visualization_data,
                "data_points": len(calculation_data)
            }
            
            logger.info(f"Technical analysis completed for {symbol} on {timeframe}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            raise

    async def _calculate_indicator(
        self, 
        indicator_type: str, 
        data: pd.DataFrame,
        timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate a specific indicator.
        
        Args:
            indicator_type: Type of indicator (RSI, MACD, etc.)
            data: Price data for calculation
            timeframe: Time interval
            
        Returns:
            Indicator calculation results
        """
        try:
            if indicator_type == "RSI":
                return self.indicator_calculator.calculate_rsi(data)
            elif indicator_type == "MACD":
                return self.indicator_calculator.calculate_macd(data)
            elif indicator_type == "BB":
                return self.indicator_calculator.calculate_bollinger_bands(data)
            elif indicator_type == "SMA":
                return self.indicator_calculator.calculate_sma(data)
            elif indicator_type == "EMA":
                return self.indicator_calculator.calculate_ema(data)
            else:
                logger.warning(f"Unknown indicator type: {indicator_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating {indicator_type}: {str(e)}")
            return None

    async def get_real_time_signals(
        self, 
        symbol: str,
        timeframe: str = "5min"
    ) -> Dict[str, Any]:
        """
        Get real-time trading signals for a symbol.
        
        Args:
            symbol: Stock symbol
            timeframe: Time interval
            
        Returns:
            Real-time signals with confidence levels
        """
        try:
            # Get latest data
            data = self.rolling_data_manager.get_latest_data(symbol, timeframe)
            if data is None or len(data) < 50:
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                    "message": "Insufficient data for analysis"
                }
            
            # Quick indicator calculation for signals
            indicators = {}
            indicators["RSI"] = self.indicator_calculator.calculate_rsi(data, period=14)
            indicators["MACD"] = self.indicator_calculator.calculate_macd(data)
            
            # Generate signal
            signal = self.signal_generator.get_current_signal(data, indicators)
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                **signal
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time signals for {symbol}: {str(e)}")
            raise

    async def batch_analyze(
        self, 
        symbols: List[str],
        timeframe: str = "5min"
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple symbols in batch.
        
        Args:
            symbols: List of stock symbols
            timeframe: Time interval
            
        Returns:
            List of analysis results
        """
        results = []
        for symbol in symbols:
            try:
                # Get data for symbol (placeholder - would integrate with data service)
                data = self.rolling_data_manager.get_latest_data(symbol, timeframe)
                if data is not None and len(data) >= 50:
                    analysis = await self.analyze_symbol(symbol, data, timeframe)
                    results.append(analysis)
                else:
                    logger.warning(f"Insufficient data for {symbol}")
            except Exception as e:
                logger.error(f"Error in batch analysis for {symbol}: {str(e)}")
                
        return results

    def get_indicator_status(self) -> Dict[str, Any]:
        """Get the status of the technical indicator agent."""
        return {
            "agent": "TechnicalIndicatorAgent",
            "status": "active",
            "supported_indicators": ["RSI", "MACD", "BB", "SMA", "EMA"],
            "supported_timeframes": ["1min", "5min", "15min"],
            "data_manager_status": self.rolling_data_manager.get_status(),
            "signal_generator_status": self.signal_generator.get_status()
        }