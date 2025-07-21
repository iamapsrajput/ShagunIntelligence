import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.simulation.paper_trading_simulator import PaperTradingSimulator, PaperOrder
from tests.mocks.crew_mock import MockCrewManager


class TestPaperTradingSimulation:
    """Test suite for paper trading simulation"""
    
    @pytest.fixture
    def simulator(self):
        """Create a paper trading simulator instance"""
        return PaperTradingSimulator(initial_capital=1000000)
        
    @pytest.fixture
    def mock_crew_manager(self):
        """Create mock crew manager"""
        return MockCrewManager()
        
    @pytest.mark.asyncio
    async def test_simulator_initialization(self, simulator):
        """Test simulator initialization"""
        assert simulator.initial_capital == 1000000
        assert simulator.available_capital == 1000000
        assert len(simulator.positions) == 0
        assert simulator.total_trades == 0
        
    @pytest.mark.asyncio
    async def test_place_market_order(self, simulator):
        """Test placing a market order"""
        # Set current price
        simulator.current_prices["RELIANCE"] = 2500
        
        # Place buy order
        order = await simulator.place_order({
            "symbol": "RELIANCE",
            "order_type": "BUY",
            "quantity": 100,
            "price_type": "MARKET"
        })
        
        assert order.status == "FILLED"
        assert order.filled_price > 2500  # Should include slippage
        assert "RELIANCE" in simulator.positions
        assert simulator.positions["RELIANCE"].quantity == 100
        
    @pytest.mark.asyncio
    async def test_place_limit_order(self, simulator):
        """Test placing a limit order"""
        simulator.current_prices["TCS"] = 3500
        
        # Place limit buy order below market
        order = await simulator.place_order({
            "symbol": "TCS",
            "order_type": "BUY",
            "quantity": 50,
            "price_type": "LIMIT",
            "price": 3480
        })
        
        assert order.status == "PENDING"
        
        # Update price to trigger order
        await simulator.update_prices({"TCS": 3475})
        
        # Re-execute pending orders
        await simulator._execute_limit_order(order)
        
        assert order.status == "FILLED"
        assert order.filled_price == 3480
        
    @pytest.mark.asyncio
    async def test_close_position(self, simulator):
        """Test closing a position"""
        simulator.current_prices["INFY"] = 1500
        
        # Open position
        buy_order = await simulator.place_order({
            "symbol": "INFY",
            "order_type": "BUY",
            "quantity": 200,
            "price_type": "MARKET"
        })
        
        # Update price
        await simulator.update_prices({"INFY": 1550})
        
        # Close position
        sell_order = await simulator.place_order({
            "symbol": "INFY",
            "order_type": "SELL",
            "quantity": 200,
            "price_type": "MARKET"
        })
        
        assert sell_order.status == "FILLED"
        assert "INFY" not in simulator.positions
        assert simulator.total_trades == 1
        assert simulator.winning_trades == 1
        
    @pytest.mark.asyncio
    async def test_stop_loss_trigger(self, simulator):
        """Test stop loss trigger"""
        simulator.current_prices["HDFC"] = 1600
        
        # Open position with stop loss
        order = await simulator.place_order({
            "symbol": "HDFC",
            "order_type": "BUY",
            "quantity": 100,
            "price_type": "MARKET"
        })
        
        # Set stop loss
        position = simulator.positions["HDFC"]
        position.stop_loss = 1580
        
        # Price drops below stop loss
        await simulator.update_prices({"HDFC": 1575})
        
        # Position should be closed
        assert "HDFC" not in simulator.positions
        assert simulator.total_trades == 1
        assert simulator.losing_trades == 1
        
    @pytest.mark.asyncio
    async def test_take_profit_trigger(self, simulator):
        """Test take profit trigger"""
        simulator.current_prices["ICICIBANK"] = 900
        
        # Open position with take profit
        order = await simulator.place_order({
            "symbol": "ICICIBANK",
            "order_type": "BUY",
            "quantity": 150,
            "price_type": "MARKET"
        })
        
        # Set take profit
        position = simulator.positions["ICICIBANK"]
        position.take_profit = 920
        
        # Price reaches take profit
        await simulator.update_prices({"ICICIBANK": 925})
        
        # Position should be closed
        assert "ICICIBANK" not in simulator.positions
        assert simulator.total_trades == 1
        assert simulator.winning_trades == 1
        
    @pytest.mark.asyncio
    async def test_portfolio_metrics(self, simulator):
        """Test portfolio metrics calculation"""
        # Execute some trades
        simulator.current_prices = {
            "RELIANCE": 2500,
            "TCS": 3500,
            "INFY": 1500
        }
        
        # Trade 1: Winning trade
        await simulator.place_order({
            "symbol": "RELIANCE",
            "order_type": "BUY",
            "quantity": 100,
            "price_type": "MARKET"
        })
        
        await simulator.update_prices({"RELIANCE": 2600})
        
        await simulator.place_order({
            "symbol": "RELIANCE",
            "order_type": "SELL",
            "quantity": 100,
            "price_type": "MARKET"
        })
        
        # Trade 2: Losing trade
        await simulator.place_order({
            "symbol": "TCS",
            "order_type": "BUY",
            "quantity": 50,
            "price_type": "MARKET"
        })
        
        await simulator.update_prices({"TCS": 3400})
        
        await simulator.place_order({
            "symbol": "TCS",
            "order_type": "SELL",
            "quantity": 50,
            "price_type": "MARKET"
        })
        
        # Get metrics
        metrics = simulator.get_performance_metrics()
        
        assert metrics["total_trades"] == 2
        assert metrics["winning_trades"] == 1
        assert metrics["losing_trades"] == 1
        assert metrics["win_rate"] == 50.0
        assert metrics["total_commission"] > 0
        
    @pytest.mark.asyncio
    async def test_position_sizing_limits(self, simulator):
        """Test position sizing limits"""
        simulator.current_prices["RELIANCE"] = 2500
        
        # Try to place order exceeding position size limit
        large_order = await simulator.place_order({
            "symbol": "RELIANCE",
            "order_type": "BUY",
            "quantity": 1000,  # Would be 25% of capital
            "price_type": "MARKET"
        })
        
        assert large_order.status == "REJECTED"
        
    @pytest.mark.asyncio
    async def test_margin_trading(self, simulator):
        """Test margin trading functionality"""
        simulator.current_prices["SBIN"] = 500
        simulator.margin_requirement = 0.2  # 5x leverage
        
        # With margin, should be able to buy more
        order = await simulator.place_order({
            "symbol": "SBIN",
            "order_type": "BUY",
            "quantity": 1000,
            "price_type": "MARKET"
        })
        
        assert order.status == "FILLED"
        # Only 20% margin required
        assert simulator.available_capital > 800000
        
    @pytest.mark.asyncio
    async def test_drawdown_calculation(self, simulator):
        """Test maximum drawdown calculation"""
        # Set initial prices
        prices = {
            "RELIANCE": 2500,
            "TCS": 3500,
            "INFY": 1500
        }
        simulator.current_prices = prices
        
        # Open positions
        for symbol in prices:
            await simulator.place_order({
                "symbol": symbol,
                "order_type": "BUY",
                "quantity": 100,
                "price_type": "MARKET"
            })
            
        # Simulate price movements
        # Peak
        await simulator.update_prices({
            "RELIANCE": 2600,
            "TCS": 3600,
            "INFY": 1550
        })
        
        peak_value = simulator.get_portfolio_value()
        
        # Drawdown
        await simulator.update_prices({
            "RELIANCE": 2400,
            "TCS": 3300,
            "INFY": 1400
        })
        
        metrics = simulator.get_performance_metrics()
        assert metrics["max_drawdown_percent"] > 0
        
    @pytest.mark.asyncio
    async def test_multi_day_simulation(self, simulator, mock_crew_manager):
        """Test multi-day trading simulation"""
        # Simulate 5 days of trading
        for day in range(5):
            # Morning: Analyze and place trades
            symbols = ["RELIANCE", "TCS", "INFY"]
            
            for symbol in symbols:
                # Get AI analysis
                analysis = await mock_crew_manager.analyze_trade_opportunity(symbol)
                
                if analysis["confidence"] > 0.7:
                    # Set price
                    simulator.current_prices[symbol] = analysis["entry_price"]
                    
                    # Place trade based on recommendation
                    if analysis["recommendation"] == "buy":
                        await simulator.place_order({
                            "symbol": symbol,
                            "order_type": "BUY",
                            "quantity": 100,
                            "price_type": "MARKET"
                        })
                        
            # Simulate intraday price movements
            for hour in range(6):  # 6 hours of trading
                price_updates = {}
                for symbol in simulator.positions:
                    # Random walk
                    current = simulator.current_prices[symbol]
                    change = current * 0.01 * (2 * (hour % 2) - 1)  # ±1%
                    price_updates[symbol] = current + change
                    
                await simulator.update_prices(price_updates)
                
            # End of day: Close all positions
            for symbol in list(simulator.positions.keys()):
                await simulator.place_order({
                    "symbol": symbol,
                    "order_type": "SELL",
                    "quantity": simulator.positions[symbol].quantity,
                    "price_type": "MARKET"
                })
                
        # Check results
        metrics = simulator.get_performance_metrics()
        assert metrics["total_trades"] >= 0
        assert len(simulator.trade_history) > 0
        
    @pytest.mark.asyncio
    async def test_save_results(self, simulator, tmp_path):
        """Test saving simulation results"""
        # Execute some trades
        simulator.current_prices["RELIANCE"] = 2500
        
        await simulator.place_order({
            "symbol": "RELIANCE",
            "order_type": "BUY",
            "quantity": 100,
            "price_type": "MARKET"
        })
        
        # Save results
        results_file = tmp_path / "simulation_results.json"
        simulator.save_results(results_file)
        
        assert results_file.exists()
        
        # Load and verify
        import json
        with open(results_file) as f:
            results = json.load(f)
            
        assert "performance_metrics" in results
        assert "trade_history" in results
        assert "portfolio_history" in results