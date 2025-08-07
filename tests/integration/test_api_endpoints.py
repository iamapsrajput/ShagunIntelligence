"""
Integration tests for API endpoints
Tests all major API endpoints with realistic scenarios
"""

import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.main import app
from tests.mocks.crew_mock import MockCrewManager
from tests.mocks.kite_mock import MockKiteClient


class TestAPIEndpoints:
    """Test suite for API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_kite_client(self):
        """Mock Kite client for testing"""
        return MockKiteClient()

    @pytest.fixture
    def mock_crew_manager(self):
        """Mock CrewManager for testing"""
        return MockCrewManager()

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mock_kite_client, mock_crew_manager):
        """Set up mocks for all tests"""
        with (
            patch("app.main.kite_client", mock_kite_client),
            patch("app.main.crew_manager", mock_crew_manager),
        ):
            yield

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Shagun Intelligence Trading System API"
        assert data["version"] == "2.0.0"
        assert data["status"] == "running"

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data

    def test_system_status_endpoint(self, client):
        """Test system status endpoint"""
        response = client.get("/api/v1/system/status")

        assert response.status_code == 200
        data = response.json()
        assert "is_active" in data
        assert "trading_enabled" in data
        assert "active_agents" in data
        assert "risk_level" in data

    def test_system_metrics_endpoint(self, client):
        """Test system metrics endpoint"""
        response = client.get("/api/v1/system/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "services" in data
        assert "alerts" in data
        assert "timestamp" in data

        # Check metrics structure
        metrics = data["metrics"]
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "disk_usage" in metrics

    def test_start_automated_trading(self, client):
        """Test starting automated trading"""
        response = client.post("/api/v1/automated-trading/start")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert "message" in data

    def test_stop_automated_trading(self, client):
        """Test stopping automated trading"""
        response = client.post("/api/v1/automated-trading/stop")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
        assert "message" in data

    def test_get_automated_trading_status(self, client):
        """Test getting automated trading status"""
        response = client.get("/api/v1/automated-trading/status")

        assert response.status_code == 200
        data = response.json()
        assert "is_running" in data
        assert "trading_enabled" in data
        assert "daily_trades" in data
        assert "daily_pnl" in data

    def test_trigger_analysis(self, client):
        """Test triggering market analysis"""
        response = client.post("/api/v1/automated-trading/analyze")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "trades_executed" in data

    def test_emergency_stop(self, client):
        """Test emergency stop functionality"""
        payload = {"reason": "Test emergency stop"}
        response = client.post("/api/v1/automated-trading/emergency-stop", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "emergency_stop_executed"
        assert data["reason"] == "Test emergency stop"

    def test_get_portfolio_summary(self, client):
        """Test getting portfolio summary"""
        response = client.get("/api/v1/portfolio/summary")

        assert response.status_code == 200
        data = response.json()
        assert "total_value" in data
        assert "invested_value" in data
        assert "day_pnl" in data
        assert "total_pnl" in data

    def test_get_positions(self, client):
        """Test getting current positions"""
        response = client.get("/api/v1/portfolio/positions")

        assert response.status_code == 200
        data = response.json()
        assert "positions" in data
        assert isinstance(data["positions"], list)

    def test_get_holdings(self, client):
        """Test getting holdings"""
        response = client.get("/api/v1/portfolio/holdings")

        assert response.status_code == 200
        data = response.json()
        assert "holdings" in data
        assert isinstance(data["holdings"], list)

    def test_get_market_data(self, client):
        """Test getting market data"""
        response = client.get("/api/v1/market-data/quote/RELIANCE")

        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert "last_price" in data
        assert "volume" in data

    def test_get_market_status(self, client):
        """Test getting market status"""
        response = client.get("/api/v1/market-schedule/status")

        assert response.status_code == 200
        data = response.json()
        assert "is_open" in data
        assert "status" in data
        assert "message" in data

    def test_get_trading_holidays(self, client):
        """Test getting trading holidays"""
        response = client.get("/api/v1/market-schedule/holidays")

        assert response.status_code == 200
        data = response.json()
        assert "holidays" in data
        assert isinstance(data["holidays"], list)

    def test_get_agents_status(self, client):
        """Test getting agents status"""
        response = client.get("/api/v1/agents/status")

        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert isinstance(data["agents"], dict)

    def test_get_agent_performance(self, client):
        """Test getting agent performance"""
        response = client.get("/api/v1/agents/performance")

        assert response.status_code == 200
        data = response.json()
        assert "performance" in data

    def test_get_trading_config(self, client):
        """Test getting trading configuration"""
        response = client.get("/api/v1/system/trading-config")

        assert response.status_code == 200
        data = response.json()
        assert "budget" in data
        assert "risk_level" in data
        assert "max_positions" in data

    def test_update_trading_config(self, client):
        """Test updating trading configuration"""
        config = {
            "budget": 1000,
            "risk_level": "CONSERVATIVE",
            "max_positions": 3,
            "daily_loss_limit": 100,
            "position_size_limit": 300,
            "enable_stop_loss": True,
            "stop_loss_percentage": 3.0,
            "enable_take_profit": True,
            "take_profit_percentage": 6.0,
        }

        response = client.put("/api/v1/system/trading-config", json=config)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_validate_api_keys(self, client):
        """Test API key validation"""
        response = client.post("/api/v1/system/validate-keys")

        assert response.status_code == 200
        data = response.json()
        assert "kite_status" in data

    def test_get_system_settings(self, client):
        """Test getting system settings"""
        response = client.get("/api/v1/system/settings")

        assert response.status_code == 200
        data = response.json()
        assert "trading_enabled" in data
        assert "paper_trading_mode" in data

    def test_update_system_settings(self, client):
        """Test updating system settings"""
        settings = {
            "trading_enabled": True,
            "paper_trading_mode": False,
            "max_daily_trades": 5,
            "max_position_value": 500,
            "market_hours_only": True,
            "allowed_symbols": ["RELIANCE", "TCS"],
            "blocked_symbols": [],
        }

        response = client.put("/api/v1/system/settings", json=settings)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_get_risk_parameters(self, client):
        """Test getting risk parameters"""
        response = client.get("/api/v1/system/risk-parameters")

        assert response.status_code == 200
        data = response.json()
        assert "max_position_size" in data
        assert "max_portfolio_risk" in data
        assert "max_daily_loss" in data

    def test_update_risk_parameters(self, client):
        """Test updating risk parameters"""
        params = {
            "max_position_size": 10.0,
            "max_portfolio_risk": 20.0,
            "max_daily_loss": 5.0,
            "stop_loss_percent": 2.0,
            "take_profit_percent": 4.0,
            "max_open_positions": 5,
            "allow_short_selling": False,
            "use_trailing_stop": True,
            "trailing_stop_percent": 1.5,
        }

        response = client.put("/api/v1/system/risk-parameters", json=params)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_error_handling_invalid_endpoint(self, client):
        """Test error handling for invalid endpoints"""
        response = client.get("/api/v1/invalid-endpoint")

        assert response.status_code == 404

    def test_error_handling_invalid_method(self, client):
        """Test error handling for invalid HTTP methods"""
        response = client.delete("/api/v1/system/status")

        assert response.status_code == 405

    def test_error_handling_invalid_json(self, client):
        """Test error handling for invalid JSON"""
        response = client.post(
            "/api/v1/system/trading-config",
            data="invalid json",
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 422

    def test_rate_limiting(self, client):
        """Test rate limiting functionality"""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = client.get("/api/v1/system/status")
            responses.append(response.status_code)

        # All requests should succeed (rate limiting is lenient in tests)
        assert all(status == 200 for status in responses)

    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/v1/system/status")

        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

    def test_security_headers(self, client):
        """Test security headers are present"""
        response = client.get("/api/v1/system/status")

        # Check for security headers
        assert "x-frame-options" in response.headers
        assert "x-content-type-options" in response.headers
        assert "x-xss-protection" in response.headers
