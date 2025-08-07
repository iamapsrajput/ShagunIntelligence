import asyncio
import os
import sys
from collections.abc import Generator
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import types


class MockChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass


class MockAgent:
    def __init__(self, *args, **kwargs):
        pass


class MockTask:
    def __init__(self, *args, **kwargs):
        pass


class MockCrew:
    def __init__(self, *args, **kwargs):
        pass

    def kickoff(self, *args, **kwargs):
        return {"result": "mock_result"}


class MockProcess:
    sequential = "sequential"


sys.modules["crewai"] = types.SimpleNamespace(
    Agent=MockAgent, Task=MockTask, Crew=MockCrew, Process=MockProcess
)
sys.modules["langchain_openai"] = types.SimpleNamespace(ChatOpenAI=MockChatOpenAI)
sys.modules["langchain"] = types.SimpleNamespace()
sys.modules["langchain.llms"] = types.SimpleNamespace()
sys.modules["langchain.llms.base"] = types.SimpleNamespace(BaseLLM=object)
sys.modules["openai"] = types.SimpleNamespace()

from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.auth import AuthService
from app.db.base_class import Base
from app.db.session import get_db
from app.main import app
from app.models.user import User

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def db() -> Generator[Session, None, None]:
    """Create a test database session"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db: Session) -> Generator[TestClient, None, None]:
    """Create a test client"""

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def authenticated_client(client: TestClient, db: Session) -> TestClient:
    """Create an authenticated test client"""
    # Create test user
    test_user = User(
        username="testuser",
        email="test@example.com",
        full_name="Test User",
        hashed_password=AuthService.get_password_hash("testpassword"),
        is_active=True,
        is_superuser=False,
    )
    db.add(test_user)
    db.commit()

    # Login and get token
    response = client.post(
        "/api/v1/auth/token", data={"username": "testuser", "password": "testpassword"}
    )
    token = response.json()["access_token"]

    # Set authorization header
    client.headers["Authorization"] = f"Bearer {token}"

    return client


@pytest.fixture
def mock_kite_client():
    """Mock Kite client for testing"""
    from tests.mocks.kite_mock import MockKiteClient

    return MockKiteClient()


@pytest.fixture
def mock_crew_manager():
    """Mock CrewAI manager for testing"""
    from tests.mocks.crew_mock import MockCrewManager

    return MockCrewManager()


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        "symbol": "RELIANCE",
        "last_price": 2500.50,
        "change": 25.50,
        "change_percent": 1.03,
        "volume": 1234567,
        "bid": 2500.25,
        "ask": 2500.75,
        "open": 2475.00,
        "high": 2510.00,
        "low": 2470.00,
        "close": 2475.00,
        "timestamp": datetime.now(),
    }


@pytest.fixture
def sample_historical_data():
    """Sample historical OHLC data"""
    base_date = datetime.now() - timedelta(days=30)
    data = []

    for i in range(30):
        date = base_date + timedelta(days=i)
        data.append(
            {
                "date": date,
                "open": 2400 + (i * 10),
                "high": 2420 + (i * 10),
                "low": 2390 + (i * 10),
                "close": 2410 + (i * 10),
                "volume": 1000000 + (i * 10000),
            }
        )

    return data


@pytest.fixture
def sample_position():
    """Sample trading position"""
    return {
        "symbol": "RELIANCE",
        "quantity": 100,
        "average_price": 2450.00,
        "last_price": 2500.00,
        "pnl": 5000.00,
        "pnl_percent": 2.04,
        "product": "MIS",
        "exchange": "NSE",
    }


@pytest.fixture
def sample_trade_signal():
    """Sample trade signal for testing"""
    return {
        "symbol": "RELIANCE",
        "action": "BUY",
        "quantity": 100,
        "confidence": 0.85,
        "entry_price": 2500.00,
        "stop_loss": 2450.00,
        "take_profit": 2600.00,
        "rationale": "Strong bullish momentum with technical indicators aligned",
        "agent_decisions": {
            "market": {"signal": "buy", "confidence": 0.8},
            "technical": {"signal": "buy", "confidence": 0.9},
            "sentiment": {"signal": "buy", "confidence": 0.85},
            "risk": {"signal": "approved", "confidence": 0.8},
        },
    }


@pytest.fixture
def risk_parameters():
    """Sample risk parameters"""
    return {
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


# Advanced Trading System Fixtures


@pytest.fixture
def enhanced_mock_market_data_service():
    """Enhanced mock market data service for advanced testing"""
    mock_service = AsyncMock()

    # Mock quote data
    mock_service.get_quote.return_value = {
        "price": 2500.0,
        "volume": 100000,
        "bid": 2499.5,
        "ask": 2500.5,
        "last_trade_time": "2025-01-07T10:30:00Z",
        "change": 25.0,
        "change_percent": 1.0,
    }

    # Mock market depth
    mock_service.get_market_depth.return_value = {
        "bid": [
            {"price": 2499.5, "quantity": 1000},
            {"price": 2499.0, "quantity": 2000},
            {"price": 2498.5, "quantity": 1500},
        ],
        "ask": [
            {"price": 2500.5, "quantity": 1500},
            {"price": 2501.0, "quantity": 1800},
            {"price": 2501.5, "quantity": 1200},
        ],
        "spread": 1.0,
    }

    # Mock volume profile
    mock_service.get_volume_profile.return_value = [
        5000,
        8000,
        12000,
        15000,
        18000,
        20000,
        22000,
        18000,
        15000,
        12000,
        10000,
        8000,
        6000,
        4000,
        3000,
        2000,
        1500,
        1200,
        1000,
        800,
    ]

    # Mock historical data
    async def mock_get_historical_data(symbol: str, timeframe: str, limit: int = 100):
        dates = pd.date_range(end=datetime.now(), periods=limit, freq="1H")
        np.random.seed(hash(symbol) % 1000)  # Consistent data per symbol

        base_price = 2500
        returns = np.random.normal(0, 0.02, limit)
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.001, limit)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, limit))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, limit))),
                "close": prices,
                "volume": np.random.randint(10000, 100000, limit),
            },
            index=dates,
        )

    mock_service.get_historical_data = mock_get_historical_data
    return mock_service


@pytest.fixture
def sample_portfolio_positions():
    """Sample portfolio positions for testing"""
    return {
        "RELIANCE": {
            "shares": 100,
            "average_price": 2450.0,
            "current_price": 2500.0,
            "market_value": 250000,
            "pnl": 5000,
            "pnl_percent": 2.04,
        },
        "TCS": {
            "shares": 50,
            "average_price": 3500.0,
            "current_price": 3600.0,
            "market_value": 180000,
            "pnl": 5000,
            "pnl_percent": 2.86,
        },
        "HDFCBANK": {
            "shares": 75,
            "average_price": 1550.0,
            "current_price": 1600.0,
            "market_value": 120000,
            "pnl": 3750,
            "pnl_percent": 3.23,
        },
    }


@pytest.fixture
def sample_risk_limits():
    """Sample risk limits configuration for testing"""
    from app.services.enhanced_risk_management import RiskLimits

    return RiskLimits(
        max_portfolio_var=0.05,
        max_sector_exposure=0.30,
        max_single_position=0.10,
        max_correlation=0.70,
        max_leverage=2.0,
        max_drawdown=0.15,
        min_sharpe_ratio=0.5,
        max_beta=1.5,
    )


@pytest.fixture
def create_test_data():
    """Factory function to create test market data"""

    def _create_data(
        symbol: str,
        periods: int = 100,
        frequency: str = "1D",
        base_price: float = 2500,
        volatility: float = 0.02,
    ):
        """Create realistic test market data"""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=frequency)
        np.random.seed(hash(symbol) % 1000)

        returns = np.random.normal(0, volatility, periods)
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.001, periods)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
                "close": prices,
                "volume": np.random.lognormal(10, 0.5, periods).astype(int),
            },
            index=dates,
        )

    return _create_data


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "load: Load and performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_data: Tests requiring market data")


# Environment setup for tests
os.environ["TESTING"] = "1"
os.environ["DATABASE_URL"] = SQLALCHEMY_DATABASE_URL
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["ALGORITHM"] = "HS256"
os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "30"
