import pytest
import asyncio
from typing import Generator, AsyncGenerator
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import types
sys.modules['crewai'] = types.SimpleNamespace(Agent=object, Task=object, Crew=object, Process=object)
sys.modules['langchain_openai'] = types.SimpleNamespace(ChatOpenAI=object)
sys.modules['langchain'] = types.SimpleNamespace()
sys.modules['langchain.llms'] = types.SimpleNamespace()
sys.modules['langchain.llms.base'] = types.SimpleNamespace(BaseLLM=object)
sys.modules['openai'] = types.SimpleNamespace()

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.base_class import Base
from app.db.session import get_db
from app.core.config import get_settings
from app.models.user import User
from app.core.auth import AuthService

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
        is_superuser=False
    )
    db.add(test_user)
    db.commit()
    
    # Login and get token
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "testuser", "password": "testpassword"}
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
        "timestamp": datetime.now()
    }


@pytest.fixture
def sample_historical_data():
    """Sample historical OHLC data"""
    base_date = datetime.now() - timedelta(days=30)
    data = []
    
    for i in range(30):
        date = base_date + timedelta(days=i)
        data.append({
            "date": date,
            "open": 2400 + (i * 10),
            "high": 2420 + (i * 10),
            "low": 2390 + (i * 10),
            "close": 2410 + (i * 10),
            "volume": 1000000 + (i * 10000)
        })
    
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
        "exchange": "NSE"
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
            "risk": {"signal": "approved", "confidence": 0.8}
        }
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
        "trailing_stop_percent": 1.5
    }


# Environment setup for tests
os.environ["TESTING"] = "1"
os.environ["DATABASE_URL"] = SQLALCHEMY_DATABASE_URL
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["ALGORITHM"] = "HS256"
os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "30"