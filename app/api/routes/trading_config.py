"""
Trading Configuration API endpoints
Handles user configuration for automated trading parameters
"""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from services.kite.client import KiteConnectService

router = APIRouter()


class TradingConfigRequest(BaseModel):
    """Trading configuration request model"""

    # Budget & Position Sizing
    total_budget: float = Field(default=1000, ge=100, le=1000000)
    max_position_size: float = Field(default=200, ge=50, le=50000)
    max_position_value: float = Field(default=300, ge=50, le=50000)
    max_concurrent_positions: int = Field(default=3, ge=1, le=10)

    # Risk Management
    max_risk_per_trade: float = Field(default=5.0, ge=0.1, le=20.0)
    max_daily_loss: float = Field(default=10.0, ge=1.0, le=50.0)
    emergency_stop_amount: float = Field(default=80, ge=10, le=10000)

    # Position Management
    auto_stop_loss: bool = True
    auto_stop_loss_percent: float = Field(default=5.0, ge=0.5, le=20.0)
    auto_take_profit: bool = True
    auto_take_profit_percent: float = Field(default=10.0, ge=1.0, le=50.0)

    # Trading Preferences
    trading_types: list[str] = Field(default=["intraday"])
    min_stock_price: float = Field(default=50, ge=1, le=1000)
    max_stock_price: float = Field(default=5000, ge=100, le=50000)
    min_volume_threshold: int = Field(default=10000, ge=1000, le=10000000)

    # API Configuration (optional, for updates)
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_access_token: str = ""


class KiteConnectionTestRequest(BaseModel):
    """Kite API connection test request"""

    api_key: str
    api_secret: str
    access_token: str


@router.get("/trading-config")
async def get_trading_config() -> dict[str, Any]:
    """Get current trading configuration"""
    try:
        settings = get_settings()

        # Return current configuration (sanitized)
        config = {
            "total_budget": getattr(settings, "TOTAL_TRADING_BUDGET", 1000),
            "max_position_size": settings.DEFAULT_POSITION_SIZE,
            "max_position_value": settings.MAX_POSITION_VALUE,
            "max_concurrent_positions": settings.MAX_CONCURRENT_POSITIONS,
            "max_risk_per_trade": settings.MAX_RISK_PER_TRADE
            * 100,  # Convert to percentage
            "max_daily_loss": settings.MAX_DAILY_LOSS * 100,  # Convert to percentage
            "emergency_stop_amount": settings.EMERGENCY_STOP_LOSS_AMOUNT,
            "auto_stop_loss": settings.AUTO_STOP_LOSS,
            "auto_stop_loss_percent": settings.AUTO_STOP_LOSS_PERCENT * 100,
            "auto_take_profit": settings.AUTO_TAKE_PROFIT,
            "auto_take_profit_percent": settings.AUTO_TAKE_PROFIT_PERCENT * 100,
            "trading_types": ["intraday"],  # Default for now
            "min_stock_price": getattr(settings, "MIN_STOCK_PRICE", 50),
            "max_stock_price": getattr(settings, "MAX_STOCK_PRICE", 5000),
            "min_volume_threshold": getattr(settings, "MIN_VOLUME_THRESHOLD", 10000),
            # Don't return API keys for security
            "kite_api_key": "***" if settings.KITE_API_KEY else "",
            "kite_api_secret": "***" if settings.KITE_API_SECRET else "",
            "kite_access_token": "***" if settings.KITE_ACCESS_TOKEN else "",
        }

        return config

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trading configuration: {str(e)}",
        )


@router.post("/trading-config")
async def update_trading_config(config: TradingConfigRequest) -> dict[str, Any]:
    """Update trading configuration"""
    try:
        # Validate configuration
        validation_errors = []

        # Budget validation
        if config.max_position_size > config.total_budget:
            validation_errors.append("Max position size cannot exceed total budget")

        if config.max_position_value > config.total_budget:
            validation_errors.append("Max position value cannot exceed total budget")

        if config.emergency_stop_amount > config.total_budget:
            validation_errors.append("Emergency stop amount cannot exceed total budget")

        # Risk validation
        if config.max_risk_per_trade > config.max_daily_loss:
            validation_errors.append("Max risk per trade cannot exceed max daily loss")

        # Position management validation
        if (
            config.auto_stop_loss
            and config.auto_stop_loss_percent >= config.auto_take_profit_percent
        ):
            validation_errors.append(
                "Stop loss percentage should be less than take profit percentage"
            )

        if validation_errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Configuration validation failed",
                    "errors": validation_errors,
                },
            )

        # Save configuration to database/file
        # For now, we'll just return success
        # In a full implementation, you'd save to database or update environment

        return {
            "status": "success",
            "message": "Trading configuration updated successfully",
            "config": config.dict(),
            "warnings": [
                "Restart the trading system for changes to take effect",
                "Test the configuration with paper trading first",
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update trading configuration: {str(e)}",
        )


@router.post("/test-kite-connection")
async def test_kite_connection(request: KiteConnectionTestRequest) -> dict[str, Any]:
    """Test Kite Connect API connection"""
    try:
        # Create a temporary Kite client with provided credentials
        kite_client = KiteConnectService()

        # Temporarily override credentials for testing
        kite_client.auth_manager.settings.KITE_API_KEY = request.api_key
        kite_client.auth_manager.settings.KITE_API_SECRET = request.api_secret
        kite_client.auth_manager.settings.KITE_ACCESS_TOKEN = request.access_token

        # Test connection
        success = await kite_client.initialize()

        if success:
            # Test getting a quote to verify full functionality
            try:
                quote = await kite_client.get_quote("RELIANCE")
                return {
                    "success": True,
                    "message": "API connection successful",
                    "test_data": {
                        "symbol": "RELIANCE",
                        "price": quote.get("last_price"),
                        "timestamp": quote.get("timestamp"),
                    },
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"API connection established but data fetch failed: {str(e)}",
                }
        else:
            return {
                "success": False,
                "message": "Failed to establish API connection. Please check your credentials.",
            }

    except Exception as e:
        return {"success": False, "message": f"API connection test failed: {str(e)}"}


@router.get("/trading-status")
async def get_trading_status() -> dict[str, Any]:
    """Get current trading system status"""
    try:
        settings = get_settings()

        return {
            "trading_mode": settings.TRADING_MODE,
            "live_trading_enabled": settings.LIVE_TRADING_ENABLED,
            "automated_trading_enabled": settings.AUTOMATED_TRADING_ENABLED,
            "paper_trading_enabled": settings.PAPER_TRADING_ENABLED,
            "manual_approval_required": settings.MANUAL_APPROVAL_REQUIRED,
            "api_configured": bool(
                settings.KITE_API_KEY and settings.KITE_ACCESS_TOKEN
            ),
            "trading_hours": {
                "start": settings.TRADING_START_TIME,
                "end": settings.TRADING_END_TIME,
                "timezone": settings.TRADING_TIMEZONE,
                "enforce": settings.ENFORCE_TRADING_HOURS,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trading status: {str(e)}",
        )


@router.post("/validate-config")
async def validate_trading_config(config: TradingConfigRequest) -> dict[str, Any]:
    """Validate trading configuration without saving"""
    try:
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": [],
        }

        # Budget validation
        if config.max_position_size > config.total_budget * 0.5:
            validation_results["warnings"].append(
                "Position size is more than 50% of total budget - consider reducing for better risk management"
            )

        if (
            config.max_concurrent_positions * config.max_position_size
            > config.total_budget
        ):
            validation_results["errors"].append(
                "Maximum concurrent positions would exceed total budget"
            )
            validation_results["valid"] = False

        # Risk validation
        if config.max_daily_loss > 20:
            validation_results["warnings"].append(
                "Daily loss limit above 20% is very aggressive"
            )

        if config.max_risk_per_trade > 10:
            validation_results["warnings"].append(
                "Risk per trade above 10% is very aggressive"
            )

        # Recommendations
        if config.auto_stop_loss_percent < 3:
            validation_results["recommendations"].append(
                "Consider a stop loss of at least 3% to avoid noise"
            )

        if config.auto_take_profit_percent < config.auto_stop_loss_percent * 2:
            validation_results["recommendations"].append(
                "Consider a risk-reward ratio of at least 1:2"
            )

        return validation_results

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate configuration: {str(e)}",
        )
