"""
API endpoints for automated trading control
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import get_current_user
from app.models.user import User
from app.services.automated_trading import automated_trading_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/start", response_model=dict[str, Any])
async def start_automated_trading(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Start the automated trading system

    This will enable fully automated trading where the multi-agent system
    will analyze market conditions and execute trades automatically.
    """
    try:
        logger.info(f"User {current_user.username} starting automated trading")

        result = await automated_trading_service.start_automated_trading()

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"]
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting automated trading: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start automated trading: {str(e)}",
        )


@router.post("/stop", response_model=dict[str, Any])
async def stop_automated_trading(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Stop the automated trading system

    This will halt all automated trading activities while preserving
    existing positions (unless configured otherwise).
    """
    try:
        logger.info(f"User {current_user.username} stopping automated trading")

        result = await automated_trading_service.stop_automated_trading()

        return result

    except Exception as e:
        logger.error(f"Error stopping automated trading: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop automated trading: {str(e)}",
        )


@router.post("/emergency-stop", response_model=dict[str, Any])
async def emergency_stop_trading(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Emergency stop - immediately halt all trading and close positions

    This is a critical safety feature that will:
    1. Immediately stop all automated trading
    2. Close all open positions
    3. Prevent any new trades from being executed
    """
    try:
        logger.critical(f"User {current_user.username} triggered EMERGENCY STOP")

        result = await automated_trading_service.emergency_stop()

        return result

    except Exception as e:
        logger.error(f"Error during emergency stop: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Emergency stop failed: {str(e)}",
        )


@router.get("/status", response_model=dict[str, Any])
async def get_automated_trading_status(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get current status of the automated trading system

    Returns information about:
    - Whether automated trading is running
    - Current market hours status
    - Daily P&L
    - Active positions count
    - Configuration settings
    """
    try:
        status_info = await automated_trading_service.get_status()

        return {"status": "success", "data": status_info}

    except Exception as e:
        logger.error(f"Error getting automated trading status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}",
        )


@router.get("/config", response_model=dict[str, Any])
async def get_automated_trading_config(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get current automated trading configuration

    Returns all safety limits and trading parameters.
    """
    try:
        from app.core.config import get_settings

        settings = get_settings()

        config = {
            "trading_mode": settings.TRADING_MODE,
            "automated_trading_enabled": settings.AUTOMATED_TRADING_ENABLED,
            "live_trading_enabled": settings.LIVE_TRADING_ENABLED,
            "manual_approval_required": settings.MANUAL_APPROVAL_REQUIRED,
            "risk_management": {
                "max_risk_per_trade": settings.MAX_RISK_PER_TRADE,
                "max_daily_loss": settings.MAX_DAILY_LOSS,
                "default_position_size": settings.DEFAULT_POSITION_SIZE,
                "max_position_value": settings.MAX_POSITION_VALUE,
                "min_position_size": settings.MIN_POSITION_SIZE,
                "max_concurrent_positions": settings.MAX_CONCURRENT_POSITIONS,
            },
            "circuit_breaker": {
                "enabled": settings.CIRCUIT_BREAKER_ENABLED,
                "loss_percent": settings.CIRCUIT_BREAKER_LOSS_PERCENT,
                "emergency_stop_amount": settings.EMERGENCY_STOP_LOSS_AMOUNT,
            },
            "trading_hours": {
                "enforce_hours": settings.ENFORCE_TRADING_HOURS,
                "start_time": settings.TRADING_START_TIME,
                "end_time": settings.TRADING_END_TIME,
                "timezone": settings.TRADING_TIMEZONE,
            },
            "position_management": {
                "auto_stop_loss": settings.AUTO_STOP_LOSS,
                "auto_stop_loss_percent": settings.AUTO_STOP_LOSS_PERCENT,
                "auto_take_profit": settings.AUTO_TAKE_PROFIT,
                "auto_take_profit_percent": settings.AUTO_TAKE_PROFIT_PERCENT,
            },
        }

        return {"status": "success", "config": config}

    except Exception as e:
        logger.error(f"Error getting automated trading config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get config: {str(e)}",
        )


@router.post("/validate-config", response_model=dict[str, Any])
async def validate_automated_trading_config(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Validate current configuration for automated trading

    Performs comprehensive checks to ensure the system is properly
    configured for safe automated trading.
    """
    try:
        from app.core.config import get_settings

        settings = get_settings()

        validation_results = []
        is_valid = True

        # Check if automated trading is enabled
        if not settings.AUTOMATED_TRADING_ENABLED:
            validation_results.append(
                {
                    "check": "Automated Trading Enabled",
                    "status": "FAIL",
                    "message": "AUTOMATED_TRADING_ENABLED is set to False",
                }
            )
            is_valid = False
        else:
            validation_results.append(
                {
                    "check": "Automated Trading Enabled",
                    "status": "PASS",
                    "message": "Automated trading is enabled",
                }
            )

        # Check if live trading is enabled
        if not settings.LIVE_TRADING_ENABLED:
            validation_results.append(
                {
                    "check": "Live Trading Enabled",
                    "status": "FAIL",
                    "message": "LIVE_TRADING_ENABLED is set to False",
                }
            )
            is_valid = False
        else:
            validation_results.append(
                {
                    "check": "Live Trading Enabled",
                    "status": "PASS",
                    "message": "Live trading is enabled",
                }
            )

        # Check risk limits
        if settings.MAX_RISK_PER_TRADE > 0.1:  # More than 10%
            validation_results.append(
                {
                    "check": "Risk Per Trade",
                    "status": "WARNING",
                    "message": f"Risk per trade is {settings.MAX_RISK_PER_TRADE*100}% (>10%)",
                }
            )
        else:
            validation_results.append(
                {
                    "check": "Risk Per Trade",
                    "status": "PASS",
                    "message": f"Risk per trade is {settings.MAX_RISK_PER_TRADE*100}%",
                }
            )

        # Check daily loss limit
        if settings.MAX_DAILY_LOSS > 0.2:  # More than 20%
            validation_results.append(
                {
                    "check": "Daily Loss Limit",
                    "status": "WARNING",
                    "message": f"Daily loss limit is {settings.MAX_DAILY_LOSS*100}% (>20%)",
                }
            )
        else:
            validation_results.append(
                {
                    "check": "Daily Loss Limit",
                    "status": "PASS",
                    "message": f"Daily loss limit is {settings.MAX_DAILY_LOSS*100}%",
                }
            )

        # Check circuit breaker
        if not settings.CIRCUIT_BREAKER_ENABLED:
            validation_results.append(
                {
                    "check": "Circuit Breaker",
                    "status": "WARNING",
                    "message": "Circuit breaker is disabled",
                }
            )
        else:
            validation_results.append(
                {
                    "check": "Circuit Breaker",
                    "status": "PASS",
                    "message": f"Circuit breaker enabled at {settings.CIRCUIT_BREAKER_LOSS_PERCENT*100}%",
                }
            )

        # Check trading hours enforcement
        if not settings.ENFORCE_TRADING_HOURS:
            validation_results.append(
                {
                    "check": "Trading Hours",
                    "status": "WARNING",
                    "message": "Trading hours enforcement is disabled",
                }
            )
        else:
            validation_results.append(
                {
                    "check": "Trading Hours",
                    "status": "PASS",
                    "message": f"Trading hours: {settings.TRADING_START_TIME} - {settings.TRADING_END_TIME}",
                }
            )

        return {
            "status": "success",
            "is_valid": is_valid,
            "validation_results": validation_results,
            "summary": {
                "total_checks": len(validation_results),
                "passed": len([r for r in validation_results if r["status"] == "PASS"]),
                "warnings": len(
                    [r for r in validation_results if r["status"] == "WARNING"]
                ),
                "failed": len([r for r in validation_results if r["status"] == "FAIL"]),
            },
        }

    except Exception as e:
        logger.error(f"Error validating automated trading config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate config: {str(e)}",
        )
