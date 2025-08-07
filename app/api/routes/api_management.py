"""
API management endpoints for monitoring and configuration.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.api_config import APIProvider, get_api_config
from app.core.api_health_monitor import get_health_monitor
from app.core.api_key_manager import get_api_key_manager
from app.core.api_rate_limiter import get_rate_limiter
from app.core.auth import get_current_user  # Assuming auth is implemented

router = APIRouter(prefix="/api/v1/management", tags=["API Management"])


class APIKeyUpdate(BaseModel):
    """Model for API key update."""

    provider: APIProvider
    key_type: str
    value: str
    expires_at: datetime | None = None


class APIConfigUpdate(BaseModel):
    """Model for API configuration update."""

    provider: APIProvider
    enabled: bool | None = None
    priority: int | None = None
    rate_limit_per_minute: int | None = None
    monthly_budget: float | None = None


@router.get("/config")
async def get_api_configuration(
    provider: APIProvider | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Get API configuration."""
    config = get_api_config()

    if provider:
        provider_config = config.get_api_config(provider)
        if not provider_config:
            raise HTTPException(
                status_code=404, detail=f"Provider {provider.value} not found"
            )
        return provider_config.dict(
            exclude={"api_key", "api_secret", "access_token", "bearer_token"}
        )

    # Return all configurations (sanitized)
    return config.to_secure_dict()


@router.get("/config/enabled")
async def get_enabled_apis(current_user: dict = Depends(get_current_user)):
    """Get all enabled API configurations."""
    config = get_api_config()
    enabled_apis = config.get_enabled_apis()

    return {
        provider.value: {
            "priority": api_config.priority,
            "tier": api_config.tier.value,
            "base_url": api_config.base_url,
            "features": {
                "websocket": api_config.supports_websocket,
                "historical": api_config.supports_historical,
                "realtime": api_config.supports_realtime,
                "news": api_config.supports_news,
                "sentiment": api_config.supports_sentiment,
            },
        }
        for provider, api_config in enabled_apis.items()
    }


@router.get("/config/costs")
async def get_api_costs(current_user: dict = Depends(get_current_user)):
    """Get estimated API costs."""
    config = get_api_config()
    return config.estimate_monthly_cost()


@router.get("/keys/status")
async def get_api_keys_status(current_user: dict = Depends(get_current_user)):
    """Get status of all API keys."""
    key_manager = get_api_key_manager()
    return key_manager.get_all_keys_status()


@router.get("/keys/rotation")
async def get_key_rotation_schedule(current_user: dict = Depends(get_current_user)):
    """Get API key rotation schedule."""
    key_manager = get_api_key_manager()
    return key_manager.get_rotation_schedule()


@router.post("/keys/update")
async def update_api_key(
    key_update: APIKeyUpdate, current_user: dict = Depends(get_current_user)
):
    """Update an API key."""
    key_manager = get_api_key_manager()

    success = await key_manager.set_key(
        provider=key_update.provider,
        key_type=key_update.key_type,
        value=key_update.value,
        expires_at=key_update.expires_at,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update API key")

    return {
        "status": "success",
        "message": f"Updated {key_update.key_type} for {key_update.provider.value}",
    }


@router.post("/keys/rotate/{provider}/{key_type}")
async def rotate_api_key(
    provider: APIProvider,
    key_type: str,
    new_value: str,
    current_user: dict = Depends(get_current_user),
):
    """Rotate an API key."""
    key_manager = get_api_key_manager()

    success = await key_manager.rotate_key(
        provider=provider, key_type=key_type, new_value=new_value
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to rotate API key")

    return {"status": "success", "message": f"Rotated {key_type} for {provider.value}"}


@router.get("/rate-limits")
async def get_rate_limit_status(
    provider: APIProvider | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Get current rate limit status."""
    rate_limiter = await get_rate_limiter()
    status = rate_limiter.get_rate_limit_status()

    if provider:
        if provider.value not in status:
            raise HTTPException(
                status_code=404, detail=f"Provider {provider.value} not found"
            )
        return {provider.value: status[provider.value]}

    return status


@router.get("/usage")
async def get_api_usage(
    provider: APIProvider | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Get API usage summary."""
    rate_limiter = await get_rate_limiter()
    return rate_limiter.get_usage_summary(provider)


@router.get("/usage/costs")
async def get_usage_costs(current_user: dict = Depends(get_current_user)):
    """Get API usage costs report."""
    rate_limiter = await get_rate_limiter()
    return rate_limiter.get_cost_report()


@router.get("/health")
async def get_api_health(
    provider: APIProvider | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Get API health status."""
    health_monitor = get_health_monitor()
    return health_monitor.get_health_status(provider)


@router.get("/health/report")
async def get_health_report(current_user: dict = Depends(get_current_user)):
    """Get comprehensive health report."""
    health_monitor = get_health_monitor()
    return health_monitor.get_health_report()


@router.post("/health/check/{provider}")
async def force_health_check(
    provider: APIProvider, current_user: dict = Depends(get_current_user)
):
    """Force an immediate health check for a provider."""
    health_monitor = get_health_monitor()

    try:
        result = await health_monitor.force_health_check(provider)

        return {
            "provider": result.provider,
            "status": result.status.value,
            "response_time_ms": result.response_time_ms,
            "status_code": result.status_code,
            "error": result.error,
            "timestamp": result.timestamp.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/failover/priority")
async def get_failover_priority(
    feature: str = Query(
        ..., description="Feature: websocket, historical, realtime, news, sentiment"
    ),
    current_user: dict = Depends(get_current_user),
):
    """Get API failover priority for a specific feature."""
    config = get_api_config()

    apis = config.get_apis_for_feature(feature)
    if not apis:
        raise HTTPException(
            status_code=404, detail=f"No APIs found for feature: {feature}"
        )

    return [
        {
            "provider": provider.value,
            "priority": api_config.priority,
            "tier": api_config.tier.value,
            "base_url": api_config.base_url,
            "quality_threshold": api_config.min_quality_score,
            "rate_limit": api_config.rate_limit_per_minute,
        }
        for provider, api_config in apis
    ]


@router.get("/dashboard")
async def get_management_dashboard(current_user: dict = Depends(get_current_user)):
    """Get comprehensive management dashboard data."""
    config = get_api_config()
    key_manager = get_api_key_manager()
    rate_limiter = await get_rate_limiter()
    health_monitor = get_health_monitor()

    # Compile dashboard data
    dashboard = {
        "timestamp": datetime.now().isoformat(),
        "environment": config.environment.value,
        # API Status
        "api_status": {
            "total": len(list(APIProvider)),
            "enabled": len(config.get_enabled_apis()),
            "configured": len(
                [
                    p
                    for p in APIProvider
                    if any(
                        key_manager.get_key(p, kt)
                        for kt in [
                            "api_key",
                            "api_secret",
                            "access_token",
                            "bearer_token",
                        ]
                    )
                ]
            ),
        },
        # Health Summary
        "health_summary": health_monitor.get_health_report()["summary"],
        # Cost Summary
        "cost_summary": rate_limiter.get_cost_report(),
        # Rate Limit Summary
        "rate_limits": {
            provider: {
                "status": "throttled" if info["throttled"] else "ok",
                "utilization": {
                    "minute": (
                        (info["current"]["minute"] / info["limits"]["per_minute"] * 100)
                        if info["limits"]["per_minute"]
                        else 0
                    ),
                    "hour": (
                        (info["current"]["hour"] / info["limits"]["per_hour"] * 100)
                        if info["limits"]["per_hour"]
                        else 0
                    ),
                    "day": (
                        (info["current"]["day"] / info["limits"]["per_day"] * 100)
                        if info["limits"]["per_day"]
                        else 0
                    ),
                },
            }
            for provider, info in rate_limiter.get_rate_limit_status().items()
        },
        # Key Rotation
        "key_rotation": {
            "overdue": len(
                [
                    s
                    for s in key_manager.get_rotation_schedule()
                    if s["status"] == "overdue"
                ]
            ),
            "due_soon": len(
                [
                    s
                    for s in key_manager.get_rotation_schedule()
                    if s["status"] == "due_soon"
                ]
            ),
        },
    }

    return dashboard


@router.post("/export/usage")
async def export_usage_data(current_user: dict = Depends(get_current_user)):
    """Export API usage data."""
    rate_limiter = await get_rate_limiter()

    filename = f"api_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    rate_limiter.export_usage_data(filename)

    return {"status": "success", "filename": filename}


@router.post("/export/health")
async def export_health_data(current_user: dict = Depends(get_current_user)):
    """Export API health data."""
    health_monitor = get_health_monitor()

    filename = f"api_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    health_monitor.export_health_data(filename)

    return {"status": "success", "filename": filename}


@router.post("/export/env-template")
async def export_env_template(current_user: dict = Depends(get_current_user)):
    """Export environment variable template."""
    key_manager = get_api_key_manager()

    filename = ".env.template"
    key_manager.export_template(filename)

    return {
        "status": "success",
        "filename": filename,
        "message": "Template created. Copy to .env and add your API keys.",
    }
