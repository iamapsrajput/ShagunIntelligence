"""
API routes for data source management and monitoring.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.auth import get_current_user
from backend.data_sources.base import DataSourceStatus

router = APIRouter(prefix="/api/v1/data-sources", tags=["data-sources"])


@router.get("/health", response_model=dict[str, Any])
async def get_data_sources_health(
    request: Request, current_user: dict = Depends(get_current_user)
) -> dict[str, Any]:
    """
    Get health status of all configured data sources.

    Returns:
        Health status including individual source health and summary statistics
    """
    try:
        integration = request.app.state.data_integration
        return integration.get_data_sources_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health-check", response_model=dict[str, Any])
async def force_health_check(
    request: Request,
    source_name: str | None = None,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Force a health check on data sources.

    Args:
        source_name: Optional specific source to check, otherwise checks all

    Returns:
        Health check results
    """
    try:
        integration = request.app.state.data_integration
        results = await integration.force_health_check(source_name)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active/{source_type}", response_model=dict[str, Any])
async def get_active_source(
    request: Request,
    source_type: str = "market_data",
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get the currently active source for a given type.

    Args:
        source_type: Type of source (market_data, sentiment, etc.)

    Returns:
        Active source information
    """
    try:
        integration = request.app.state.data_integration
        active_source = integration.get_active_source(source_type)

        if not active_source:
            return {
                "source_type": source_type,
                "active_source": None,
                "status": "No healthy sources available",
            }

        return {
            "source_type": source_type,
            "active_source": active_source,
            "status": "healthy",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=dict[str, Any])
async def get_data_source_metrics(
    request: Request, current_user: dict = Depends(get_current_user)
) -> dict[str, Any]:
    """
    Get performance metrics for all data sources.

    Returns:
        Detailed metrics including latency, success rates, and request counts
    """
    try:
        integration = request.app.state.data_integration
        manager = integration.manager

        metrics = {}
        for source_type, sources in manager._sources.items():
            for source in sources:
                source_metrics = manager.get_source_metrics(source.config.name)
                if source_metrics:
                    metrics[source.config.name] = {
                        "type": source_type.value,
                        "priority": source.config.priority,
                        "enabled": source.config.enabled,
                        "health": {
                            "status": source_metrics["health"].status.value,
                            "latency_ms": source_metrics["health"].latency_ms,
                            "success_rate": source_metrics["health"].success_rate,
                            "error_count": source_metrics["health"].error_count,
                        },
                        "performance": source_metrics["metrics"],
                    }

        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/failover-test", response_model=dict[str, Any])
async def test_failover(
    request: Request,
    source_type: str = "market_data",
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Test failover mechanism by simulating primary source failure.

    Args:
        source_type: Type of source to test failover for

    Returns:
        Failover test results
    """
    try:
        integration = request.app.state.data_integration
        manager = integration.manager

        # Get sources for the type
        from backend.data_sources.base import DataSourceType

        type_enum = DataSourceType(source_type)
        sources = manager._sources.get(type_enum, [])

        if len(sources) < 2:
            return {
                "status": "insufficient_sources",
                "message": f"Need at least 2 sources for failover testing, found {len(sources)}",
            }

        # Simulate primary source failure
        primary = sources[0]
        original_status = primary.health.status

        # Force unhealthy status
        primary.update_health_status(
            DataSourceStatus.UNHEALTHY, "Simulated failure for testing"
        )

        # Try to get data (should failover)
        try:
            result = await manager.execute_with_failover(
                type_enum, "get_quote", "TEST_SYMBOL"
            )

            # Restore original status
            primary.health.status = original_status

            return {
                "status": "success",
                "message": "Failover test completed successfully",
                "primary_source": primary.config.name,
                "failover_source": result.get("source", "unknown"),
                "test_result": result,
            }

        except Exception as e:
            # Restore original status
            primary.health.status = original_status
            raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
