"""
System Health and Resilience Monitoring API
Provides comprehensive health checks and resilience status
"""

import time
from datetime import datetime
from typing import Any

import psutil
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.resilience import CircuitState, resilience_manager
from app.services.market_schedule import market_schedule

router = APIRouter(prefix="/system", tags=["system-health"])
settings = get_settings()


@router.get("/health/comprehensive")
async def get_comprehensive_health() -> dict[str, Any]:
    """Get comprehensive system health status including resilience metrics"""

    start_time = time.time()

    try:
        # Get resilience system health
        resilience_health = resilience_manager.get_system_health()

        # Get system resource metrics
        system_metrics = await _get_system_metrics()

        # Get service-specific health checks
        service_health = await _get_service_health()

        # Get trading system status
        trading_health = await _get_trading_health()

        # Calculate overall health score
        health_score = _calculate_health_score(
            resilience_health, system_metrics, service_health, trading_health
        )

        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return {
            "status": (
                "healthy"
                if health_score >= 80
                else "degraded" if health_score >= 60 else "unhealthy"
            ),
            "health_score": health_score,
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": round(response_time, 2),
            "resilience": resilience_health,
            "system_metrics": system_metrics,
            "services": service_health,
            "trading": trading_health,
            "recommendations": _get_health_recommendations(
                resilience_health, system_metrics, service_health
            ),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            },
        )


@router.get("/health/resilience")
async def get_resilience_status() -> dict[str, Any]:
    """Get detailed resilience system status"""
    return resilience_manager.get_system_health()


@router.get("/health/circuit-breakers")
async def get_circuit_breaker_status() -> dict[str, Any]:
    """Get status of all circuit breakers"""

    circuit_status = {}

    for name, breaker in resilience_manager.circuit_breakers.items():
        circuit_status[name] = {
            "state": breaker.state.value,
            "failure_threshold": breaker.config.failure_threshold,
            "success_threshold": breaker.config.success_threshold,
            "timeout": breaker.config.timeout,
            "metrics": {
                "success_count": breaker.metrics.success_count,
                "failure_count": breaker.metrics.failure_count,
                "total_requests": breaker.metrics.total_requests,
                "success_rate": (
                    breaker.metrics.success_count
                    / max(breaker.metrics.total_requests, 1)
                    * 100
                ),
                "consecutive_failures": breaker.metrics.consecutive_failures,
                "avg_response_time": breaker.metrics.avg_response_time,
                "last_success": (
                    breaker.metrics.last_success.isoformat()
                    if breaker.metrics.last_success
                    else None
                ),
                "last_failure": (
                    breaker.metrics.last_failure.isoformat()
                    if breaker.metrics.last_failure
                    else None
                ),
            },
            "state_changed_at": breaker.state_changed_at.isoformat(),
        }

    return {"circuit_breakers": circuit_status, "timestamp": datetime.now().isoformat()}


@router.post("/health/circuit-breakers/{service_name}/reset")
async def reset_circuit_breaker(service_name: str) -> dict[str, Any]:
    """Manually reset a circuit breaker"""

    breaker = resilience_manager.get_circuit_breaker(service_name)
    if not breaker:
        raise HTTPException(
            status_code=404, detail=f"Circuit breaker '{service_name}' not found"
        )

    # Reset the circuit breaker
    breaker.state = CircuitState.CLOSED
    breaker.state_changed_at = datetime.now()
    breaker.metrics.consecutive_failures = 0

    return {
        "message": f"Circuit breaker '{service_name}' has been reset",
        "new_state": breaker.state.value,
        "timestamp": datetime.now().isoformat(),
    }


async def _get_system_metrics() -> dict[str, Any]:
    """Get system resource metrics"""

    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)

        # Disk metrics
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent
        disk_free_gb = disk.free / (1024**3)

        # Network metrics (if available)
        network_io = psutil.net_io_counters()

        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "status": (
                    "healthy"
                    if cpu_percent < 80
                    else "warning" if cpu_percent < 95 else "critical"
                ),
            },
            "memory": {
                "usage_percent": memory_percent,
                "available_gb": round(memory_available_gb, 2),
                "total_gb": round(memory.total / (1024**3), 2),
                "status": (
                    "healthy"
                    if memory_percent < 80
                    else "warning" if memory_percent < 95 else "critical"
                ),
            },
            "disk": {
                "usage_percent": disk_percent,
                "free_gb": round(disk_free_gb, 2),
                "total_gb": round(disk.total / (1024**3), 2),
                "status": (
                    "healthy"
                    if disk_percent < 80
                    else "warning" if disk_percent < 95 else "critical"
                ),
            },
            "network": {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv,
            },
        }

    except Exception as e:
        return {"error": f"Failed to get system metrics: {str(e)}", "status": "error"}


async def _get_service_health() -> dict[str, Any]:
    """Get health status of individual services"""

    services = {}

    # Check market schedule service
    try:
        market_status = market_schedule.get_market_status()
        services["market_schedule"] = {
            "status": "healthy",
            "market_open": market_status["is_open"],
            "market_status": market_status["status"],
        }
    except Exception as e:
        services["market_schedule"] = {"status": "unhealthy", "error": str(e)}

    # Add more service checks here as needed

    return services


async def _get_trading_health() -> dict[str, Any]:
    """Get trading system health status"""

    try:
        # This would typically check the trading service
        # For now, return basic status
        return {
            "status": "healthy",
            "trading_enabled": True,
            "last_check": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "trading_enabled": False}


def _calculate_health_score(
    resilience_health: dict,
    system_metrics: dict,
    service_health: dict,
    trading_health: dict,
) -> int:
    """Calculate overall system health score (0-100)"""

    score = 100

    # Resilience system impact (40% weight)
    if resilience_health["overall_status"] == "critical":
        score -= 40
    elif resilience_health["overall_status"] == "unhealthy":
        score -= 30
    elif resilience_health["overall_status"] == "degraded":
        score -= 15

    # System metrics impact (30% weight)
    if "error" not in system_metrics:
        if system_metrics["cpu"]["status"] == "critical":
            score -= 15
        elif system_metrics["cpu"]["status"] == "warning":
            score -= 8

        if system_metrics["memory"]["status"] == "critical":
            score -= 15
        elif system_metrics["memory"]["status"] == "warning":
            score -= 7
    else:
        score -= 30

    # Service health impact (20% weight)
    unhealthy_services = sum(
        1 for service in service_health.values() if service.get("status") != "healthy"
    )
    total_services = len(service_health)
    if total_services > 0:
        service_penalty = (unhealthy_services / total_services) * 20
        score -= service_penalty

    # Trading system impact (10% weight)
    if trading_health.get("status") != "healthy":
        score -= 10

    return max(0, min(100, int(score)))


def _get_health_recommendations(
    resilience_health: dict, system_metrics: dict, service_health: dict
) -> list[str]:
    """Get health improvement recommendations"""

    recommendations = []

    # Resilience recommendations
    if resilience_health["overall_status"] in ["unhealthy", "critical"]:
        recommendations.append(
            "Check circuit breaker status and consider manual reset if appropriate"
        )
        recommendations.append("Review service logs for recurring errors")

    # System resource recommendations
    if "error" not in system_metrics:
        if system_metrics["cpu"]["usage_percent"] > 80:
            recommendations.append(
                "High CPU usage detected - consider scaling resources"
            )

        if system_metrics["memory"]["usage_percent"] > 80:
            recommendations.append(
                "High memory usage detected - check for memory leaks"
            )

        if system_metrics["disk"]["usage_percent"] > 80:
            recommendations.append("Low disk space - consider cleanup or expansion")

    # Service recommendations
    unhealthy_services = [
        name
        for name, status in service_health.items()
        if status.get("status") != "healthy"
    ]
    if unhealthy_services:
        recommendations.append(
            f"Unhealthy services detected: {', '.join(unhealthy_services)}"
        )

    if not recommendations:
        recommendations.append("System is operating normally")

    return recommendations
