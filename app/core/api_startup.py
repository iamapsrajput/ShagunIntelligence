"""
API management system startup and initialization.
"""

import asyncio
import os

from loguru import logger

from .api_config import APIProvider, get_api_config
from .api_health_monitor import AlertSeverity, HealthAlert, get_health_monitor
from .api_key_manager import get_api_key_manager
from .api_rate_limiter import get_rate_limiter


async def handle_health_alert(alert: HealthAlert):
    """Handle API health alerts."""
    # Log the alert
    if alert.severity == AlertSeverity.CRITICAL:
        logger.critical(f"API Health Alert: {alert.provider} - {alert.message}")
    elif alert.severity == AlertSeverity.ERROR:
        logger.error(f"API Health Alert: {alert.provider} - {alert.message}")
    elif alert.severity == AlertSeverity.WARNING:
        logger.warning(f"API Health Alert: {alert.provider} - {alert.message}")
    else:
        logger.info(f"API Health Alert: {alert.provider} - {alert.message}")

    # In production, send notifications (email, Slack, etc.)
    # Example: await send_slack_notification(alert)


class APIManagementSystem:
    """
    Manages the initialization and lifecycle of API integrations.
    """

    def __init__(self):
        self.initialized = False
        self.health_monitor = None
        self.rate_limiter = None
        self.key_manager = None
        self.api_config = None

        logger.info("APIManagementSystem created")

    async def initialize(self):
        """Initialize all API management components."""
        if self.initialized:
            logger.warning("API Management System already initialized")
            return

        try:
            logger.info("Initializing API Management System...")

            # 1. Load API configuration
            self.api_config = get_api_config()
            env = self.api_config.environment.value
            logger.info(f"Loaded API configuration for environment: {env}")

            # 2. Initialize API key manager
            storage_path = os.getenv("API_KEY_STORAGE", "./config/api_keys.enc")
            master_password = os.getenv("API_KEY_MASTER_PASSWORD")

            self.key_manager = get_api_key_manager(storage_path, master_password)
            logger.info("API Key Manager initialized")

            # 3. Initialize rate limiter
            self.rate_limiter = await get_rate_limiter()
            logger.info("API Rate Limiter initialized")

            # 4. Initialize health monitor
            self.health_monitor = get_health_monitor(handle_health_alert)
            await self.health_monitor.start_monitoring()
            logger.info("API Health Monitor started")

            # 5. Start key rotation monitor
            await self.key_manager.start_rotation_monitor()
            logger.info("API Key Rotation Monitor started")

            # 6. Validate configuration
            await self._validate_configuration()

            # 7. Perform initial health checks
            await self._initial_health_checks()

            self.initialized = True
            logger.info("API Management System initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize API Management System: {e}")
            raise

    async def _validate_configuration(self):
        """Validate API configuration and keys."""
        logger.info("Validating API configuration...")

        enabled_apis = self.api_config.get_enabled_apis()
        logger.info(f"Found {len(enabled_apis)} enabled APIs")

        # Check for required keys
        missing_keys = []

        for provider, _config in enabled_apis.items():
            # Check if we have necessary keys
            has_key = False

            for key_type in ["api_key", "api_secret", "access_token", "bearer_token"]:
                if self.key_manager.get_key(provider, key_type):
                    has_key = True
                    break

            if not has_key:
                missing_keys.append(provider.value)
                logger.warning(f"No API keys found for {provider.value}")

        if missing_keys:
            logger.warning(
                f"Missing API keys for: {', '.join(missing_keys)}. "
                f"Some features may be unavailable."
            )

        # Validate rate limits
        total_cost = sum(self.api_config.estimate_monthly_cost().values())
        logger.info(f"Estimated monthly API cost: ${total_cost:.2f}")

        # Check failover configuration
        for feature in ["realtime", "historical", "news", "sentiment"]:
            apis = self.api_config.get_apis_for_feature(feature)
            if not apis:
                logger.warning(f"No APIs configured for feature: {feature}")
            else:
                logger.info(f"Feature '{feature}' has {len(apis)} API providers")

    async def _initial_health_checks(self):
        """Perform initial health checks on critical APIs."""
        logger.info("Performing initial health checks...")

        # Check critical APIs
        critical_providers = [
            APIProvider.KITE_CONNECT,  # Primary for Indian markets
            APIProvider.ALPHA_VANTAGE,  # Backup data source
            APIProvider.FINNHUB,  # Real-time and news
        ]

        tasks = []
        for provider in critical_providers:
            config = self.api_config.get_api_config(provider)
            if config and config.enabled:
                task = self.health_monitor.force_health_check(provider)
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Health check failed for {critical_providers[i].value}: {result}"
                    )
                else:
                    logger.info(
                        f"Health check for {result.provider}: "
                        f"{result.status.value} ({result.response_time_ms:.0f}ms)"
                    )

    async def shutdown(self):
        """Shutdown API management system."""
        logger.info("Shutting down API Management System...")

        try:
            # Stop health monitoring
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()

            # Stop key rotation monitor
            if self.key_manager:
                await self.key_manager.stop_rotation_monitor()

            # Cleanup rate limiter
            if self.rate_limiter:
                await self.rate_limiter.cleanup()

            self.initialized = False
            logger.info("API Management System shutdown complete")

        except Exception as e:
            logger.error(f"Error during API Management System shutdown: {e}")

    def get_status(self) -> dict:
        """Get current system status."""
        if not self.initialized:
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "environment": self.api_config.environment.value,
            "enabled_apis": len(self.api_config.get_enabled_apis()),
            "health_monitor": "running" if self.health_monitor else "stopped",
            "rate_limiter": "active" if self.rate_limiter else "inactive",
            "key_manager": "active" if self.key_manager else "inactive",
        }


# Singleton instance
_api_system_instance: APIManagementSystem | None = None


async def initialize_api_system():
    """Initialize the API management system."""
    global _api_system_instance

    if _api_system_instance is None:
        _api_system_instance = APIManagementSystem()

    await _api_system_instance.initialize()
    return _api_system_instance


async def get_api_system() -> APIManagementSystem:
    """Get the API management system instance."""
    if _api_system_instance is None:
        await initialize_api_system()

    return _api_system_instance


async def shutdown_api_system():
    """Shutdown the API management system."""
    if _api_system_instance:
        await _api_system_instance.shutdown()


# FastAPI startup/shutdown events
async def on_startup():
    """Called on application startup."""
    # Check if we should initialize API system
    if os.getenv("DISABLE_API_MANAGEMENT", "false").lower() != "true":
        await initialize_api_system()
    else:
        logger.info("API Management System disabled by environment variable")


async def on_shutdown():
    """Called on application shutdown."""
    await shutdown_api_system()
