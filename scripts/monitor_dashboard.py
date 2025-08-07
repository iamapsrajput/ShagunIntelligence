#!/usr/bin/env python3
"""
Shagun Intelligence Trading System - Real-time Monitoring Dashboard

A simple web-based dashboard to monitor the trading system in real-time.
Shows system health, trading activity, and performance metrics.

Usage:
    python scripts/monitor_dashboard.py --mode paper
    python scripts/monitor_dashboard.py --mode live --token your_jwt_token
"""

import argparse
import asyncio
from datetime import datetime
from typing import Optional

import aiohttp
from aiohttp import web
from loguru import logger

# Configure logging
logger.add("logs/monitor_dashboard.log", rotation="1 day", retention="7 days")


class TradingMonitorDashboard:
    """Real-time monitoring dashboard for the trading system."""

    def __init__(
        self,
        trading_api_url: str = "http://127.0.0.1:8000",
        token: str | None = None,
    ):
        self.trading_api_url = trading_api_url
        self.token = token
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        """Setup web routes for the dashboard."""
        self.app.router.add_get("/", self.dashboard_page)
        self.app.router.add_get("/api/status", self.get_system_status)
        self.app.router.add_get("/api/trades", self.get_recent_trades)
        self.app.router.add_get("/api/performance", self.get_performance_metrics)
        self.app.router.add_static("/", path="static/", name="static")

    async def get_system_status(self, request):
        """Get current system status."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get basic health
                async with session.get(f"{self.trading_api_url}/api/v1/health") as resp:
                    health_data = await resp.json()

                # Get detailed health if token available
                detailed_health = {}
                if self.token:
                    headers = {"Authorization": f"Bearer {self.token}"}
                    async with session.get(
                        f"{self.trading_api_url}/api/v1/health/detailed",
                        headers=headers,
                    ) as resp:
                        if resp.status == 200:
                            detailed_health = await resp.json()

                return web.json_response(
                    {
                        "status": "success",
                        "timestamp": datetime.utcnow().isoformat(),
                        "health": health_data,
                        "detailed": detailed_health,
                    }
                )

        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    async def get_recent_trades(self, request):
        """Get recent trading activity."""
        if not self.token:
            return web.json_response({"error": "Authentication required"}, status=401)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.token}"}

                # Get recent trades
                async with session.get(
                    f"{self.trading_api_url}/api/v1/trading/history?limit=20",
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        trades_data = await resp.json()
                        return web.json_response(
                            {"status": "success", "trades": trades_data}
                        )
                    else:
                        return web.json_response(
                            {"status": "error", "error": "Failed to fetch trades"},
                            status=resp.status,
                        )

        except Exception as e:
            logger.error(f"Error getting recent trades: {str(e)}")
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    async def get_performance_metrics(self, request):
        """Get performance metrics."""
        if not self.token:
            return web.json_response({"error": "Authentication required"}, status=401)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.token}"}

                # Get agent performance
                async with session.get(
                    f"{self.trading_api_url}/api/v1/agents/performance/metrics",
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        performance_data = await resp.json()
                        return web.json_response(
                            {"status": "success", "performance": performance_data}
                        )
                    else:
                        return web.json_response(
                            {
                                "status": "error",
                                "error": "Failed to fetch performance metrics",
                            },
                            status=resp.status,
                        )

        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    async def dashboard_page(self, request):
        """Serve the main dashboard page."""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shagun Intelligence - Trading Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .status-item { padding: 15px; border-radius: 5px; text-align: center; }
        .status-healthy { background-color: #d4edda; color: #155724; }
        .status-warning { background-color: #fff3cd; color: #856404; }
        .status-error { background-color: #f8d7da; color: #721c24; }
        .trades-table { width: 100%; border-collapse: collapse; }
        .trades-table th, .trades-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        .trades-table th { background-color: #f8f9fa; }
        .refresh-btn { background-color: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        .refresh-btn:hover { background-color: #0056b3; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Shagun Intelligence Trading Monitor</h1>
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh</button>
            <div class="timestamp" id="lastUpdate">Last updated: Loading...</div>
        </div>

        <div class="card">
            <h2>📊 System Status</h2>
            <div class="status-grid" id="systemStatus">
                <div class="status-item">Loading...</div>
            </div>
        </div>

        <div class="card">
            <h2>📈 Recent Trades</h2>
            <div id="recentTrades">Loading...</div>
        </div>

        <div class="card">
            <h2>⚡ Performance Metrics</h2>
            <div id="performanceMetrics">Loading...</div>
        </div>
    </div>

    <script>
        async function fetchData(endpoint) {
            try {
                const response = await fetch(endpoint);
                return await response.json();
            } catch (error) {
                console.error('Error fetching data:', error);
                return { status: 'error', error: error.message };
            }
        }

        async function updateSystemStatus() {
            const data = await fetchData('/api/status');
            const container = document.getElementById('systemStatus');

            if (data.status === 'success') {
                const health = data.health;
                const detailed = data.detailed;

                let html = `
                    <div class="status-item status-${health.status === 'healthy' ? 'healthy' : 'error'}">
                        <h3>System Health</h3>
                        <p>${health.status.toUpperCase()}</p>
                    </div>
                `;

                if (detailed.services) {
                    Object.entries(detailed.services).forEach(([service, status]) => {
                        const statusClass = typeof status === 'object' ?
                            (status.status === 'healthy' ? 'healthy' : 'warning') :
                            (status === 'connected' || status === true ? 'healthy' : 'warning');

                        html += `
                            <div class="status-item status-${statusClass}">
                                <h3>${service.replace('_', ' ').toUpperCase()}</h3>
                                <p>${typeof status === 'object' ? status.status : status}</p>
                            </div>
                        `;
                    });
                }

                container.innerHTML = html;
            } else {
                container.innerHTML = '<div class="status-item status-error">Error loading status</div>';
            }
        }

        async function updateRecentTrades() {
            const data = await fetchData('/api/trades');
            const container = document.getElementById('recentTrades');

            if (data.status === 'success' && data.trades) {
                let html = '<table class="trades-table"><tr><th>Time</th><th>Symbol</th><th>Action</th><th>Quantity</th><th>Price</th><th>Status</th></tr>';

                data.trades.forEach(trade => {
                    html += `
                        <tr>
                            <td>${new Date(trade.timestamp).toLocaleString()}</td>
                            <td>${trade.symbol}</td>
                            <td>${trade.action}</td>
                            <td>${trade.quantity}</td>
                            <td>${trade.price}</td>
                            <td>${trade.status}</td>
                        </tr>
                    `;
                });

                html += '</table>';
                container.innerHTML = html;
            } else {
                container.innerHTML = '<p>No recent trades or authentication required</p>';
            }
        }

        async function updatePerformanceMetrics() {
            const data = await fetchData('/api/performance');
            const container = document.getElementById('performanceMetrics');

            if (data.status === 'success' && data.performance) {
                let html = '<div class="status-grid">';

                Object.entries(data.performance).forEach(([metric, value]) => {
                    html += `
                        <div class="status-item status-healthy">
                            <h3>${metric.replace('_', ' ').toUpperCase()}</h3>
                            <p>${typeof value === 'object' ? JSON.stringify(value) : value}</p>
                        </div>
                    `;
                });

                html += '</div>';
                container.innerHTML = html;
            } else {
                container.innerHTML = '<p>No performance data or authentication required</p>';
            }
        }

        async function refreshData() {
            document.getElementById('lastUpdate').textContent = 'Updating...';

            await Promise.all([
                updateSystemStatus(),
                updateRecentTrades(),
                updatePerformanceMetrics()
            ]);

            document.getElementById('lastUpdate').textContent = `Last updated: ${new Date().toLocaleString()}`;
        }

        // Initial load and auto-refresh
        refreshData();
        setInterval(refreshData, 30000); // Refresh every 30 seconds
    </script>
</body>
</html>
        """
        return web.Response(text=html_content, content_type="text/html")

    async def start_server(self, host: str = "127.0.0.1", port: int = 8080):
        """Start the monitoring dashboard server."""
        logger.info(f"Starting monitoring dashboard at http://{host}:{port}")

        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, host, port)
        await site.start()

        logger.success(f"Dashboard running at http://{host}:{port}")
        return runner


async def main():
    """Main function to start the monitoring dashboard."""
    parser = argparse.ArgumentParser(description="Trading System Monitoring Dashboard")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode to monitor",
    )
    parser.add_argument("--token", help="JWT token for authenticated endpoints")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    parser.add_argument(
        "--api-url", default="http://127.0.0.1:8000", help="Trading system API URL"
    )

    args = parser.parse_args()

    # Create and start dashboard
    dashboard = TradingMonitorDashboard(trading_api_url=args.api_url, token=args.token)

    runner = await dashboard.start_server(args.host, args.port)

    try:
        # Keep the server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard...")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
