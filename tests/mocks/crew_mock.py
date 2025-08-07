import random
import uuid
from datetime import datetime
from typing import Any


class MockAgent:
    """Mock agent for testing"""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.execution_count = 0
        self.last_result = None

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a mock task"""
        self.execution_count += 1

        result = {
            "agent": self.name,
            "task": task.get("description", "Unknown task"),
            "result": f"Mock result from {self.name}",
            "confidence": random.uniform(0.6, 0.95),
            "timestamp": datetime.now().isoformat(),
        }

        self.last_result = result
        return result


class MockCrewManager:
    """Mock CrewAI manager for testing"""

    def __init__(self):
        self.agents = {
            "market_analyst": MockAgent("Market Analyst", "Analyze market trends"),
            "technical_indicator": MockAgent(
                "Technical Indicator", "Calculate indicators"
            ),
            "sentiment_analyst": MockAgent(
                "Sentiment Analyst", "Analyze market sentiment"
            ),
            "risk_manager": MockAgent("Risk Manager", "Assess and manage risks"),
            "trade_executor": MockAgent("Trade Executor", "Execute trades"),
            "data_processor": MockAgent("Data Processor", "Process market data"),
            "coordinator": MockAgent("Coordinator", "Coordinate agents"),
        }
        self.shared_memory = {}
        self.analysis_history = []

    async def initialize(self):
        """Initialize mock crew manager"""
        return True

    async def analyze_trade_opportunity(self, symbol: str) -> dict[str, Any]:
        """Mock trade opportunity analysis"""
        # Simulate agent analysis
        market_signal = random.choice(["buy", "sell", "hold"])
        confidence = random.uniform(0.5, 0.95)

        analysis = {
            "symbol": symbol,
            "recommendation": market_signal,
            "confidence": confidence,
            "timestamp": datetime.now(),
            "agents_analysis": {
                "market": {
                    "signal": market_signal,
                    "confidence": random.uniform(0.6, 0.9),
                    "reasoning": "Mock market analysis",
                },
                "technical": {
                    "signal": market_signal,
                    "confidence": random.uniform(0.7, 0.95),
                    "indicators": {
                        "rsi": random.uniform(30, 70),
                        "macd": random.choice(["bullish", "bearish"]),
                        "moving_averages": random.choice(["bullish", "bearish"]),
                    },
                },
                "sentiment": {
                    "signal": market_signal,
                    "confidence": random.uniform(0.5, 0.85),
                    "sentiment_score": random.uniform(-1, 1),
                },
                "risk": {
                    "approved": confidence > 0.7,
                    "risk_score": random.uniform(0.1, 0.5),
                    "position_size": random.randint(10, 100),
                },
            },
            "entry_price": 2500 * (1 + random.uniform(-0.02, 0.02)),
            "stop_loss": 2450,
            "take_profit": 2600,
            "rationale": f"Mock analysis for {symbol} with {market_signal} signal",
        }

        self.analysis_history.append(analysis)
        return analysis

    async def execute_trade(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Mock trade execution"""
        order_id = str(uuid.uuid4())

        result = {
            "success": random.random() > 0.1,  # 90% success rate
            "order_id": order_id,
            "symbol": signal.get("symbol"),
            "action": signal.get("action"),
            "quantity": signal.get("quantity"),
            "price": signal.get("entry_price"),
            "timestamp": datetime.now(),
            "message": "Mock order placed successfully",
        }

        if not result["success"]:
            result["error"] = "Mock error: Insufficient funds"

        return result

    async def analyze_exit_conditions(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        quantity: int,
        pnl: float,
    ) -> dict[str, Any]:
        """Mock exit condition analysis"""
        pnl_percent = ((current_price - entry_price) / entry_price) * 100

        # Simple exit logic
        should_exit = (
            pnl_percent > 2.0  # Take profit at 2%
            or pnl_percent < -1.0  # Stop loss at -1%
            or random.random() < 0.05  # 5% random exit
        )

        reason = "No exit signal"
        if pnl_percent > 2.0:
            reason = "Take profit target reached"
        elif pnl_percent < -1.0:
            reason = "Stop loss triggered"
        elif should_exit:
            reason = "Technical exit signal"

        return {
            "should_exit": should_exit,
            "reason": reason,
            "confidence": random.uniform(0.7, 0.95),
            "pnl": pnl,
            "pnl_percent": pnl_percent,
        }

    async def get_agent_status(self, agent_name: str) -> dict[str, Any]:
        """Get mock agent status"""
        if agent_name not in self.agents:
            return {"error": "Agent not found"}

        agent = self.agents[agent_name]
        return {
            "name": agent.name,
            "role": agent.role,
            "status": "active",
            "execution_count": agent.execution_count,
            "last_execution": (
                agent.last_result.get("timestamp") if agent.last_result else None
            ),
            "memory_usage": random.randint(50, 200),  # MB
            "cpu_usage": random.uniform(5, 30),  # Percentage
        }

    async def get_all_agents_status(self) -> list[dict[str, Any]]:
        """Get all agents status"""
        statuses = []
        for agent_name in self.agents:
            status = await self.get_agent_status(agent_name)
            statuses.append(status)
        return statuses

    async def update_agent_memory(self, agent_name: str, key: str, value: Any):
        """Update agent memory"""
        if agent_name not in self.shared_memory:
            self.shared_memory[agent_name] = {}
        self.shared_memory[agent_name][key] = value

    async def get_agent_memory(self, agent_name: str, key: str) -> Any:
        """Get agent memory"""
        return self.shared_memory.get(agent_name, {}).get(key)

    async def clear_agent_memory(self, agent_name: str):
        """Clear agent memory"""
        if agent_name in self.shared_memory:
            self.shared_memory[agent_name] = {}

    async def get_daily_agent_performance(self) -> dict[str, Any]:
        """Get mock daily performance metrics"""
        performance = {}

        for agent_name, agent in self.agents.items():
            performance[agent_name] = {
                "tasks_completed": agent.execution_count,
                "success_rate": random.uniform(0.85, 0.98),
                "average_execution_time": random.uniform(0.1, 2.0),  # seconds
                "accuracy": random.uniform(0.7, 0.95),
            }

        return performance

    async def run_backtest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        """Run mock backtest"""
        total_trades = random.randint(20, 100)
        winning_trades = int(total_trades * random.uniform(0.45, 0.65))

        return {
            "symbol": symbol,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "strategy": strategy or "default",
            "metrics": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate": winning_trades / total_trades,
                "total_return": random.uniform(-10, 30),  # percentage
                "sharpe_ratio": random.uniform(0.5, 2.5),
                "max_drawdown": random.uniform(5, 20),  # percentage
                "profit_factor": random.uniform(0.8, 2.0),
            },
            "trades": [],  # Would contain detailed trade history
        }

    def get_shared_memory_snapshot(self) -> dict[str, Any]:
        """Get snapshot of shared memory"""
        return self.shared_memory.copy()

    def reset(self):
        """Reset mock crew manager"""
        for agent in self.agents.values():
            agent.execution_count = 0
            agent.last_result = None
        self.shared_memory.clear()
        self.analysis_history.clear()
