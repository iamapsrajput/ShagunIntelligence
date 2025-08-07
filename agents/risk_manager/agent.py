from typing import Any

from crewai import Agent
from langchain.llms.base import BaseLLM

from .enhanced_agent import EnhancedRiskManagerAgent


class RiskManagerAgent:
    def __init__(self, llm: BaseLLM, capital: float = 100000):
        self.llm = llm
        self.capital = capital
        self.agent = self._create_agent()
        # Initialize enhanced risk manager with all components
        self.enhanced_risk_manager = EnhancedRiskManagerAgent(
            llm=llm,
            capital=capital,
            max_risk_per_trade=0.02,
            max_portfolio_risk=0.06,
            max_correlation=0.7,
        )

    def _create_agent(self) -> Agent:
        return Agent(
            role="Risk Manager",
            goal="Assess and manage trading risks, calculate position sizes, and ensure compliance with risk parameters",
            backstory="""You are a seasoned risk management expert with extensive experience in
            portfolio risk assessment and position sizing. You understand the importance of
            capital preservation and have deep knowledge of various risk metrics including
            VaR, maximum drawdown, and position sizing strategies. Your primary focus is to
            protect capital while allowing for profitable trading opportunities.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[],
        )

    # Delegate to enhanced risk manager for advanced functionality
    async def evaluate_trade_risk(self, *args, **kwargs):
        """Evaluate trade risk using enhanced risk management."""
        return await self.enhanced_risk_manager.evaluate_trade_risk(*args, **kwargs)

    async def calculate_optimal_position_size(self, *args, **kwargs):
        """Calculate optimal position size."""
        return await self.enhanced_risk_manager.calculate_optimal_position_size(
            *args, **kwargs
        )

    async def set_dynamic_stop_loss(self, *args, **kwargs):
        """Set dynamic stop loss levels."""
        return await self.enhanced_risk_manager.set_dynamic_stop_loss(*args, **kwargs)

    async def monitor_portfolio_risk(self, *args, **kwargs):
        """Monitor portfolio risk metrics."""
        return await self.enhanced_risk_manager.monitor_portfolio_risk(*args, **kwargs)

    async def should_block_trade(self, *args, **kwargs):
        """Check if trade should be blocked."""
        return await self.enhanced_risk_manager.should_block_trade(*args, **kwargs)

    def get_status(self) -> dict[str, Any]:
        """Get risk manager status."""
        return self.enhanced_risk_manager.get_status()
