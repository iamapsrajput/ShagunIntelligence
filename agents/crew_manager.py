from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from typing import Dict, Any
import asyncio
from loguru import logger

from .market_analyst.agent import MarketAnalystAgent
from .risk_manager.agent import RiskManagerAgent
from .trader.agent import TraderAgent
from .data_processor.agent import DataProcessorAgent
from app.core.config import get_settings

class CrewManager:
    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.OPENAI_MODEL,
            api_key=self.settings.OPENAI_API_KEY,
            temperature=0.1
        )
        
        # Initialize agents
        self.market_analyst = MarketAnalystAgent(self.llm)
        self.risk_manager = RiskManagerAgent(self.llm)
        self.trader = TraderAgent(self.llm)
        self.data_processor = DataProcessorAgent(self.llm)
        
        logger.info("CrewManager initialized with all agents")
    
    async def analyze_trade_opportunity(self, symbol: str) -> Dict[str, Any]:
        """Analyze a trading opportunity using the crew of agents"""
        try:
            # Create tasks for the crew
            data_task = Task(
                description=f"Collect and process market data for {symbol}",
                agent=self.data_processor.agent,
                expected_output="Processed market data with technical indicators"
            )
            
            analysis_task = Task(
                description=f"Analyze market conditions and trends for {symbol}",
                agent=self.market_analyst.agent,
                expected_output="Market analysis with trade recommendations",
                context=[data_task]
            )
            
            risk_task = Task(
                description=f"Assess risk factors for trading {symbol}",
                agent=self.risk_manager.agent,
                expected_output="Risk assessment with position sizing recommendations",
                context=[data_task, analysis_task]
            )
            
            trade_task = Task(
                description=f"Make final trading decision for {symbol}",
                agent=self.trader.agent,
                expected_output="Final trade decision with entry/exit points",
                context=[data_task, analysis_task, risk_task]
            )
            
            # Create and run the crew
            crew = Crew(
                agents=[
                    self.data_processor.agent,
                    self.market_analyst.agent,
                    self.risk_manager.agent,
                    self.trader.agent
                ],
                tasks=[data_task, analysis_task, risk_task, trade_task],
                process=Process.sequential,
                verbose=True
            )
            
            # Execute the crew
            result = await asyncio.to_thread(crew.kickoff)
            
            return {
                "symbol": symbol,
                "recommended": "BUY" in str(result).upper() or "LONG" in str(result).upper(),
                "analysis": str(result),
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Error in crew analysis: {str(e)}")
            return {
                "symbol": symbol,
                "recommended": False,
                "analysis": f"Error in analysis: {str(e)}",
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def monitor_positions(self, positions: list) -> Dict[str, Any]:
        """Monitor existing positions and provide recommendations"""
        try:
            monitoring_tasks = []
            
            for position in positions:
                task = Task(
                    description=f"Monitor position for {position['symbol']} and provide recommendations",
                    agent=self.risk_manager.agent,
                    expected_output="Position monitoring recommendations"
                )
                monitoring_tasks.append(task)
            
            if monitoring_tasks:
                crew = Crew(
                    agents=[self.risk_manager.agent, self.market_analyst.agent],
                    tasks=monitoring_tasks,
                    process=Process.parallel,
                    verbose=True
                )
                
                result = await asyncio.to_thread(crew.kickoff)
                return {"recommendations": str(result)}
            
            return {"recommendations": "No positions to monitor"}
            
        except Exception as e:
            logger.error(f"Error in position monitoring: {str(e)}")
            return {"recommendations": f"Error: {str(e)}"}