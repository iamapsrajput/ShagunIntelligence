from crewai import Agent
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain.llms.base import BaseLLM

class MarketAnalystAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        return Agent(
            role='Market Analyst',
            goal='Analyze market conditions, trends, and provide technical analysis for trading decisions',
            backstory="""You are an experienced market analyst with deep expertise in technical analysis,
            market psychology, and trend identification. You have years of experience in analyzing
            Indian stock markets and understand the nuances of intraday trading. Your analysis is
            based on technical indicators, chart patterns, volume analysis, and market sentiment.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[
                # Add tools for market analysis
                # SerperDevTool(),  # For web search if needed
                # WebsiteSearchTool()  # For specific website search
            ]
        )