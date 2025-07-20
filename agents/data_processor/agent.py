from crewai import Agent
from langchain.llms.base import BaseLLM

class DataProcessorAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        return Agent(
            role='Data Processor',
            goal='Collect, clean, and process market data with technical indicators for analysis',
            backstory="""You are a data processing specialist with expertise in financial
            market data analysis. You excel at collecting real-time and historical market
            data, calculating technical indicators, and preparing clean, structured datasets
            for analysis. You have deep knowledge of various technical indicators including
            moving averages, RSI, MACD, Bollinger Bands, and volume-based indicators.
            Your processed data forms the foundation for all trading decisions.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[]
        )