from crewai import Agent
from langchain.llms.base import BaseLLM

class RiskManagerAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        return Agent(
            role='Risk Manager',
            goal='Assess and manage trading risks, calculate position sizes, and ensure compliance with risk parameters',
            backstory="""You are a seasoned risk management expert with extensive experience in
            portfolio risk assessment and position sizing. You understand the importance of
            capital preservation and have deep knowledge of various risk metrics including
            VaR, maximum drawdown, and position sizing strategies. Your primary focus is to
            protect capital while allowing for profitable trading opportunities.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[]
        )