import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
import asyncio
from dataclasses import dataclass

from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .prompt_templates import PromptTemplates
from .rate_limiter import RateLimiter
from .cost_tracker import CostTracker
from .response_cache import ResponseCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """Available AI service providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    
    
class AIModel(Enum):
    """Available AI models"""
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT35_TURBO = "gpt-3.5-turbo"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"


@dataclass
class AIServiceConfig:
    """Configuration for AI services"""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 150000
    
    # Cost limits
    max_cost_per_hour: float = 10.0
    max_cost_per_day: float = 100.0
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Fallback settings
    enable_fallback: bool = True
    primary_provider: AIProvider = AIProvider.OPENAI
    fallback_provider: AIProvider = AIProvider.ANTHROPIC


class AIServiceManager:
    """Manages AI service integrations with rate limiting and cost management"""
    
    def __init__(self, config: AIServiceConfig):
        self.config = config
        
        # Initialize AI clients
        self.clients: Dict[AIProvider, Any] = {}
        
        if config.openai_api_key:
            self.clients[AIProvider.OPENAI] = OpenAIClient(
                api_key=config.openai_api_key,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay
            )
            
        if config.anthropic_api_key:
            self.clients[AIProvider.ANTHROPIC] = AnthropicClient(
                api_key=config.anthropic_api_key,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay
            )
        
        # Initialize components
        self.prompt_templates = PromptTemplates()
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=config.max_requests_per_minute,
            max_tokens_per_minute=config.max_tokens_per_minute
        )
        self.cost_tracker = CostTracker(
            max_cost_per_hour=config.max_cost_per_hour,
            max_cost_per_day=config.max_cost_per_day
        )
        
        # Initialize cache if enabled
        self.cache = None
        if config.cache_enabled:
            self.cache = ResponseCache(ttl_seconds=config.cache_ttl_seconds)
        
        # Service health tracking
        self.service_health = {
            AIProvider.OPENAI: {"available": True, "last_error": None},
            AIProvider.ANTHROPIC: {"available": True, "last_error": None}
        }
        
        logger.info("AIServiceManager initialized")
    
    async def analyze_market_sentiment(self, text: str, 
                                     provider: Optional[AIProvider] = None) -> Dict[str, Any]:
        """Analyze market sentiment using AI"""
        prompt = self.prompt_templates.get_market_sentiment_prompt(text)
        
        response = await self._execute_request(
            prompt=prompt,
            provider=provider,
            use_case="market_sentiment",
            max_tokens=500
        )
        
        return self._parse_sentiment_response(response)
    
    async def generate_trade_analysis(self, market_data: Dict[str, Any],
                                    provider: Optional[AIProvider] = None) -> Dict[str, Any]:
        """Generate comprehensive trade analysis"""
        prompt = self.prompt_templates.get_trade_analysis_prompt(market_data)
        
        response = await self._execute_request(
            prompt=prompt,
            provider=provider,
            use_case="trade_analysis",
            max_tokens=1000
        )
        
        return self._parse_trade_analysis_response(response)
    
    async def summarize_news(self, articles: List[str],
                           provider: Optional[AIProvider] = None) -> str:
        """Summarize multiple news articles"""
        prompt = self.prompt_templates.get_news_summary_prompt(articles)
        
        # Prefer Claude for summarization
        if not provider and AIProvider.ANTHROPIC in self.clients:
            provider = AIProvider.ANTHROPIC
        
        response = await self._execute_request(
            prompt=prompt,
            provider=provider,
            use_case="news_summary",
            max_tokens=500
        )
        
        return response.get("content", "")
    
    async def extract_key_metrics(self, financial_text: str,
                                provider: Optional[AIProvider] = None) -> Dict[str, Any]:
        """Extract key financial metrics from text"""
        prompt = self.prompt_templates.get_metric_extraction_prompt(financial_text)
        
        response = await self._execute_request(
            prompt=prompt,
            provider=provider,
            use_case="metric_extraction",
            max_tokens=500
        )
        
        return self._parse_metrics_response(response)
    
    async def generate_risk_assessment(self, trade_data: Dict[str, Any],
                                     provider: Optional[AIProvider] = None) -> Dict[str, Any]:
        """Generate AI-powered risk assessment"""
        prompt = self.prompt_templates.get_risk_assessment_prompt(trade_data)
        
        response = await self._execute_request(
            prompt=prompt,
            provider=provider,
            use_case="risk_assessment",
            max_tokens=800
        )
        
        return self._parse_risk_assessment_response(response)
    
    async def _execute_request(self, prompt: str, provider: Optional[AIProvider],
                             use_case: str, max_tokens: int = 500,
                             temperature: float = 0.7) -> Dict[str, Any]:
        """Execute AI request with fallback and error handling"""
        # Check cache first
        cache_key = f"{use_case}:{hash(prompt)}"
        if self.cache:
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                logger.info(f"Cache hit for {use_case}")
                return cached_response
        
        # Select provider
        if not provider:
            provider = self.config.primary_provider
        
        # Check rate limits
        if not await self.rate_limiter.check_request(provider, max_tokens):
            logger.warning(f"Rate limit exceeded for {provider.value}")
            
            # Try fallback provider
            if self.config.enable_fallback:
                provider = self._get_fallback_provider(provider)
                if not provider or not await self.rate_limiter.check_request(provider, max_tokens):
                    raise Exception("All providers rate limited")
        
        # Check cost limits
        estimated_cost = self._estimate_cost(provider, prompt, max_tokens)
        if not self.cost_tracker.check_budget(estimated_cost):
            raise Exception("Cost limit exceeded")
        
        # Execute request
        try:
            response = await self._call_provider(
                provider=provider,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                use_case=use_case
            )
            
            # Track usage
            actual_cost = self._calculate_actual_cost(provider, response)
            self.cost_tracker.track_usage(provider, use_case, actual_cost)
            
            # Update rate limiter
            await self.rate_limiter.record_request(
                provider,
                response.get("usage", {}).get("total_tokens", max_tokens)
            )
            
            # Cache response
            if self.cache:
                await self.cache.set(cache_key, response)
            
            # Mark service as healthy
            self.service_health[provider]["available"] = True
            
            return response
            
        except Exception as e:
            logger.error(f"Error calling {provider.value}: {str(e)}")
            
            # Mark service as unhealthy
            self.service_health[provider]["available"] = False
            self.service_health[provider]["last_error"] = str(e)
            
            # Try fallback
            if self.config.enable_fallback:
                fallback_provider = self._get_fallback_provider(provider)
                if fallback_provider and self.service_health[fallback_provider]["available"]:
                    logger.info(f"Falling back to {fallback_provider.value}")
                    return await self._execute_request(
                        prompt=prompt,
                        provider=fallback_provider,
                        use_case=use_case,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
            
            raise
    
    async def _call_provider(self, provider: AIProvider, prompt: str,
                           max_tokens: int, temperature: float,
                           use_case: str) -> Dict[str, Any]:
        """Call specific AI provider"""
        if provider not in self.clients:
            raise ValueError(f"Provider {provider.value} not configured")
        
        client = self.clients[provider]
        
        if provider == AIProvider.OPENAI:
            return await client.complete(
                prompt=prompt,
                model=AIModel.GPT4_TURBO.value,
                max_tokens=max_tokens,
                temperature=temperature,
                metadata={"use_case": use_case}
            )
        elif provider == AIProvider.ANTHROPIC:
            return await client.complete(
                prompt=prompt,
                model=AIModel.CLAUDE_3_SONNET.value,
                max_tokens=max_tokens,
                temperature=temperature,
                metadata={"use_case": use_case}
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _get_fallback_provider(self, primary: AIProvider) -> Optional[AIProvider]:
        """Get fallback provider"""
        if primary == AIProvider.OPENAI:
            return AIProvider.ANTHROPIC if AIProvider.ANTHROPIC in self.clients else None
        else:
            return AIProvider.OPENAI if AIProvider.OPENAI in self.clients else None
    
    def _estimate_cost(self, provider: AIProvider, prompt: str, max_tokens: int) -> float:
        """Estimate cost for a request"""
        # Rough token estimation (4 chars = 1 token)
        prompt_tokens = len(prompt) // 4
        
        # Cost per 1K tokens (approximate)
        costs = {
            AIProvider.OPENAI: {"input": 0.01, "output": 0.03},  # GPT-4
            AIProvider.ANTHROPIC: {"input": 0.008, "output": 0.024}  # Claude 3
        }
        
        provider_costs = costs.get(provider, costs[AIProvider.OPENAI])
        
        input_cost = (prompt_tokens / 1000) * provider_costs["input"]
        output_cost = (max_tokens / 1000) * provider_costs["output"]
        
        return input_cost + output_cost
    
    def _calculate_actual_cost(self, provider: AIProvider, response: Dict[str, Any]) -> float:
        """Calculate actual cost from response"""
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        costs = {
            AIProvider.OPENAI: {"input": 0.01, "output": 0.03},
            AIProvider.ANTHROPIC: {"input": 0.008, "output": 0.024}
        }
        
        provider_costs = costs.get(provider, costs[AIProvider.OPENAI])
        
        input_cost = (prompt_tokens / 1000) * provider_costs["input"]
        output_cost = (completion_tokens / 1000) * provider_costs["output"]
        
        return input_cost + output_cost
    
    def _parse_sentiment_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse sentiment analysis response"""
        content = response.get("content", "")
        
        # Default sentiment
        result = {
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.5,
            "key_factors": [],
            "raw_response": content
        }
        
        # Parse structured response
        try:
            import json
            # Try to extract JSON from response
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                parsed = json.loads(json_str)
                result.update(parsed)
        except:
            # Fallback to keyword analysis
            content_lower = content.lower()
            if "bullish" in content_lower or "positive" in content_lower:
                result["sentiment"] = "bullish"
                result["score"] = 0.7
            elif "bearish" in content_lower or "negative" in content_lower:
                result["sentiment"] = "bearish"
                result["score"] = -0.7
        
        return result
    
    def _parse_trade_analysis_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse trade analysis response"""
        content = response.get("content", "")
        
        result = {
            "recommendation": "hold",
            "confidence": 0.5,
            "entry_points": [],
            "exit_points": [],
            "risk_factors": [],
            "opportunities": [],
            "raw_response": content
        }
        
        # Parse structured response
        try:
            import json
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                parsed = json.loads(json_str)
                result.update(parsed)
        except:
            pass
        
        return result
    
    def _parse_metrics_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse metrics extraction response"""
        content = response.get("content", "")
        
        result = {
            "metrics": {},
            "raw_response": content
        }
        
        try:
            import json
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                parsed = json.loads(json_str)
                result["metrics"] = parsed.get("metrics", {})
        except:
            pass
        
        return result
    
    def _parse_risk_assessment_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse risk assessment response"""
        content = response.get("content", "")
        
        result = {
            "risk_level": "medium",
            "risk_score": 5.0,
            "risk_factors": [],
            "mitigation_strategies": [],
            "confidence": 0.7,
            "raw_response": content
        }
        
        try:
            import json
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                parsed = json.loads(json_str)
                result.update(parsed)
        except:
            pass
        
        return result
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all AI services"""
        return {
            "services": {
                provider.value: {
                    "configured": provider in self.clients,
                    "available": self.service_health[provider]["available"],
                    "last_error": self.service_health[provider]["last_error"]
                }
                for provider in AIProvider
            },
            "rate_limits": self.rate_limiter.get_status(),
            "cost_tracking": self.cost_tracker.get_summary(),
            "cache_stats": self.cache.get_stats() if self.cache else None
        }
    
    async def shutdown(self):
        """Shutdown AI service manager"""
        # Close client connections
        for client in self.clients.values():
            if hasattr(client, 'close'):
                await client.close()
        
        # Save cost tracking data
        self.cost_tracker.save_to_file()
        
        logger.info("AIServiceManager shutdown complete")