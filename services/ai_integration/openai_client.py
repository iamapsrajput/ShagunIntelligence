import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import backoff

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI API client with retry logic and error handling"""
    
    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: float = 1.0):
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # API configuration
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }
        
        logger.info("OpenAI client initialized")
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def complete(self, prompt: str, model: str = "gpt-4-turbo-preview",
                      max_tokens: int = 500, temperature: float = 0.7,
                      system_prompt: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate completion using OpenAI API"""
        await self._ensure_session()
        
        self.stats["total_requests"] += 1
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        
        # Add response format for GPT-4
        if model.startswith("gpt-4"):
            payload["response_format"] = {"type": "json_object"}
        
        try:
            start_time = datetime.now()
            
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    error_msg = response_data.get("error", {}).get("message", "Unknown error")
                    raise Exception(f"OpenAI API error: {error_msg}")
                
                # Extract completion
                completion = response_data["choices"][0]["message"]["content"]
                usage = response_data["usage"]
                
                # Update statistics
                self.stats["successful_requests"] += 1
                self.stats["total_tokens"] += usage["total_tokens"]
                
                # Calculate cost
                cost = self._calculate_cost(model, usage)
                self.stats["total_cost"] += cost
                
                # Prepare response
                result = {
                    "content": completion,
                    "usage": usage,
                    "model": model,
                    "cost": cost,
                    "latency": (datetime.now() - start_time).total_seconds(),
                    "metadata": metadata or {}
                }
                
                logger.debug(f"OpenAI completion successful - Tokens: {usage['total_tokens']}, Cost: ${cost:.4f}")
                
                return result
                
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def complete_with_functions(self, prompt: str, functions: List[Dict[str, Any]],
                                    model: str = "gpt-4-turbo-preview",
                                    max_tokens: int = 500,
                                    temperature: float = 0.7) -> Dict[str, Any]:
        """Generate completion with function calling"""
        await self._ensure_session()
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": model,
            "messages": messages,
            "functions": functions,
            "function_call": "auto",
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    raise Exception(f"OpenAI API error: {response_data}")
                
                choice = response_data["choices"][0]
                
                # Check if function was called
                if "function_call" in choice["message"]:
                    function_call = choice["message"]["function_call"]
                    return {
                        "type": "function_call",
                        "function_name": function_call["name"],
                        "arguments": json.loads(function_call["arguments"]),
                        "usage": response_data["usage"]
                    }
                else:
                    return {
                        "type": "text",
                        "content": choice["message"]["content"],
                        "usage": response_data["usage"]
                    }
                    
        except Exception as e:
            logger.error(f"OpenAI function calling error: {str(e)}")
            raise
    
    async def create_embedding(self, text: str, 
                             model: str = "text-embedding-3-small") -> List[float]:
        """Create text embedding"""
        await self._ensure_session()
        
        payload = {
            "model": model,
            "input": text
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    raise Exception(f"OpenAI API error: {response_data}")
                
                embedding = response_data["data"][0]["embedding"]
                
                # Update stats
                self.stats["total_tokens"] += response_data["usage"]["total_tokens"]
                
                return embedding
                
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise
    
    async def moderate_content(self, text: str) -> Dict[str, Any]:
        """Check content for policy violations"""
        await self._ensure_session()
        
        payload = {"input": text}
        
        try:
            async with self.session.post(
                f"{self.base_url}/moderations",
                headers=self.headers,
                json=payload
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    raise Exception(f"OpenAI API error: {response_data}")
                
                return response_data["results"][0]
                
        except Exception as e:
            logger.error(f"OpenAI moderation error: {str(e)}")
            raise
    
    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost based on token usage"""
        # Pricing per 1K tokens (as of 2024)
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "text-embedding-3-small": {"input": 0.00002, "output": 0},
            "text-embedding-3-large": {"input": 0.00013, "output": 0}
        }
        
        # Get model family
        model_family = model
        for family in pricing.keys():
            if model.startswith(family):
                model_family = family
                break
        
        rates = pricing.get(model_family, pricing["gpt-3.5-turbo"])
        
        input_cost = (usage.get("prompt_tokens", 0) / 1000) * rates["input"]
        output_cost = (usage.get("completion_tokens", 0) / 1000) * rates["output"]
        
        return input_cost + output_cost
    
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized method for market analysis"""
        prompt = f"""Analyze the following market data and provide insights:

Market Data:
{json.dumps(market_data, indent=2)}

Provide your analysis in the following JSON format:
{{
    "trend": "bullish/bearish/neutral",
    "strength": 0-10,
    "key_levels": {{"support": [], "resistance": []}},
    "signals": [],
    "confidence": 0-1,
    "reasoning": "detailed explanation"
}}"""

        system_prompt = """You are an expert financial analyst specializing in technical and fundamental analysis. 
        Provide clear, actionable insights based on the data provided. Always respond in valid JSON format."""
        
        response = await self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            model="gpt-4-turbo-preview",
            max_tokens=800,
            temperature=0.3,  # Lower temperature for more consistent analysis
            metadata={"analysis_type": "market_data"}
        )
        
        # Parse JSON response
        try:
            analysis = json.loads(response["content"])
            analysis["raw_response"] = response["content"]
            return analysis
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "trend": "neutral",
                "strength": 5,
                "confidence": 0.5,
                "reasoning": response["content"],
                "parse_error": True
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            **self.stats,
            "average_cost_per_request": (
                self.stats["total_cost"] / self.stats["successful_requests"]
                if self.stats["successful_requests"] > 0 else 0
            ),
            "average_tokens_per_request": (
                self.stats["total_tokens"] / self.stats["successful_requests"]
                if self.stats["successful_requests"] > 0 else 0
            ),
            "success_rate": (
                self.stats["successful_requests"] / self.stats["total_requests"] * 100
                if self.stats["total_requests"] > 0 else 0
            )
        }
    
    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info("OpenAI client closed")