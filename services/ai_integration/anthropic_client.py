import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import backoff

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnthropicClient:
    """Anthropic Claude API client with retry logic and error handling"""
    
    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: float = 1.0):
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # API configuration
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
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
        
        logger.info("Anthropic client initialized")
    
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
    async def complete(self, prompt: str, model: str = "claude-3-sonnet-20240229",
                      max_tokens: int = 500, temperature: float = 0.7,
                      system_prompt: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate completion using Anthropic API"""
        await self._ensure_session()
        
        self.stats["total_requests"] += 1
        
        # Prepare messages
        messages = []
        if prompt:
            messages.append({
                "role": "user",
                "content": prompt
            })
        
        # Prepare request
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            start_time = datetime.now()
            
            async with self.session.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    error_msg = response_data.get("error", {}).get("message", "Unknown error")
                    raise Exception(f"Anthropic API error: {error_msg}")
                
                # Extract completion
                completion = response_data["content"][0]["text"]
                
                # Calculate usage
                usage = {
                    "prompt_tokens": response_data.get("usage", {}).get("input_tokens", 0),
                    "completion_tokens": response_data.get("usage", {}).get("output_tokens", 0),
                    "total_tokens": (
                        response_data.get("usage", {}).get("input_tokens", 0) +
                        response_data.get("usage", {}).get("output_tokens", 0)
                    )
                }
                
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
                    "metadata": metadata or {},
                    "stop_reason": response_data.get("stop_reason")
                }
                
                logger.debug(f"Anthropic completion successful - Tokens: {usage['total_tokens']}, Cost: ${cost:.4f}")
                
                return result
                
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    async def complete_with_tools(self, prompt: str, tools: List[Dict[str, Any]],
                                 model: str = "claude-3-sonnet-20240229",
                                 max_tokens: int = 500,
                                 temperature: float = 0.7) -> Dict[str, Any]:
        """Generate completion with tool use (Claude 3 feature)"""
        await self._ensure_session()
        
        messages = [{
            "role": "user",
            "content": prompt
        }]
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": {"type": "auto"}
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    raise Exception(f"Anthropic API error: {response_data}")
                
                content = response_data["content"]
                
                # Check if tool was used
                for item in content:
                    if item["type"] == "tool_use":
                        return {
                            "type": "tool_use",
                            "tool_name": item["name"],
                            "tool_input": item["input"],
                            "usage": response_data.get("usage", {})
                        }
                
                # Otherwise return text
                return {
                    "type": "text",
                    "content": content[0]["text"] if content else "",
                    "usage": response_data.get("usage", {})
                }
                
        except Exception as e:
            logger.error(f"Anthropic tool use error: {str(e)}")
            raise
    
    async def summarize_text(self, text: str, max_length: int = 500) -> Dict[str, Any]:
        """Specialized method for text summarization"""
        prompt = f"""Please provide a concise summary of the following text in no more than {max_length} words:

{text}

Focus on the key points and main insights."""

        system_prompt = """You are an expert at creating clear, concise summaries that capture the essential information while maintaining accuracy. Focus on actionable insights and key takeaways."""
        
        response = await self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            model="claude-3-haiku-20240307",  # Use Haiku for faster, cheaper summarization
            max_tokens=max_length * 2,  # Allow some buffer
            temperature=0.3,  # Lower temperature for consistency
            metadata={"task": "summarization"}
        )
        
        return {
            "summary": response["content"],
            "original_length": len(text.split()),
            "summary_length": len(response["content"].split()),
            "compression_ratio": len(response["content"].split()) / len(text.split()),
            **response
        }
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment with structured output"""
        prompt = f"""Analyze the sentiment of the following text and provide a structured analysis:

Text: {text}

Provide your analysis in the following JSON format:
{{
    "overall_sentiment": "positive/negative/neutral",
    "sentiment_score": -1.0 to 1.0,
    "confidence": 0.0 to 1.0,
    "key_phrases": ["phrase1", "phrase2"],
    "emotional_tone": "descriptive word",
    "reasoning": "brief explanation"
}}"""

        system_prompt = """You are an expert sentiment analyst. Analyze text objectively and provide nuanced insights. Always respond in valid JSON format."""
        
        response = await self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            model="claude-3-sonnet-20240229",
            max_tokens=400,
            temperature=0.3,
            metadata={"task": "sentiment_analysis"}
        )
        
        # Parse JSON response
        try:
            analysis = json.loads(response["content"])
            analysis["raw_response"] = response["content"]
            return analysis
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.5,
                "reasoning": response["content"],
                "parse_error": True
            }
    
    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost based on token usage"""
        # Pricing per 1M tokens (as of 2024)
        pricing = {
            "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
            "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "claude-2.1": {"input": 8.00, "output": 24.00},
            "claude-2.0": {"input": 8.00, "output": 24.00}
        }
        
        rates = pricing.get(model, pricing["claude-3-sonnet-20240229"])
        
        input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * rates["input"]
        output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * rates["output"]
        
        return input_cost + output_cost
    
    async def compare_options(self, options: List[Dict[str, Any]], criteria: List[str]) -> Dict[str, Any]:
        """Compare multiple options based on criteria"""
        prompt = f"""Compare the following options based on the given criteria:

Options:
{json.dumps(options, indent=2)}

Criteria:
{json.dumps(criteria, indent=2)}

Provide a structured comparison with rankings and reasoning for each criterion."""

        response = await self.complete(
            prompt=prompt,
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.5,
            metadata={"task": "comparison"}
        )
        
        return {
            "comparison": response["content"],
            "options_count": len(options),
            "criteria_count": len(criteria),
            **response
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
        logger.info("Anthropic client closed")