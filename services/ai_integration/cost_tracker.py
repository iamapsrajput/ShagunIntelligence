import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """Record of AI service usage"""
    timestamp: datetime
    provider: str
    use_case: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    success: bool
    metadata: Dict[str, Any]


class CostTracker:
    """Tracks AI service usage costs and enforces budget limits"""
    
    def __init__(self, max_cost_per_hour: float = 10.0, 
                 max_cost_per_day: float = 100.0,
                 save_path: Optional[str] = "data/ai_usage_logs"):
        self.max_cost_per_hour = max_cost_per_hour
        self.max_cost_per_day = max_cost_per_day
        self.save_path = Path(save_path) if save_path else None
        
        # Usage tracking
        self.usage_records: List[UsageRecord] = []
        self.hourly_costs: Dict[str, float] = defaultdict(float)
        self.daily_costs: Dict[str, float] = defaultdict(float)
        
        # Provider-specific tracking
        self.provider_costs = defaultdict(lambda: {
            "total_cost": 0.0,
            "total_tokens": 0,
            "request_count": 0,
            "by_use_case": defaultdict(float),
            "by_model": defaultdict(float)
        })
        
        # Budget alerts
        self.budget_alerts = {
            "50_percent": False,
            "75_percent": False,
            "90_percent": False,
            "exceeded": False
        }
        
        # Cost optimization suggestions
        self.optimization_data = defaultdict(lambda: {
            "avg_tokens_per_request": 0,
            "avg_cost_per_request": 0,
            "high_cost_patterns": []
        })
        
        # Create save directory if needed
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            self._load_historical_data()
        
        logger.info(f"CostTracker initialized - Hourly: ${max_cost_per_hour}, "
                   f"Daily: ${max_cost_per_day}")
    
    def check_budget(self, estimated_cost: float) -> bool:
        """Check if estimated cost is within budget"""
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        current_day = datetime.now().strftime("%Y-%m-%d")
        
        # Check hourly budget
        hourly_total = self.hourly_costs[current_hour] + estimated_cost
        if hourly_total > self.max_cost_per_hour:
            logger.warning(f"Hourly budget would be exceeded: ${hourly_total:.2f}/${self.max_cost_per_hour}")
            return False
        
        # Check daily budget
        daily_total = self.daily_costs[current_day] + estimated_cost
        if daily_total > self.max_cost_per_day:
            logger.warning(f"Daily budget would be exceeded: ${daily_total:.2f}/${self.max_cost_per_day}")
            return False
        
        # Check and trigger budget alerts
        self._check_budget_alerts(daily_total)
        
        return True
    
    def track_usage(self, provider: str, use_case: str, cost: float,
                   model: Optional[str] = None, tokens: Optional[Dict[str, int]] = None,
                   success: bool = True, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track AI service usage"""
        provider_name = provider.value if hasattr(provider, 'value') else str(provider)
        
        # Create usage record
        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider_name,
            use_case=use_case,
            model=model or "unknown",
            prompt_tokens=tokens.get("prompt_tokens", 0) if tokens else 0,
            completion_tokens=tokens.get("completion_tokens", 0) if tokens else 0,
            total_tokens=tokens.get("total_tokens", 0) if tokens else 0,
            cost=cost,
            success=success,
            metadata=metadata or {}
        )
        
        self.usage_records.append(record)
        
        # Update cost tracking
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        current_day = datetime.now().strftime("%Y-%m-%d")
        
        self.hourly_costs[current_hour] += cost
        self.daily_costs[current_day] += cost
        
        # Update provider-specific tracking
        provider_data = self.provider_costs[provider_name]
        provider_data["total_cost"] += cost
        provider_data["total_tokens"] += record.total_tokens
        provider_data["request_count"] += 1
        provider_data["by_use_case"][use_case] += cost
        provider_data["by_model"][record.model] += cost
        
        # Update optimization data
        self._update_optimization_data(provider_name, record)
        
        # Log high-cost requests
        if cost > 0.50:  # Log requests over $0.50
            logger.warning(f"High-cost request: {provider_name} - ${cost:.2f} for {use_case}")
        
        # Auto-save if configured
        if self.save_path and len(self.usage_records) % 100 == 0:
            self.save_to_file()
    
    def _check_budget_alerts(self, daily_total: float) -> None:
        """Check and trigger budget alerts"""
        daily_percentage = (daily_total / self.max_cost_per_day) * 100
        
        if daily_percentage >= 50 and not self.budget_alerts["50_percent"]:
            self.budget_alerts["50_percent"] = True
            logger.warning(f"50% of daily budget used: ${daily_total:.2f}")
        
        if daily_percentage >= 75 and not self.budget_alerts["75_percent"]:
            self.budget_alerts["75_percent"] = True
            logger.warning(f"75% of daily budget used: ${daily_total:.2f}")
        
        if daily_percentage >= 90 and not self.budget_alerts["90_percent"]:
            self.budget_alerts["90_percent"] = True
            logger.error(f"90% of daily budget used: ${daily_total:.2f}")
        
        if daily_percentage >= 100 and not self.budget_alerts["exceeded"]:
            self.budget_alerts["exceeded"] = True
            logger.error(f"Daily budget EXCEEDED: ${daily_total:.2f}")
    
    def _update_optimization_data(self, provider: str, record: UsageRecord) -> None:
        """Update optimization data for cost analysis"""
        opt_data = self.optimization_data[f"{provider}:{record.use_case}"]
        
        # Update averages
        count = self.provider_costs[provider]["by_use_case"][record.use_case]
        
        if count > 0:
            # Running average for tokens
            opt_data["avg_tokens_per_request"] = (
                (opt_data["avg_tokens_per_request"] * (count - 1) + record.total_tokens) / count
            )
            
            # Running average for cost
            opt_data["avg_cost_per_request"] = (
                (opt_data["avg_cost_per_request"] * (count - 1) + record.cost) / count
            )
        
        # Track high-cost patterns
        if record.cost > opt_data["avg_cost_per_request"] * 2:  # 2x average
            pattern = {
                "timestamp": record.timestamp,
                "cost": record.cost,
                "tokens": record.total_tokens,
                "metadata": record.metadata
            }
            opt_data["high_cost_patterns"].append(pattern)
            
            # Keep only recent patterns
            if len(opt_data["high_cost_patterns"]) > 10:
                opt_data["high_cost_patterns"] = opt_data["high_cost_patterns"][-10:]
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        current_day = datetime.now().strftime("%Y-%m-%d")
        
        return {
            "hourly": {
                "cost": self.hourly_costs[current_hour],
                "limit": self.max_cost_per_hour,
                "percentage": (self.hourly_costs[current_hour] / self.max_cost_per_hour * 100),
                "remaining": max(0, self.max_cost_per_hour - self.hourly_costs[current_hour])
            },
            "daily": {
                "cost": self.daily_costs[current_day],
                "limit": self.max_cost_per_day,
                "percentage": (self.daily_costs[current_day] / self.max_cost_per_day * 100),
                "remaining": max(0, self.max_cost_per_day - self.daily_costs[current_day])
            },
            "alerts": self.budget_alerts
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary"""
        # Calculate totals
        total_cost = sum(record.cost for record in self.usage_records)
        total_tokens = sum(record.total_tokens for record in self.usage_records)
        total_requests = len(self.usage_records)
        
        # Get time-based summaries
        hourly_summary = self._get_hourly_summary()
        daily_summary = self._get_daily_summary()
        
        # Get provider breakdown
        provider_summary = {}
        for provider, data in self.provider_costs.items():
            provider_summary[provider] = {
                "total_cost": data["total_cost"],
                "total_tokens": data["total_tokens"],
                "request_count": data["request_count"],
                "average_cost": data["total_cost"] / data["request_count"] if data["request_count"] > 0 else 0,
                "top_use_cases": self._get_top_items(data["by_use_case"], 5),
                "top_models": self._get_top_items(data["by_model"], 3)
            }
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "average_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "current_usage": self.get_current_usage(),
            "hourly_summary": hourly_summary,
            "daily_summary": daily_summary,
            "provider_breakdown": provider_summary,
            "optimization_suggestions": self._get_optimization_suggestions()
        }
    
    def _get_hourly_summary(self) -> Dict[str, Any]:
        """Get hourly cost summary"""
        # Get last 24 hours
        hourly_data = []
        current_time = datetime.now()
        
        for i in range(24):
            hour_time = current_time - timedelta(hours=i)
            hour_key = hour_time.strftime("%Y-%m-%d-%H")
            
            hourly_data.append({
                "hour": hour_time.strftime("%H:00"),
                "date": hour_time.strftime("%Y-%m-%d"),
                "cost": self.hourly_costs.get(hour_key, 0)
            })
        
        return {
            "last_24_hours": list(reversed(hourly_data)),
            "peak_hour": max(hourly_data, key=lambda x: x["cost"]) if hourly_data else None,
            "average_hourly": sum(h["cost"] for h in hourly_data) / len(hourly_data) if hourly_data else 0
        }
    
    def _get_daily_summary(self) -> Dict[str, Any]:
        """Get daily cost summary"""
        # Get last 7 days
        daily_data = []
        current_date = datetime.now().date()
        
        for i in range(7):
            day = current_date - timedelta(days=i)
            day_key = day.strftime("%Y-%m-%d")
            
            daily_data.append({
                "date": day_key,
                "cost": self.daily_costs.get(day_key, 0)
            })
        
        return {
            "last_7_days": list(reversed(daily_data)),
            "total_week": sum(d["cost"] for d in daily_data),
            "average_daily": sum(d["cost"] for d in daily_data) / len(daily_data) if daily_data else 0
        }
    
    def _get_top_items(self, items: Dict[str, float], limit: int) -> List[Dict[str, Any]]:
        """Get top items by cost"""
        sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"name": name, "cost": cost}
            for name, cost in sorted_items[:limit]
        ]
    
    def _get_optimization_suggestions(self) -> List[Dict[str, str]]:
        """Generate cost optimization suggestions"""
        suggestions = []
        
        # Analyze high-cost use cases
        for key, data in self.optimization_data.items():
            if data["avg_cost_per_request"] > 0.25:  # High average cost
                provider, use_case = key.split(":", 1)
                suggestions.append({
                    "type": "high_cost_use_case",
                    "description": f"Consider optimizing {use_case} on {provider}",
                    "detail": f"Average cost: ${data['avg_cost_per_request']:.2f}, "
                             f"Avg tokens: {data['avg_tokens_per_request']:.0f}",
                    "action": "Review prompts, consider using smaller models or caching"
                })
        
        # Check for provider concentration
        if self.provider_costs:
            total_cost = sum(p["total_cost"] for p in self.provider_costs.values())
            for provider, data in self.provider_costs.items():
                if total_cost > 0 and data["total_cost"] / total_cost > 0.8:
                    suggestions.append({
                        "type": "provider_concentration",
                        "description": f"High concentration on {provider}",
                        "detail": f"{data['total_cost'] / total_cost * 100:.1f}% of costs",
                        "action": "Consider distributing load across providers"
                    })
        
        # Check for inefficient token usage
        for provider, data in self.provider_costs.items():
            if data["request_count"] > 0:
                avg_tokens = data["total_tokens"] / data["request_count"]
                if avg_tokens > 2000:
                    suggestions.append({
                        "type": "high_token_usage",
                        "description": f"High average token usage on {provider}",
                        "detail": f"Average: {avg_tokens:.0f} tokens per request",
                        "action": "Consider more concise prompts or response limits"
                    })
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def save_to_file(self, filename: Optional[str] = None) -> None:
        """Save usage data to file"""
        if not self.save_path:
            return
        
        if not filename:
            filename = f"usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.save_path / filename
        
        # Prepare data for serialization
        data = {
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "total_records": len(self.usage_records),
                "date_range": {
                    "start": min(r.timestamp for r in self.usage_records).isoformat() if self.usage_records else None,
                    "end": max(r.timestamp for r in self.usage_records).isoformat() if self.usage_records else None
                }
            },
            "summary": self.get_summary(),
            "records": [
                {
                    **asdict(record),
                    "timestamp": record.timestamp.isoformat()
                }
                for record in self.usage_records[-1000:]  # Save last 1000 records
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved usage data to {filepath}")
    
    def _load_historical_data(self) -> None:
        """Load historical usage data if available"""
        if not self.save_path or not self.save_path.exists():
            return
        
        # Load most recent file
        files = sorted(self.save_path.glob("usage_*.json"))
        if not files:
            return
        
        latest_file = files[-1]
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Restore daily costs from today
            today = datetime.now().strftime("%Y-%m-%d")
            if "summary" in data and "daily_summary" in data["summary"]:
                for day_data in data["summary"]["daily_summary"].get("last_7_days", []):
                    if day_data["date"] == today:
                        self.daily_costs[today] = day_data["cost"]
                        logger.info(f"Restored today's cost: ${day_data['cost']:.2f}")
                        break
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
    
    def reset_daily_budget(self) -> None:
        """Reset daily budget tracking"""
        # Reset alerts
        self.budget_alerts = {
            "50_percent": False,
            "75_percent": False,
            "90_percent": False,
            "exceeded": False
        }
        
        # Clear old daily costs (keep last 7 days)
        cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        self.daily_costs = {
            k: v for k, v in self.daily_costs.items()
            if k >= cutoff_date
        }
        
        logger.info("Daily budget tracking reset")