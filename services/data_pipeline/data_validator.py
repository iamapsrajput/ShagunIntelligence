import logging
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Represents a data validation rule"""
    name: str
    field: str
    rule_type: str  # range, required, format, consistency
    params: Dict[str, Any]
    severity: str  # error, warning, info
    auto_correct: bool = False


class DataValidator:
    """Validates and sanitizes market data with error recovery"""
    
    def __init__(self):
        # Validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        # Data quality tracking
        self.quality_stats = {
            "total_validations": 0,
            "passed": 0,
            "failed": 0,
            "corrected": 0,
            "warnings": 0
        }
        
        # Error patterns
        self.error_patterns = {}
        
        # Price limits (circuit breaker)
        self.price_limits = {
            "daily_limit_percent": 20,  # 20% daily movement
            "tick_limit_percent": 5     # 5% per tick movement
        }
        
        # Last known good values (for recovery)
        self.last_good_values: Dict[int, Dict[str, Any]] = {}
        
        logger.info("DataValidator initialized")
    
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize default validation rules"""
        return [
            # Required fields
            ValidationRule(
                name="instrument_token_required",
                field="instrument_token",
                rule_type="required",
                params={},
                severity="error"
            ),
            ValidationRule(
                name="last_price_required",
                field="last_price",
                rule_type="required",
                params={},
                severity="error"
            ),
            ValidationRule(
                name="timestamp_required",
                field="timestamp",
                rule_type="required",
                params={},
                severity="error"
            ),
            
            # Price validations
            ValidationRule(
                name="price_positive",
                field="last_price",
                rule_type="range",
                params={"min": 0.01, "max": float('inf')},
                severity="error"
            ),
            ValidationRule(
                name="price_reasonable",
                field="last_price",
                rule_type="range",
                params={"min": 0.01, "max": 1000000},
                severity="warning"
            ),
            
            # Volume validations
            ValidationRule(
                name="volume_non_negative",
                field="volume",
                rule_type="range",
                params={"min": 0, "max": float('inf')},
                severity="error",
                auto_correct=True
            ),
            ValidationRule(
                name="quantity_non_negative",
                field="last_quantity",
                rule_type="range",
                params={"min": 0, "max": float('inf')},
                severity="error",
                auto_correct=True
            ),
            
            # Timestamp validations
            ValidationRule(
                name="timestamp_not_future",
                field="timestamp",
                rule_type="timestamp",
                params={"max_future_seconds": 5},
                severity="error"
            ),
            ValidationRule(
                name="timestamp_not_stale",
                field="timestamp",
                rule_type="timestamp",
                params={"max_age_seconds": 300},  # 5 minutes
                severity="warning"
            ),
            
            # OHLC consistency
            ValidationRule(
                name="ohlc_consistency",
                field="ohlc",
                rule_type="consistency",
                params={},
                severity="error",
                auto_correct=True
            ),
            
            # Spread validation
            ValidationRule(
                name="spread_reasonable",
                field="spread",
                rule_type="spread",
                params={"max_spread_percent": 5},
                severity="warning"
            )
        ]
    
    def validate_tick(self, tick_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate a single tick and return cleaned data"""
        self.quality_stats["total_validations"] += 1
        
        validation_errors = []
        validation_warnings = []
        cleaned_data = tick_data.copy()
        
        # Apply validation rules
        for rule in self.validation_rules:
            is_valid, message, corrected_value = self._apply_rule(rule, cleaned_data)
            
            if not is_valid:
                if rule.severity == "error":
                    validation_errors.append(f"{rule.name}: {message}")
                    
                    # Apply auto-correction if enabled
                    if rule.auto_correct and corrected_value is not None:
                        cleaned_data[rule.field] = corrected_value
                        self.quality_stats["corrected"] += 1
                        logger.debug(f"Auto-corrected {rule.field}: {corrected_value}")
                        
                elif rule.severity == "warning":
                    validation_warnings.append(f"{rule.name}: {message}")
                    self.quality_stats["warnings"] += 1
        
        # Additional consistency checks
        cleaned_data = self._ensure_data_consistency(cleaned_data)
        
        # Circuit breaker check
        if not self._check_circuit_breaker(cleaned_data):
            validation_errors.append("Circuit breaker triggered")
        
        # Determine overall validity
        is_valid = len(validation_errors) == 0
        
        if is_valid:
            self.quality_stats["passed"] += 1
            
            # Store as last known good value
            instrument_token = cleaned_data.get("instrument_token")
            if instrument_token:
                self.last_good_values[instrument_token] = cleaned_data.copy()
        else:
            self.quality_stats["failed"] += 1
            
            # Try to recover using last known good values
            recovered_data = self._attempt_recovery(cleaned_data)
            if recovered_data:
                cleaned_data = recovered_data
                is_valid = True
                self.quality_stats["corrected"] += 1
                logger.info("Recovered tick data using last known good values")
        
        # Log warnings
        for warning in validation_warnings:
            logger.debug(f"Validation warning: {warning}")
        
        return is_valid, cleaned_data
    
    def _apply_rule(self, rule: ValidationRule, data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Apply a single validation rule"""
        try:
            if rule.rule_type == "required":
                return self._validate_required(rule, data)
            elif rule.rule_type == "range":
                return self._validate_range(rule, data)
            elif rule.rule_type == "timestamp":
                return self._validate_timestamp(rule, data)
            elif rule.rule_type == "consistency":
                return self._validate_consistency(rule, data)
            elif rule.rule_type == "spread":
                return self._validate_spread(rule, data)
            else:
                return True, None, None
                
        except Exception as e:
            logger.error(f"Error applying rule {rule.name}: {str(e)}")
            return False, str(e), None
    
    def _validate_required(self, rule: ValidationRule, data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate required field"""
        if rule.field not in data or data[rule.field] is None:
            return False, f"Required field '{rule.field}' is missing", None
        return True, None, None
    
    def _validate_range(self, rule: ValidationRule, data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate numeric range"""
        if rule.field not in data:
            return True, None, None  # Skip if field not present
        
        value = data[rule.field]
        if value is None:
            return True, None, None
        
        try:
            value = float(value)
            min_val = rule.params.get("min", float('-inf'))
            max_val = rule.params.get("max", float('inf'))
            
            if value < min_val or value > max_val:
                # Auto-correct if enabled
                if rule.auto_correct:
                    corrected = max(min_val, min(value, max_val))
                    return False, f"Value {value} outside range [{min_val}, {max_val}]", corrected
                else:
                    return False, f"Value {value} outside range [{min_val}, {max_val}]", None
                    
        except (TypeError, ValueError):
            return False, f"Invalid numeric value for '{rule.field}'", None
        
        return True, None, None
    
    def _validate_timestamp(self, rule: ValidationRule, data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate timestamp"""
        if rule.field not in data:
            return True, None, None
        
        timestamp = data[rule.field]
        
        try:
            # Parse timestamp
            if isinstance(timestamp, str):
                ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, datetime):
                ts = timestamp
            else:
                return False, "Invalid timestamp format", None
            
            now = datetime.now()
            
            # Check future timestamp
            max_future = rule.params.get("max_future_seconds", 0)
            if ts > now + timedelta(seconds=max_future):
                return False, f"Timestamp is {(ts - now).total_seconds():.1f}s in the future", None
            
            # Check stale timestamp
            max_age = rule.params.get("max_age_seconds")
            if max_age and ts < now - timedelta(seconds=max_age):
                age = (now - ts).total_seconds()
                return False, f"Timestamp is {age:.1f}s old", None
                
        except Exception as e:
            return False, f"Invalid timestamp: {str(e)}", None
        
        return True, None, None
    
    def _validate_consistency(self, rule: ValidationRule, data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate OHLC consistency"""
        if rule.field not in data or not data[rule.field]:
            return True, None, None
        
        ohlc = data[rule.field]
        
        try:
            open_price = float(ohlc.get("open", 0))
            high_price = float(ohlc.get("high", 0))
            low_price = float(ohlc.get("low", 0))
            close_price = float(ohlc.get("close", 0))
            
            # Check basic OHLC rules
            if high_price < low_price:
                if rule.auto_correct:
                    # Swap high and low
                    ohlc["high"], ohlc["low"] = low_price, high_price
                    return False, "High < Low (corrected)", ohlc
                return False, "High price less than low price", None
            
            if open_price > high_price or open_price < low_price:
                if rule.auto_correct:
                    ohlc["high"] = max(high_price, open_price)
                    ohlc["low"] = min(low_price, open_price)
                    return False, "Open outside high-low range (corrected)", ohlc
                return False, "Open price outside high-low range", None
            
            if close_price > high_price or close_price < low_price:
                if rule.auto_correct:
                    ohlc["high"] = max(high_price, close_price)
                    ohlc["low"] = min(low_price, close_price)
                    return False, "Close outside high-low range (corrected)", ohlc
                return False, "Close price outside high-low range", None
                
        except (TypeError, ValueError) as e:
            return False, f"Invalid OHLC data: {str(e)}", None
        
        return True, None, None
    
    def _validate_spread(self, rule: ValidationRule, data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Validate bid-ask spread"""
        bid_depth = data.get("bid_depth", [])
        ask_depth = data.get("ask_depth", [])
        
        if not bid_depth or not ask_depth:
            return True, None, None
        
        try:
            best_bid = float(bid_depth[0]["price"])
            best_ask = float(ask_depth[0]["price"])
            
            if best_bid >= best_ask:
                return False, "Bid price >= Ask price", None
            
            spread_percent = ((best_ask - best_bid) / best_ask) * 100
            max_spread = rule.params.get("max_spread_percent", 5)
            
            if spread_percent > max_spread:
                return False, f"Spread {spread_percent:.2f}% exceeds limit", None
                
        except (IndexError, KeyError, TypeError, ValueError):
            return True, None, None  # Skip if data is incomplete
        
        return True, None, None
    
    def _ensure_data_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure overall data consistency"""
        # Ensure numeric fields are properly typed
        numeric_fields = ["last_price", "volume", "last_quantity", "average_price",
                         "buy_quantity", "sell_quantity"]
        
        for field in numeric_fields:
            if field in data and data[field] is not None:
                try:
                    data[field] = float(data[field])
                except (TypeError, ValueError):
                    data[field] = 0.0
        
        # Ensure timestamp format
        if "timestamp" in data and isinstance(data["timestamp"], str):
            try:
                data["timestamp"] = datetime.fromisoformat(
                    data["timestamp"].replace('Z', '+00:00')
                )
            except ValueError:
                data["timestamp"] = datetime.now()
        
        return data
    
    def _check_circuit_breaker(self, data: Dict[str, Any]) -> bool:
        """Check if price movement triggers circuit breaker"""
        instrument_token = data.get("instrument_token")
        if not instrument_token:
            return True
        
        last_price = data.get("last_price", 0)
        if not last_price:
            return True
        
        # Check against last known good value
        if instrument_token in self.last_good_values:
            last_good = self.last_good_values[instrument_token]
            last_good_price = last_good.get("last_price", last_price)
            
            # Calculate price change
            price_change_percent = abs((last_price - last_good_price) / last_good_price) * 100
            
            # Check tick limit
            if price_change_percent > self.price_limits["tick_limit_percent"]:
                logger.warning(f"Price movement {price_change_percent:.2f}% exceeds tick limit")
                return False
        
        return True
    
    def _attempt_recovery(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to recover invalid data using last known good values"""
        instrument_token = data.get("instrument_token")
        if not instrument_token or instrument_token not in self.last_good_values:
            return None
        
        last_good = self.last_good_values[instrument_token]
        recovered = data.copy()
        
        # Recover critical fields
        critical_fields = ["last_price", "volume", "ohlc"]
        
        for field in critical_fields:
            if field not in recovered or recovered[field] is None:
                if field in last_good:
                    recovered[field] = last_good[field]
                    logger.debug(f"Recovered {field} from last known good value")
        
        return recovered
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Get data quality statistics"""
        total = self.quality_stats["total_validations"]
        if total == 0:
            return self.quality_stats
        
        return {
            **self.quality_stats,
            "pass_rate": (self.quality_stats["passed"] / total) * 100,
            "fail_rate": (self.quality_stats["failed"] / total) * 100,
            "correction_rate": (self.quality_stats["corrected"] / total) * 100,
            "warning_rate": (self.quality_stats["warnings"] / total) * 100
        }
    
    def add_custom_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule"""
        self.validation_rules.append(rule)
        logger.info(f"Added custom validation rule: {rule.name}")