"""Data validation and sanitization utilities for Kite Connect service"""

import re
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from loguru import logger

from .exceptions import KiteValidationError


class Exchange(Enum):
    """Supported exchanges"""
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"
    BFO = "BFO"
    CDS = "CDS"
    MCX = "MCX"


class InstrumentType(Enum):
    """Instrument types"""
    EQ = "EQ"  # Equity
    FUT = "FUT"  # Futures
    CE = "CE"  # Call Option
    PE = "PE"  # Put Option
    COMMODITY = "COMMODITY"


@dataclass
class ValidationResult:
    """Validation result structure"""
    is_valid: bool
    value: Any = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class DataValidator:
    """Comprehensive data validation for trading operations"""
    
    # Symbol pattern validation
    SYMBOL_PATTERNS = {
        Exchange.NSE: re.compile(r'^[A-Z0-9&\-]+$'),
        Exchange.BSE: re.compile(r'^[A-Z0-9&\-]+$'),
        Exchange.NFO: re.compile(r'^[A-Z0-9&\-]+\d{2}[A-Z]{3}\d{2}(CE|PE)\d+$|^[A-Z0-9&\-]+\d{2}[A-Z]{3}\d{2}FUT$'),
        Exchange.BFO: re.compile(r'^[A-Z0-9&\-]+\d{2}[A-Z]{3}\d{2}(CE|PE)\d+$|^[A-Z0-9&\-]+\d{2}[A-Z]{3}\d{2}FUT$'),
        Exchange.MCX: re.compile(r'^[A-Z0-9&\-]+\d{2}[A-Z]{3}\d{2}FUT$'),
    }
    
    # Price limits (in INR)
    MIN_PRICE = 0.01
    MAX_PRICE = 1000000.0  # 10 Lakh
    
    # Quantity limits
    MIN_QUANTITY = 1
    MAX_QUANTITY = 1000000
    
    # Date limits
    MIN_DATE = date(2000, 1, 1)
    MAX_DATE = date(2030, 12, 31)
    
    @staticmethod
    def validate_symbol(symbol: str, exchange: str = "NSE") -> ValidationResult:
        """Validate trading symbol format"""
        result = ValidationResult(is_valid=False)
        
        try:
            # Basic validation
            if not symbol or not isinstance(symbol, str):
                result.errors.append("Symbol must be a non-empty string")
                return result
            
            # Clean symbol
            cleaned_symbol = symbol.strip().upper()
            
            # Length validation
            if len(cleaned_symbol) < 1 or len(cleaned_symbol) > 30:
                result.errors.append("Symbol length must be between 1 and 30 characters")
                return result
            
            # Exchange-specific pattern validation
            try:
                exchange_enum = Exchange(exchange.upper())
                pattern = DataValidator.SYMBOL_PATTERNS.get(exchange_enum)
                
                if pattern and not pattern.match(cleaned_symbol):
                    result.errors.append(f"Symbol '{cleaned_symbol}' doesn't match {exchange} format")
                    return result
                    
            except ValueError:
                result.warnings.append(f"Unknown exchange '{exchange}', skipping pattern validation")
            
            result.is_valid = True
            result.value = cleaned_symbol
            
        except Exception as e:
            result.errors.append(f"Symbol validation error: {str(e)}")
        
        return result
    
    @staticmethod
    def validate_price(price: Union[float, int, str, Decimal], allow_zero: bool = False) -> ValidationResult:
        """Validate price value"""
        result = ValidationResult(is_valid=False)
        
        try:
            # Convert to Decimal for precise validation
            if isinstance(price, str):
                price = price.strip()
                if not price:
                    result.errors.append("Price cannot be empty")
                    return result
            
            try:
                decimal_price = Decimal(str(price))
            except (InvalidOperation, ValueError):
                result.errors.append(f"Invalid price format: {price}")
                return result
            
            # Convert to float for further validation
            float_price = float(decimal_price)
            
            # Zero check
            if not allow_zero and float_price == 0:
                result.errors.append("Price cannot be zero")
                return result
            
            # Negative check
            if float_price < 0:
                result.errors.append("Price cannot be negative")
                return result
            
            # Range validation
            if float_price > DataValidator.MAX_PRICE:
                result.errors.append(f"Price {float_price} exceeds maximum limit {DataValidator.MAX_PRICE}")
                return result
            
            if float_price > 0 and float_price < DataValidator.MIN_PRICE:
                result.errors.append(f"Price {float_price} below minimum limit {DataValidator.MIN_PRICE}")
                return result
            
            # Precision check (max 2 decimal places for Indian markets)
            if decimal_price.as_tuple().exponent < -2:
                result.warnings.append("Price has more than 2 decimal places, will be rounded")
                float_price = round(float_price, 2)
            
            result.is_valid = True
            result.value = float_price
            
        except Exception as e:
            result.errors.append(f"Price validation error: {str(e)}")
        
        return result
    
    @staticmethod
    def validate_quantity(quantity: Union[int, str]) -> ValidationResult:
        """Validate order quantity"""
        result = ValidationResult(is_valid=False)
        
        try:
            # Convert to integer
            if isinstance(quantity, str):
                quantity = quantity.strip()
                if not quantity:
                    result.errors.append("Quantity cannot be empty")
                    return result
            
            try:
                int_quantity = int(quantity)
            except (ValueError, TypeError):
                result.errors.append(f"Invalid quantity format: {quantity}")
                return result
            
            # Positive check
            if int_quantity <= 0:
                result.errors.append("Quantity must be positive")
                return result
            
            # Range validation
            if int_quantity < DataValidator.MIN_QUANTITY:
                result.errors.append(f"Quantity {int_quantity} below minimum {DataValidator.MIN_QUANTITY}")
                return result
            
            if int_quantity > DataValidator.MAX_QUANTITY:
                result.errors.append(f"Quantity {int_quantity} exceeds maximum {DataValidator.MAX_QUANTITY}")
                return result
            
            result.is_valid = True
            result.value = int_quantity
            
        except Exception as e:
            result.errors.append(f"Quantity validation error: {str(e)}")
        
        return result
    
    @staticmethod
    def validate_date(input_date: Union[str, date, datetime], allow_future: bool = True) -> ValidationResult:
        """Validate date input"""
        result = ValidationResult(is_valid=False)
        
        try:
            # Convert to date object
            if isinstance(input_date, str):
                input_date = input_date.strip()
                if not input_date:
                    result.errors.append("Date cannot be empty")
                    return result
                
                # Try parsing common date formats
                date_formats = [
                    "%Y-%m-%d",
                    "%d-%m-%Y",
                    "%d/%m/%Y",
                    "%Y/%m/%d",
                    "%d-%m-%y",
                    "%d/%m/%y"
                ]
                
                parsed_date = None
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(input_date, fmt).date()
                        break
                    except ValueError:
                        continue
                
                if not parsed_date:
                    result.errors.append(f"Invalid date format: {input_date}")
                    return result
                
                input_date = parsed_date
                
            elif isinstance(input_date, datetime):
                input_date = input_date.date()
            elif not isinstance(input_date, date):
                result.errors.append(f"Invalid date type: {type(input_date)}")
                return result
            
            # Range validation
            if input_date < DataValidator.MIN_DATE:
                result.errors.append(f"Date {input_date} is too far in the past")
                return result
            
            if input_date > DataValidator.MAX_DATE:
                result.errors.append(f"Date {input_date} is too far in the future")
                return result
            
            # Future date check
            if not allow_future and input_date > date.today():
                result.errors.append(f"Future date not allowed: {input_date}")
                return result
            
            # Weekend check for trading dates
            if input_date.weekday() >= 5:  # Saturday or Sunday
                result.warnings.append(f"Date {input_date} is a weekend")
            
            result.is_valid = True
            result.value = input_date
            
        except Exception as e:
            result.errors.append(f"Date validation error: {str(e)}")
        
        return result
    
    @staticmethod
    def validate_order_data(order_data: Dict[str, Any]) -> ValidationResult:
        """Validate complete order data"""
        result = ValidationResult(is_valid=True)
        validated_data = {}
        
        try:
            # Required fields
            required_fields = ['symbol', 'exchange', 'transaction_type', 'quantity', 'order_type', 'product']
            
            for field in required_fields:
                if field not in order_data:
                    result.errors.append(f"Missing required field: {field}")
                    result.is_valid = False
            
            if not result.is_valid:
                return result
            
            # Validate symbol
            symbol_result = DataValidator.validate_symbol(
                order_data['symbol'], 
                order_data.get('exchange', 'NSE')
            )
            if not symbol_result.is_valid:
                result.errors.extend(symbol_result.errors)
                result.is_valid = False
            else:
                validated_data['symbol'] = symbol_result.value
                result.warnings.extend(symbol_result.warnings)
            
            # Validate exchange
            try:
                Exchange(order_data['exchange'].upper())
                validated_data['exchange'] = order_data['exchange'].upper()
            except ValueError:
                result.errors.append(f"Invalid exchange: {order_data['exchange']}")
                result.is_valid = False
            
            # Validate transaction type
            valid_transaction_types = ['BUY', 'SELL']
            transaction_type = order_data['transaction_type'].upper()
            if transaction_type not in valid_transaction_types:
                result.errors.append(f"Invalid transaction type: {transaction_type}")
                result.is_valid = False
            else:
                validated_data['transaction_type'] = transaction_type
            
            # Validate quantity
            quantity_result = DataValidator.validate_quantity(order_data['quantity'])
            if not quantity_result.is_valid:
                result.errors.extend(quantity_result.errors)
                result.is_valid = False
            else:
                validated_data['quantity'] = quantity_result.value
            
            # Validate order type
            valid_order_types = ['MARKET', 'LIMIT', 'SL', 'SL-M']
            order_type = order_data['order_type'].upper()
            if order_type not in valid_order_types:
                result.errors.append(f"Invalid order type: {order_type}")
                result.is_valid = False
            else:
                validated_data['order_type'] = order_type
            
            # Validate price (if required)
            if order_type in ['LIMIT', 'SL'] and 'price' in order_data:
                price_result = DataValidator.validate_price(order_data['price'])
                if not price_result.is_valid:
                    result.errors.extend(price_result.errors)
                    result.is_valid = False
                else:
                    validated_data['price'] = price_result.value
                    result.warnings.extend(price_result.warnings)
            
            # Validate trigger price (if required)
            if order_type in ['SL', 'SL-M'] and 'trigger_price' in order_data:
                trigger_result = DataValidator.validate_price(order_data['trigger_price'])
                if not trigger_result.is_valid:
                    result.errors.extend(trigger_result.errors)
                    result.is_valid = False
                else:
                    validated_data['trigger_price'] = trigger_result.value
            
            # Validate product type
            valid_products = ['CNC', 'MIS', 'NRML']
            product = order_data['product'].upper()
            if product not in valid_products:
                result.errors.append(f"Invalid product type: {product}")
                result.is_valid = False
            else:
                validated_data['product'] = product
            
            # Copy other valid fields
            optional_fields = ['validity', 'disclosed_quantity', 'tag']
            for field in optional_fields:
                if field in order_data:
                    validated_data[field] = order_data[field]
            
            if result.is_valid:
                result.value = validated_data
            
        except Exception as e:
            result.errors.append(f"Order validation error: {str(e)}")
            result.is_valid = False
        
        return result
    
    @staticmethod
    def validate_historical_data_request(
        symbol: str,
        from_date: Union[str, date, datetime],
        to_date: Union[str, date, datetime],
        interval: str,
        exchange: str = "NSE"
    ) -> ValidationResult:
        """Validate historical data request parameters"""
        result = ValidationResult(is_valid=True)
        validated_data = {}
        
        try:
            # Validate symbol
            symbol_result = DataValidator.validate_symbol(symbol, exchange)
            if not symbol_result.is_valid:
                result.errors.extend(symbol_result.errors)
                result.is_valid = False
            else:
                validated_data['symbol'] = symbol_result.value
            
            # Validate from_date
            from_date_result = DataValidator.validate_date(from_date, allow_future=False)
            if not from_date_result.is_valid:
                result.errors.extend([f"From date: {err}" for err in from_date_result.errors])
                result.is_valid = False
            else:
                validated_data['from_date'] = from_date_result.value
            
            # Validate to_date
            to_date_result = DataValidator.validate_date(to_date, allow_future=False)
            if not to_date_result.is_valid:
                result.errors.extend([f"To date: {err}" for err in to_date_result.errors])
                result.is_valid = False
            else:
                validated_data['to_date'] = to_date_result.value
            
            # Validate date range
            if result.is_valid:
                if validated_data['from_date'] > validated_data['to_date']:
                    result.errors.append("From date cannot be after to date")
                    result.is_valid = False
                
                # Check if date range is too large
                date_diff = (validated_data['to_date'] - validated_data['from_date']).days
                max_days = {
                    'minute': 60,
                    '3minute': 60,
                    '5minute': 60,
                    '15minute': 200,
                    '30minute': 200,
                    '60minute': 400,
                    'day': 2000
                }
                
                max_allowed = max_days.get(interval, 365)
                if date_diff > max_allowed:
                    result.warnings.append(f"Date range ({date_diff} days) is large for {interval} interval")
            
            # Validate interval
            valid_intervals = ['minute', '3minute', '5minute', '15minute', '30minute', '60minute', 'day']
            if interval not in valid_intervals:
                result.errors.append(f"Invalid interval: {interval}")
                result.is_valid = False
            else:
                validated_data['interval'] = interval
            
            # Validate exchange
            try:
                Exchange(exchange.upper())
                validated_data['exchange'] = exchange.upper()
            except ValueError:
                result.errors.append(f"Invalid exchange: {exchange}")
                result.is_valid = False
            
            if result.is_valid:
                result.value = validated_data
            
        except Exception as e:
            result.errors.append(f"Historical data request validation error: {str(e)}")
            result.is_valid = False
        
        return result


class DataSanitizer:
    """Data sanitization utilities"""
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """Sanitize trading symbol"""
        if not symbol:
            return ""
        
        # Remove special characters except allowed ones
        sanitized = re.sub(r'[^A-Za-z0-9&\-]', '', symbol.strip())
        return sanitized.upper()
    
    @staticmethod
    def sanitize_price(price: Union[float, str, Decimal]) -> float:
        """Sanitize price value"""
        try:
            float_price = float(price)
            return round(max(0, float_price), 2)
        except (ValueError, TypeError):
            return 0.0
    
    @staticmethod
    def sanitize_quantity(quantity: Union[int, str]) -> int:
        """Sanitize quantity value"""
        try:
            int_quantity = int(float(quantity))
            return max(0, int_quantity)
        except (ValueError, TypeError):
            return 0
    
    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize OHLC DataFrame"""
        if df.empty:
            return df
        
        try:
            sanitized_df = df.copy()
            
            # Sanitize numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in sanitized_df.columns:
                    # Replace inf and -inf with NaN
                    sanitized_df[col] = sanitized_df[col].replace([float('inf'), float('-inf')], pd.NA)
                    
                    # Fill NaN values
                    if col == 'volume':
                        sanitized_df[col] = sanitized_df[col].fillna(0)
                    else:
                        # Forward fill price data
                        sanitized_df[col] = sanitized_df[col].fillna(method='ffill')
                    
                    # Ensure positive values for prices
                    if col != 'volume':
                        sanitized_df[col] = sanitized_df[col].abs()
            
            # Validate OHLC relationships
            if all(col in sanitized_df.columns for col in ['open', 'high', 'low', 'close']):
                # Ensure high >= max(open, close) and low <= min(open, close)
                sanitized_df['high'] = sanitized_df[['high', 'open', 'close']].max(axis=1)
                sanitized_df['low'] = sanitized_df[['low', 'open', 'close']].min(axis=1)
            
            # Remove rows with all NaN values
            sanitized_df = sanitized_df.dropna(how='all')
            
            logger.debug(f"Sanitized DataFrame: {len(df)} -> {len(sanitized_df)} rows")
            return sanitized_df
            
        except Exception as e:
            logger.error(f"DataFrame sanitization error: {str(e)}")
            return df
    
    @staticmethod
    def sanitize_order_response(order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize order response data"""
        sanitized = {}
        
        try:
            # String fields
            string_fields = ['order_id', 'status', 'tradingsymbol', 'exchange', 'transaction_type']
            for field in string_fields:
                if field in order_data:
                    value = order_data[field]
                    sanitized[field] = str(value).strip() if value is not None else ""
            
            # Numeric fields
            numeric_fields = ['quantity', 'price', 'average_price', 'filled_quantity']
            for field in numeric_fields:
                if field in order_data:
                    try:
                        sanitized[field] = float(order_data[field]) if order_data[field] is not None else 0.0
                    except (ValueError, TypeError):
                        sanitized[field] = 0.0
            
            # Integer fields
            int_fields = ['instrument_token']
            for field in int_fields:
                if field in order_data:
                    try:
                        sanitized[field] = int(order_data[field]) if order_data[field] is not None else 0
                    except (ValueError, TypeError):
                        sanitized[field] = 0
            
            # Copy other fields as-is
            other_fields = set(order_data.keys()) - set(string_fields + numeric_fields + int_fields)
            for field in other_fields:
                sanitized[field] = order_data[field]
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Order response sanitization error: {str(e)}")
            return order_data