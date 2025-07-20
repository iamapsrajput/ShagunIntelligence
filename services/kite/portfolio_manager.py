"""Portfolio and position management service"""

import asyncio
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from loguru import logger

from .auth import KiteAuthManager
from .rate_limiter import RateLimiter
from .exceptions import KiteDataError


class PositionType(Enum):
    """Position types"""
    DAY = "day"
    NET = "net"


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    exchange: str
    instrument_token: int
    product: str
    quantity: int
    overnight_quantity: int
    multiplier: float
    average_price: float
    close_price: float
    last_price: float
    value: float
    pnl: float
    m2m: float
    unrealised: float
    realised: float
    buy_quantity: int
    buy_price: float
    buy_value: float
    sell_quantity: int
    sell_price: float
    sell_value: float
    day_buy_quantity: int
    day_buy_price: float
    day_buy_value: float
    day_sell_quantity: int
    day_sell_price: float
    day_sell_value: float
    timestamp: datetime


@dataclass
class Holding:
    """Holding data structure"""
    symbol: str
    exchange: str
    instrument_token: int
    isin: str
    product: str
    price: float
    quantity: int
    used_quantity: int
    t1_quantity: int
    realised_quantity: int
    authorised_quantity: int
    authorised_date: Optional[date]
    opening_quantity: int
    collateral_quantity: int
    collateral_type: str
    discrepancy: bool
    average_price: float
    last_price: float
    close_price: float
    pnl: float
    day_change: float
    day_change_percentage: float


@dataclass
class PortfolioSummary:
    """Portfolio summary data structure"""
    equity: Dict[str, float]
    commodity: Dict[str, float]
    total_portfolio_value: float
    total_pnl: float
    total_day_change: float
    total_day_change_percentage: float
    margin_available: float
    margin_utilised: float
    timestamp: datetime


class PortfolioManager:
    """Comprehensive portfolio and position management"""
    
    def __init__(self, auth_manager: KiteAuthManager):
        self.auth_manager = auth_manager
        self.rate_limiter = RateLimiter(max_requests=30, time_window=60)
        self._positions_cache: Dict[str, Position] = {}
        self._holdings_cache: Dict[str, Holding] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 30  # Cache for 30 seconds
        
    async def get_positions(self, refresh: bool = False) -> Dict[str, List[Position]]:
        """Get all positions (day and net)
        
        Returns:
            Dictionary with 'day' and 'net' position lists
        """
        try:
            if not refresh and self._is_cache_valid():
                logger.debug("Returning cached positions")
                return self._format_positions_response()
            
            await self.rate_limiter.acquire()
            
            # Fetch positions from Kite API
            positions_data = await asyncio.to_thread(self.auth_manager.kite.positions)
            
            # Process and cache positions
            day_positions = []
            net_positions = []
            
            for pos_data in positions_data.get('day', []):
                position = await self._create_position_from_data(pos_data)
                day_positions.append(position)
                self._positions_cache[f"day:{position.symbol}"] = position
            
            for pos_data in positions_data.get('net', []):
                position = await self._create_position_from_data(pos_data)
                net_positions.append(position)
                self._positions_cache[f"net:{position.symbol}"] = position
            
            self._cache_timestamp = datetime.now()
            
            logger.info(f"Fetched {len(day_positions)} day positions and {len(net_positions)} net positions")
            
            return {
                'day': day_positions,
                'net': net_positions
            }
            
        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            raise KiteDataError(f"Positions fetch failed: {str(e)}")
    
    async def get_holdings(self, refresh: bool = False) -> List[Holding]:
        """Get all holdings"""
        try:
            if not refresh and self._is_cache_valid():
                cached_holdings = [
                    holding for key, holding in self._holdings_cache.items()
                    if key.startswith("holding:")
                ]
                if cached_holdings:
                    logger.debug("Returning cached holdings")
                    return cached_holdings
            
            await self.rate_limiter.acquire()
            
            # Fetch holdings from Kite API
            holdings_data = await asyncio.to_thread(self.auth_manager.kite.holdings)
            
            # Process holdings
            holdings = []
            for holding_data in holdings_data:
                holding = await self._create_holding_from_data(holding_data)
                holdings.append(holding)
                self._holdings_cache[f"holding:{holding.symbol}"] = holding
            
            self._cache_timestamp = datetime.now()
            
            logger.info(f"Fetched {len(holdings)} holdings")
            return holdings
            
        except Exception as e:
            logger.error(f"Failed to get holdings: {str(e)}")
            raise KiteDataError(f"Holdings fetch failed: {str(e)}")
    
    async def get_portfolio_summary(self) -> PortfolioSummary:
        """Get comprehensive portfolio summary"""
        try:
            # Get positions and holdings
            positions = await self.get_positions()
            holdings = await self.get_holdings()
            
            # Calculate portfolio metrics
            total_portfolio_value = 0
            total_pnl = 0
            total_day_change = 0
            
            equity_value = 0
            equity_pnl = 0
            commodity_value = 0
            commodity_pnl = 0
            
            # Sum up holdings
            for holding in holdings:
                holding_value = holding.quantity * holding.last_price
                total_portfolio_value += holding_value
                total_pnl += holding.pnl
                total_day_change += holding.day_change
                
                if holding.exchange in ['NSE', 'BSE']:
                    equity_value += holding_value
                    equity_pnl += holding.pnl
                else:
                    commodity_value += holding_value
                    commodity_pnl += holding.pnl
            
            # Sum up net positions
            for position in positions['net']:
                if position.quantity != 0:  # Only count non-zero positions
                    position_value = abs(position.quantity) * position.last_price
                    total_portfolio_value += position_value
                    total_pnl += position.pnl
                    
                    if position.exchange in ['NSE', 'BSE']:
                        equity_value += position_value
                        equity_pnl += position.pnl
                    else:
                        commodity_value += position_value
                        commodity_pnl += position.pnl
            
            # Calculate percentage changes
            total_day_change_percentage = (
                (total_day_change / (total_portfolio_value - total_day_change)) * 100
                if (total_portfolio_value - total_day_change) != 0 else 0
            )
            
            # Get margin information
            margins = await self.get_margins()
            
            summary = PortfolioSummary(
                equity={
                    'value': equity_value,
                    'pnl': equity_pnl,
                    'day_change': equity_pnl  # Simplified
                },
                commodity={
                    'value': commodity_value,
                    'pnl': commodity_pnl,
                    'day_change': commodity_pnl  # Simplified
                },
                total_portfolio_value=total_portfolio_value,
                total_pnl=total_pnl,
                total_day_change=total_day_change,
                total_day_change_percentage=total_day_change_percentage,
                margin_available=margins.get('available', {}).get('cash', 0),
                margin_utilised=margins.get('utilised', {}).get('debits', 0),
                timestamp=datetime.now()
            )
            
            logger.info(f"Generated portfolio summary: Total value ₹{total_portfolio_value:,.2f}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {str(e)}")
            raise KiteDataError(f"Portfolio summary generation failed: {str(e)}")
    
    async def get_position_by_symbol(self, symbol: str, position_type: PositionType = PositionType.NET) -> Optional[Position]:
        """Get position for a specific symbol"""
        try:
            positions = await self.get_positions()
            
            position_list = positions[position_type.value]
            for position in position_list:
                if position.symbol == symbol:
                    return position
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {str(e)}")
            return None
    
    async def get_holding_by_symbol(self, symbol: str) -> Optional[Holding]:
        """Get holding for a specific symbol"""
        try:
            holdings = await self.get_holdings()
            
            for holding in holdings:
                if holding.symbol == symbol:
                    return holding
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get holding for {symbol}: {str(e)}")
            return None
    
    async def get_margins(self) -> Dict[str, Any]:
        """Get margin information"""
        try:
            await self.rate_limiter.acquire()
            
            margins = await asyncio.to_thread(self.auth_manager.kite.margins)
            return margins
            
        except Exception as e:
            logger.error(f"Failed to get margins: {str(e)}")
            raise KiteDataError(f"Margins fetch failed: {str(e)}")
    
    async def get_margin_for_symbol(self, symbol: str, exchange: str = "NSE") -> Dict[str, float]:
        """Get margin required for a specific symbol"""
        try:
            await self.rate_limiter.acquire()
            
            # Get instrument token first
            instruments = await asyncio.to_thread(self.auth_manager.kite.instruments, exchange)
            instrument_token = None
            
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol:
                    instrument_token = instrument['instrument_token']
                    break
            
            if not instrument_token:
                raise ValueError(f"Symbol {symbol} not found in {exchange}")
            
            # Get margin for the instrument
            margin_data = await asyncio.to_thread(
                self.auth_manager.kite.margins,
                [instrument_token]
            )
            
            return margin_data.get(str(instrument_token), {})
            
        except Exception as e:
            logger.error(f"Failed to get margin for {symbol}: {str(e)}")
            raise KiteDataError(f"Margin calculation failed: {str(e)}")
    
    async def calculate_position_metrics(self, positions: List[Position]) -> Dict[str, Any]:
        """Calculate various position metrics"""
        try:
            if not positions:
                return {}
            
            total_value = sum(abs(pos.quantity) * pos.last_price for pos in positions)
            total_pnl = sum(pos.pnl for pos in positions)
            total_unrealised = sum(pos.unrealised for pos in positions)
            total_realised = sum(pos.realised for pos in positions)
            
            profitable_positions = [pos for pos in positions if pos.pnl > 0]
            losing_positions = [pos for pos in positions if pos.pnl < 0]
            
            win_rate = (len(profitable_positions) / len(positions)) * 100 if positions else 0
            
            avg_profit = (
                sum(pos.pnl for pos in profitable_positions) / len(profitable_positions)
                if profitable_positions else 0
            )
            
            avg_loss = (
                sum(pos.pnl for pos in losing_positions) / len(losing_positions)
                if losing_positions else 0
            )
            
            profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            
            return {
                'total_positions': len(positions),
                'total_value': total_value,
                'total_pnl': total_pnl,
                'total_unrealised': total_unrealised,
                'total_realised': total_realised,
                'profitable_positions': len(profitable_positions),
                'losing_positions': len(losing_positions),
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate position metrics: {str(e)}")
            return {}
    
    async def export_portfolio_to_dataframe(self) -> pd.DataFrame:
        """Export complete portfolio data to DataFrame"""
        try:
            # Get all data
            positions = await self.get_positions()
            holdings = await self.get_holdings()
            
            # Combine all data
            all_data = []
            
            # Add holdings
            for holding in holdings:
                all_data.append({
                    'type': 'holding',
                    'symbol': holding.symbol,
                    'exchange': holding.exchange,
                    'quantity': holding.quantity,
                    'average_price': holding.average_price,
                    'last_price': holding.last_price,
                    'pnl': holding.pnl,
                    'day_change': holding.day_change,
                    'value': holding.quantity * holding.last_price,
                    'product': holding.product
                })
            
            # Add net positions
            for position in positions['net']:
                if position.quantity != 0:
                    all_data.append({
                        'type': 'position',
                        'symbol': position.symbol,
                        'exchange': position.exchange,
                        'quantity': position.quantity,
                        'average_price': position.average_price,
                        'last_price': position.last_price,
                        'pnl': position.pnl,
                        'day_change': position.unrealised,
                        'value': abs(position.quantity) * position.last_price,
                        'product': position.product
                    })
            
            df = pd.DataFrame(all_data)
            logger.info(f"Exported portfolio data with {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to export portfolio to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    async def _create_position_from_data(self, data: Dict[str, Any]) -> Position:
        """Create Position object from Kite API data"""
        return Position(
            symbol=data.get('tradingsymbol', ''),
            exchange=data.get('exchange', ''),
            instrument_token=data.get('instrument_token', 0),
            product=data.get('product', ''),
            quantity=data.get('quantity', 0),
            overnight_quantity=data.get('overnight_quantity', 0),
            multiplier=data.get('multiplier', 1),
            average_price=data.get('average_price', 0),
            close_price=data.get('close_price', 0),
            last_price=data.get('last_price', 0),
            value=data.get('value', 0),
            pnl=data.get('pnl', 0),
            m2m=data.get('m2m', 0),
            unrealised=data.get('unrealised', 0),
            realised=data.get('realised', 0),
            buy_quantity=data.get('buy_quantity', 0),
            buy_price=data.get('buy_price', 0),
            buy_value=data.get('buy_value', 0),
            sell_quantity=data.get('sell_quantity', 0),
            sell_price=data.get('sell_price', 0),
            sell_value=data.get('sell_value', 0),
            day_buy_quantity=data.get('day_buy_quantity', 0),
            day_buy_price=data.get('day_buy_price', 0),
            day_buy_value=data.get('day_buy_value', 0),
            day_sell_quantity=data.get('day_sell_quantity', 0),
            day_sell_price=data.get('day_sell_price', 0),
            day_sell_value=data.get('day_sell_value', 0),
            timestamp=datetime.now()
        )
    
    async def _create_holding_from_data(self, data: Dict[str, Any]) -> Holding:
        """Create Holding object from Kite API data"""
        authorised_date = None
        if data.get('authorised_date'):
            try:
                authorised_date = datetime.strptime(data['authorised_date'], '%Y-%m-%d').date()
            except:
                pass
        
        return Holding(
            symbol=data.get('tradingsymbol', ''),
            exchange=data.get('exchange', ''),
            instrument_token=data.get('instrument_token', 0),
            isin=data.get('isin', ''),
            product=data.get('product', ''),
            price=data.get('price', 0),
            quantity=data.get('quantity', 0),
            used_quantity=data.get('used_quantity', 0),
            t1_quantity=data.get('t1_quantity', 0),
            realised_quantity=data.get('realised_quantity', 0),
            authorised_quantity=data.get('authorised_quantity', 0),
            authorised_date=authorised_date,
            opening_quantity=data.get('opening_quantity', 0),
            collateral_quantity=data.get('collateral_quantity', 0),
            collateral_type=data.get('collateral_type', ''),
            discrepancy=data.get('discrepancy', False),
            average_price=data.get('average_price', 0),
            last_price=data.get('last_price', 0),
            close_price=data.get('close_price', 0),
            pnl=data.get('pnl', 0),
            day_change=data.get('day_change', 0),
            day_change_percentage=data.get('day_change_percentage', 0)
        )
    
    def _format_positions_response(self) -> Dict[str, List[Position]]:
        """Format cached positions into response format"""
        day_positions = [
            pos for key, pos in self._positions_cache.items() 
            if key.startswith("day:")
        ]
        net_positions = [
            pos for key, pos in self._positions_cache.items() 
            if key.startswith("net:")
        ]
        
        return {
            'day': day_positions,
            'net': net_positions
        }
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._cache_timestamp:
            return False
        
        cache_age = (datetime.now() - self._cache_timestamp).total_seconds()
        return cache_age < self._cache_ttl_seconds