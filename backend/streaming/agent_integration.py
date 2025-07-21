"""
Integration layer between streaming system and trading agents.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from loguru import logger
from dataclasses import dataclass

from .stream_manager import StreamManager, get_stream_manager
from .realtime_pipeline import StreamMessage, DataQualityStatus


@dataclass
class AgentNotification:
    """Notification sent to agents from streaming system."""
    type: str  # 'market_data', 'sentiment', 'news', 'quality_alert'
    symbol: str
    data: Dict[str, Any]
    quality_score: float
    stream_source: str
    timestamp: datetime
    priority: str = 'normal'  # 'high', 'normal', 'low'


class StreamingAgentBridge:
    """
    Bridge between streaming system and trading agents.
    
    Features:
    - Filters and routes data to appropriate agents
    - Aggregates multi-stream data for agents
    - Handles quality-based routing
    - Provides agent-specific data transformations
    """
    
    def __init__(self, stream_manager: StreamManager = None):
        self.stream_manager = stream_manager or get_stream_manager()
        self.agent_subscriptions: Dict[str, Dict[str, Any]] = {}
        self.data_transformers: Dict[str, Callable] = {}
        self.quality_thresholds: Dict[str, float] = {}
        
        # Default quality thresholds for different agent types
        self.default_quality_thresholds = {
            'market_analyst': 0.7,
            'technical_indicator': 0.8,
            'sentiment_analyst': 0.6,
            'risk_manager': 0.85,
            'trade_executor': 0.9,
            'coordinator': 0.75
        }
        
        logger.info("StreamingAgentBridge initialized")
    
    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        callback: Callable,
        symbols: List[str],
        data_types: List[str] = None,
        quality_threshold: Optional[float] = None
    ):
        """Register an agent for streaming updates."""
        
        # Set quality threshold
        if quality_threshold is None:
            quality_threshold = self.default_quality_thresholds.get(agent_type, 0.7)
        
        self.quality_thresholds[agent_id] = quality_threshold
        
        # Store subscription info
        self.agent_subscriptions[agent_id] = {
            'type': agent_type,
            'callback': callback,
            'symbols': symbols,
            'data_types': data_types or ['market_data', 'sentiment', 'news'],
            'quality_threshold': quality_threshold,
            'messages_received': 0,
            'last_update': None
        }
        
        # Create wrapped callback with filtering
        wrapped_callback = self._create_agent_callback(agent_id)
        
        # Register with stream manager
        self.stream_manager.register_agent_callback(agent_id, wrapped_callback)
        
        # Subscribe to symbols if not already subscribed
        await self.stream_manager.subscribe_symbols(symbols)
        
        logger.info(
            f"Registered agent {agent_id} ({agent_type}) for symbols: {symbols} "
            f"with quality threshold: {quality_threshold}"
        )
    
    def _create_agent_callback(self, agent_id: str) -> Callable:
        """Create a wrapped callback with agent-specific filtering."""
        
        async def wrapped_callback(message: Any):
            try:
                # Get agent subscription
                sub = self.agent_subscriptions.get(agent_id)
                if not sub:
                    return
                
                # Handle different message types
                if isinstance(message, StreamMessage):
                    await self._handle_stream_message(agent_id, message)
                elif isinstance(message, dict):
                    # System messages (quality alerts, status updates)
                    await self._handle_system_message(agent_id, message)
                    
            except Exception as e:
                logger.error(f"Error in agent callback for {agent_id}: {e}")
        
        return wrapped_callback
    
    async def _handle_stream_message(self, agent_id: str, message: StreamMessage):
        """Handle data stream messages."""
        sub = self.agent_subscriptions[agent_id]
        
        # Check symbol subscription
        if message.symbol not in sub['symbols']:
            return
        
        # Check quality threshold
        if message.quality_score < sub['quality_threshold']:
            logger.debug(
                f"Skipping message for {agent_id}: quality {message.quality_score:.2f} "
                f"below threshold {sub['quality_threshold']}"
            )
            return
        
        # Determine data type
        data_type = self._classify_data_type(message)
        
        # Check if agent wants this data type
        if data_type not in sub['data_types']:
            return
        
        # Transform data for agent
        transformed_data = await self._transform_data_for_agent(
            agent_id, 
            sub['type'], 
            message
        )
        
        # Create agent notification
        notification = AgentNotification(
            type=data_type,
            symbol=message.symbol,
            data=transformed_data,
            quality_score=message.quality_score,
            stream_source=message.stream_name,
            timestamp=message.timestamp,
            priority=self._determine_priority(message, sub['type'])
        )
        
        # Send to agent
        await self._send_to_agent(agent_id, notification)
    
    async def _handle_system_message(self, agent_id: str, message: Dict[str, Any]):
        """Handle system messages like quality alerts."""
        sub = self.agent_subscriptions[agent_id]
        
        if message.get('type') == 'stream_switch':
            # Quality-based stream switch notification
            notification = AgentNotification(
                type='quality_alert',
                symbol=message['symbol'],
                data={
                    'alert_type': 'stream_switch',
                    'old_stream': message['old_stream'],
                    'new_stream': message['new_stream'],
                    'reason': message['reason']
                },
                quality_score=0.0,  # System message
                stream_source='system',
                timestamp=datetime.now(),
                priority='high'
            )
            
            await self._send_to_agent(agent_id, notification)
        
        elif message.get('type') == '_status_':
            # Pipeline status update
            if sub['type'] == 'coordinator':  # Only coordinators get status updates
                notification = AgentNotification(
                    type='system_status',
                    symbol='*',
                    data=message,
                    quality_score=1.0,
                    stream_source='system',
                    timestamp=datetime.now(),
                    priority='low'
                )
                
                await self._send_to_agent(agent_id, notification)
    
    def _classify_data_type(self, message: StreamMessage) -> str:
        """Classify the type of data in the message."""
        data = message.data
        
        if data.get('type') == 'social_sentiment':
            return 'sentiment'
        elif data.get('type') == 'news':
            return 'news'
        elif 'price' in data or 'last_price' in data:
            return 'market_data'
        else:
            return 'unknown'
    
    async def _transform_data_for_agent(
        self,
        agent_id: str,
        agent_type: str,
        message: StreamMessage
    ) -> Dict[str, Any]:
        """Transform raw stream data for specific agent types."""
        
        # Get custom transformer if registered
        if agent_id in self.data_transformers:
            return await self.data_transformers[agent_id](message)
        
        # Default transformations by agent type
        if agent_type == 'market_analyst':
            return self._transform_for_market_analyst(message)
        elif agent_type == 'technical_indicator':
            return self._transform_for_technical_analyst(message)
        elif agent_type == 'sentiment_analyst':
            return self._transform_for_sentiment_analyst(message)
        elif agent_type == 'risk_manager':
            return self._transform_for_risk_manager(message)
        elif agent_type == 'trade_executor':
            return self._transform_for_trade_executor(message)
        else:
            return message.data
    
    def _transform_for_market_analyst(self, message: StreamMessage) -> Dict[str, Any]:
        """Transform data for market analyst agent."""
        data = message.data
        
        return {
            'symbol': message.symbol,
            'price': data.get('price') or data.get('last_price'),
            'volume': data.get('volume') or data.get('volume_traded'),
            'change': data.get('change'),
            'change_percent': data.get('change_percent'),
            'high': data.get('high') or data.get('ohlc', {}).get('high'),
            'low': data.get('low') or data.get('ohlc', {}).get('low'),
            'open': data.get('open') or data.get('ohlc', {}).get('open'),
            'timestamp': message.timestamp.isoformat(),
            'quality_score': message.quality_score,
            'latency_ms': message.latency_ms
        }
    
    def _transform_for_technical_analyst(self, message: StreamMessage) -> Dict[str, Any]:
        """Transform data for technical indicator agent."""
        data = message.data
        
        return {
            'symbol': message.symbol,
            'ohlc': {
                'open': data.get('open') or data.get('ohlc', {}).get('open'),
                'high': data.get('high') or data.get('ohlc', {}).get('high'),
                'low': data.get('low') or data.get('ohlc', {}).get('low'),
                'close': data.get('price') or data.get('last_price')
            },
            'volume': data.get('volume') or data.get('volume_traded'),
            'timestamp': message.timestamp.isoformat(),
            'quality_score': message.quality_score
        }
    
    def _transform_for_sentiment_analyst(self, message: StreamMessage) -> Dict[str, Any]:
        """Transform data for sentiment analyst agent."""
        data = message.data
        
        if data.get('type') == 'social_sentiment':
            return {
                'symbol': message.symbol,
                'source': data.get('source', 'unknown'),
                'sentiment_score': data.get('sentiment_score', 0),
                'text': data.get('text', ''),
                'engagement_score': data.get('engagement_score', 0),
                'author_influence': data.get('author_influence', 0),
                'timestamp': message.timestamp.isoformat()
            }
        elif data.get('type') == 'news':
            return {
                'symbol': message.symbol,
                'source': 'news',
                'headline': data.get('title', ''),
                'summary': data.get('summary', ''),
                'sentiment': data.get('sentiment', 0),
                'relevance': data.get('relevance_score', 0),
                'timestamp': message.timestamp.isoformat()
            }
        else:
            return data
    
    def _transform_for_risk_manager(self, message: StreamMessage) -> Dict[str, Any]:
        """Transform data for risk manager agent."""
        data = message.data
        
        return {
            'symbol': message.symbol,
            'price': data.get('price') or data.get('last_price'),
            'volume': data.get('volume') or data.get('volume_traded'),
            'volatility_indicator': data.get('change_percent'),
            'bid_ask_spread': self._calculate_spread(data),
            'quality_score': message.quality_score,
            'latency_ms': message.latency_ms,
            'timestamp': message.timestamp.isoformat()
        }
    
    def _transform_for_trade_executor(self, message: StreamMessage) -> Dict[str, Any]:
        """Transform data for trade executor agent."""
        data = message.data
        
        return {
            'symbol': message.symbol,
            'bid': data.get('depth', {}).get('buy', [{}])[0].get('price'),
            'ask': data.get('depth', {}).get('sell', [{}])[0].get('price'),
            'last_price': data.get('price') or data.get('last_price'),
            'volume': data.get('volume') or data.get('volume_traded'),
            'market_depth': data.get('depth', {}),
            'quality_score': message.quality_score,
            'latency_ms': message.latency_ms,
            'timestamp': message.timestamp.isoformat()
        }
    
    def _calculate_spread(self, data: Dict[str, Any]) -> Optional[float]:
        """Calculate bid-ask spread if available."""
        depth = data.get('depth', {})
        if depth:
            best_bid = depth.get('buy', [{}])[0].get('price')
            best_ask = depth.get('sell', [{}])[0].get('price')
            
            if best_bid and best_ask:
                return (best_ask - best_bid) / best_bid
        
        return None
    
    def _determine_priority(self, message: StreamMessage, agent_type: str) -> str:
        """Determine notification priority based on data and agent type."""
        
        # High priority for execution agents with high quality data
        if agent_type == 'trade_executor' and message.quality_score > 0.9:
            return 'high'
        
        # High priority for risk manager on volatile movements
        if agent_type == 'risk_manager':
            change = message.data.get('change_percent', '0%')
            if isinstance(change, str):
                change = float(change.strip('%'))
            if abs(change) > 2:  # >2% movement
                return 'high'
        
        # High priority for breaking news
        if message.data.get('type') == 'news' and 'breaking' in message.data.get('title', '').lower():
            return 'high'
        
        # Low priority for low quality data
        if message.quality_score < 0.5:
            return 'low'
        
        return 'normal'
    
    async def _send_to_agent(self, agent_id: str, notification: AgentNotification):
        """Send notification to agent."""
        sub = self.agent_subscriptions[agent_id]
        
        try:
            # Update stats
            sub['messages_received'] += 1
            sub['last_update'] = datetime.now()
            
            # Call agent callback
            callback = sub['callback']
            if asyncio.iscoroutinefunction(callback):
                await callback(notification)
            else:
                callback(notification)
                
        except Exception as e:
            logger.error(f"Error sending notification to agent {agent_id}: {e}")
    
    def register_data_transformer(self, agent_id: str, transformer: Callable):
        """Register custom data transformer for an agent."""
        self.data_transformers[agent_id] = transformer
        logger.info(f"Registered custom transformer for agent {agent_id}")
    
    async def get_agent_data_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get summary of data received by an agent."""
        sub = self.agent_subscriptions.get(agent_id)
        if not sub:
            return {}
        
        return {
            'agent_id': agent_id,
            'agent_type': sub['type'],
            'symbols': sub['symbols'],
            'data_types': sub['data_types'],
            'quality_threshold': sub['quality_threshold'],
            'messages_received': sub['messages_received'],
            'last_update': sub['last_update'].isoformat() if sub['last_update'] else None,
            'active': (datetime.now() - sub['last_update']).seconds < 60 if sub['last_update'] else False
        }
    
    async def update_agent_quality_threshold(self, agent_id: str, new_threshold: float):
        """Update quality threshold for an agent."""
        if agent_id in self.agent_subscriptions:
            self.agent_subscriptions[agent_id]['quality_threshold'] = new_threshold
            self.quality_thresholds[agent_id] = new_threshold
            logger.info(f"Updated quality threshold for {agent_id} to {new_threshold}")
    
    async def pause_agent_updates(self, agent_id: str):
        """Temporarily pause updates to an agent."""
        if agent_id in self.agent_subscriptions:
            # Store callback and remove it
            sub = self.agent_subscriptions[agent_id]
            sub['paused_callback'] = sub['callback']
            sub['callback'] = lambda x: None  # No-op callback
            logger.info(f"Paused updates for agent {agent_id}")
    
    async def resume_agent_updates(self, agent_id: str):
        """Resume updates to a paused agent."""
        if agent_id in self.agent_subscriptions:
            sub = self.agent_subscriptions[agent_id]
            if 'paused_callback' in sub:
                sub['callback'] = sub['paused_callback']
                del sub['paused_callback']
                logger.info(f"Resumed updates for agent {agent_id}")
    
    def get_all_agent_summaries(self) -> List[Dict[str, Any]]:
        """Get summary of all registered agents."""
        summaries = []
        for agent_id in self.agent_subscriptions:
            summary = asyncio.run(self.get_agent_data_summary(agent_id))
            summaries.append(summary)
        return summaries