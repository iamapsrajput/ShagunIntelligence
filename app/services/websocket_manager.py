import json
from typing import Dict, List, Set, Any
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from datetime import datetime
import asyncio


class ConnectionManager:
    """Manages WebSocket connections and broadcasting"""
    
    def __init__(self):
        # Store active connections by client ID
        self.active_connections: Dict[str, WebSocket] = {}
        # Store subscriptions by topic
        self.subscriptions: Dict[str, Set[str]] = {}
        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            "connected_at": datetime.utcnow(),
            "subscriptions": set()
        }
        logger.info(f"WebSocket client {client_id} connected")
        
        # Send welcome message
        await self.send_personal_message(
            {
                "type": "connection",
                "status": "connected",
                "client_id": client_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            client_id
        )
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
            # Remove from all subscriptions
            for topic in list(self.subscriptions.keys()):
                if client_id in self.subscriptions[topic]:
                    self.subscriptions[topic].remove(client_id)
                    if not self.subscriptions[topic]:
                        del self.subscriptions[topic]
            
            # Remove metadata
            if client_id in self.connection_metadata:
                del self.connection_metadata[client_id]
            
            logger.info(f"WebSocket client {client_id} disconnected")
    
    async def disconnect_all(self):
        """Disconnect all clients"""
        client_ids = list(self.active_connections.keys())
        for client_id in client_ids:
            try:
                await self.active_connections[client_id].close()
            except Exception as e:
                logger.error(f"Error closing connection {client_id}: {e}")
            self.disconnect(client_id)
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any], topic: str = None):
        """Broadcast message to all connected clients or topic subscribers"""
        if topic and topic in self.subscriptions:
            # Send to topic subscribers only
            disconnected = []
            for client_id in self.subscriptions[topic]:
                if client_id in self.active_connections:
                    try:
                        await self.active_connections[client_id].send_json(message)
                    except Exception as e:
                        logger.error(f"Error broadcasting to {client_id}: {e}")
                        disconnected.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected:
                self.disconnect(client_id)
        else:
            # Broadcast to all
            disconnected = []
            for client_id, websocket in self.active_connections.items():
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    disconnected.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected:
                self.disconnect(client_id)
    
    def subscribe(self, client_id: str, topic: str):
        """Subscribe client to a topic"""
        if client_id in self.active_connections:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
            
            self.subscriptions[topic].add(client_id)
            
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id]["subscriptions"].add(topic)
            
            logger.info(f"Client {client_id} subscribed to {topic}")
    
    def unsubscribe(self, client_id: str, topic: str):
        """Unsubscribe client from a topic"""
        if topic in self.subscriptions and client_id in self.subscriptions[topic]:
            self.subscriptions[topic].remove(client_id)
            
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
            
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id]["subscriptions"].discard(topic)
            
            logger.info(f"Client {client_id} unsubscribed from {topic}")
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_client_info(self, client_id: str) -> Dict[str, Any]:
        """Get client connection info"""
        if client_id in self.connection_metadata:
            return {
                "connected": True,
                **self.connection_metadata[client_id]
            }
        return {"connected": False}
    
    def get_topic_subscribers(self, topic: str) -> List[str]:
        """Get list of subscribers for a topic"""
        return list(self.subscriptions.get(topic, set()))
    
    async def handle_client_message(self, client_id: str, data: Dict[str, Any]):
        """Handle incoming client messages"""
        message_type = data.get("type")
        
        if message_type == "subscribe":
            topic = data.get("topic")
            if topic:
                self.subscribe(client_id, topic)
                await self.send_personal_message(
                    {
                        "type": "subscription",
                        "status": "subscribed",
                        "topic": topic,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    client_id
                )
        
        elif message_type == "unsubscribe":
            topic = data.get("topic")
            if topic:
                self.unsubscribe(client_id, topic)
                await self.send_personal_message(
                    {
                        "type": "subscription",
                        "status": "unsubscribed",
                        "topic": topic,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    client_id
                )
        
        elif message_type == "ping":
            await self.send_personal_message(
                {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                },
                client_id
            )
        
        else:
            logger.warning(f"Unknown message type from {client_id}: {message_type}")


# Broadcast helpers for specific data types
class WebSocketBroadcaster:
    """Helper class for broadcasting specific types of updates"""
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
    
    async def broadcast_market_update(self, symbol: str, data: Dict[str, Any]):
        """Broadcast market data update"""
        message = {
            "type": "market:update",
            "data": {
                "symbol": symbol,
                **data,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        await self.manager.broadcast(message, f"market:{symbol}")
    
    async def broadcast_agent_activity(self, agent_data: Dict[str, Any]):
        """Broadcast agent activity"""
        message = {
            "type": "agent:activity",
            "data": agent_data
        }
        await self.manager.broadcast(message, "agents")
    
    async def broadcast_trade_execution(self, trade_data: Dict[str, Any]):
        """Broadcast trade execution"""
        message = {
            "type": "trade:execution",
            "data": trade_data
        }
        await self.manager.broadcast(message, "trades")
    
    async def broadcast_portfolio_update(self, portfolio_data: Dict[str, Any]):
        """Broadcast portfolio update"""
        message = {
            "type": "portfolio:update",
            "data": portfolio_data
        }
        await self.manager.broadcast(message, "portfolio")
    
    async def broadcast_system_status(self, status_data: Dict[str, Any]):
        """Broadcast system status"""
        message = {
            "type": "system:status",
            "data": status_data
        }
        await self.manager.broadcast(message, "system")
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Broadcast alert or notification"""
        message = {
            "type": "alert",
            "data": alert_data
        }
        await self.manager.broadcast(message)


# Global instances
websocket_manager = ConnectionManager()
websocket_broadcaster = WebSocketBroadcaster(websocket_manager)