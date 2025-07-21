from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from loguru import logger
import json
import uuid
from typing import Optional

from app.core.config import get_settings
from app.services.websocket_manager import websocket_manager
from app.db.session import get_db
from sqlalchemy.orm import Session

settings = get_settings()
websocket_router = APIRouter()

# Optional authentication for WebSocket
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token", auto_error=False)


async def get_current_user_ws(
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> Optional[str]:
    """Get current user from WebSocket token"""
    if not token:
        return None
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        return username
    except JWTError:
        return None


@websocket_router.websocket("/market")
async def websocket_market_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """WebSocket endpoint for market data updates"""
    client_id = str(uuid.uuid4())
    username = await get_current_user_ws(token)
    
    if username:
        client_id = f"{username}:{client_id}"
    
    await websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "subscribe":
                symbol = data.get("symbol")
                if symbol:
                    websocket_manager.subscribe(client_id, f"market:{symbol}")
                    logger.info(f"Client {client_id} subscribed to {symbol}")
            
            elif data.get("type") == "unsubscribe":
                symbol = data.get("symbol")
                if symbol:
                    websocket_manager.unsubscribe(client_id, f"market:{symbol}")
                    logger.info(f"Client {client_id} unsubscribed from {symbol}")
            
            else:
                await websocket_manager.handle_client_message(client_id, data)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
        logger.info(f"Market WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        websocket_manager.disconnect(client_id)


@websocket_router.websocket("/agents")
async def websocket_agents_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """WebSocket endpoint for agent activity updates"""
    client_id = str(uuid.uuid4())
    username = await get_current_user_ws(token)
    
    if username:
        client_id = f"{username}:{client_id}"
    
    await websocket_manager.connect(websocket, client_id)
    
    # Auto-subscribe to agents topic
    websocket_manager.subscribe(client_id, "agents")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            await websocket_manager.handle_client_message(client_id, data)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
        logger.info(f"Agents WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        websocket_manager.disconnect(client_id)


@websocket_router.websocket("/portfolio")
async def websocket_portfolio_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """WebSocket endpoint for portfolio updates"""
    client_id = str(uuid.uuid4())
    username = await get_current_user_ws(token)
    
    if not username:
        await websocket.close(code=1008, reason="Authentication required")
        return
    
    client_id = f"{username}:{client_id}"
    await websocket_manager.connect(websocket, client_id)
    
    # Auto-subscribe to portfolio and trades
    websocket_manager.subscribe(client_id, "portfolio")
    websocket_manager.subscribe(client_id, "trades")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            await websocket_manager.handle_client_message(client_id, data)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
        logger.info(f"Portfolio WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        websocket_manager.disconnect(client_id)


@websocket_router.websocket("/system")
async def websocket_system_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """WebSocket endpoint for system status updates"""
    client_id = str(uuid.uuid4())
    username = await get_current_user_ws(token)
    
    if not username:
        await websocket.close(code=1008, reason="Authentication required")
        return
    
    client_id = f"{username}:{client_id}"
    await websocket_manager.connect(websocket, client_id)
    
    # Auto-subscribe to system topic
    websocket_manager.subscribe(client_id, "system")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Handle system commands
            if data.get("type") == "command":
                command = data.get("command")
                params = data.get("params", {})
                
                # Process commands (implement based on your needs)
                if command == "system:toggle":
                    # Toggle system on/off
                    pass
                elif command == "update:risk":
                    # Update risk parameters
                    pass
                elif command == "trade:execute":
                    # Execute manual trade
                    pass
                
            await websocket_manager.handle_client_message(client_id, data)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
        logger.info(f"System WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        websocket_manager.disconnect(client_id)


@websocket_router.websocket("/")
async def websocket_general_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """General WebSocket endpoint for all updates"""
    client_id = str(uuid.uuid4())
    username = await get_current_user_ws(token)
    
    if username:
        client_id = f"{username}:{client_id}"
    
    await websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Handle subscription requests
            if data.get("type") == "subscribe":
                topics = data.get("topics", [])
                for topic in topics:
                    websocket_manager.subscribe(client_id, topic)
            
            elif data.get("type") == "unsubscribe":
                topics = data.get("topics", [])
                for topic in topics:
                    websocket_manager.unsubscribe(client_id, topic)
            
            else:
                await websocket_manager.handle_client_message(client_id, data)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
        logger.info(f"General WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        websocket_manager.disconnect(client_id)