import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
import threading
from multiprocessing import shared_memory
import pickle
import json
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SharedDataSegment:
    """Represents a shared memory segment"""
    name: str
    size: int
    memory: Optional[shared_memory.SharedMemory] = None
    last_update: Optional[datetime] = None
    version: int = 0


class DataSynchronizer:
    """Manages data synchronization between agents using shared memory"""
    
    def __init__(self):
        # Shared memory management
        self.segments: Dict[str, SharedDataSegment] = {}
        self.segment_lock = threading.Lock()
        
        # Agent registry
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Synchronization queues
        self.sync_queues: Dict[str, asyncio.Queue] = {}
        
        # Configuration
        self.segment_size = 1024 * 1024  # 1MB per segment
        self.max_segments = 100
        
        # Statistics
        self.stats = {
            "total_syncs": 0,
            "sync_errors": 0,
            "active_segments": 0,
            "total_bytes_synced": 0
        }
        
        logger.info("DataSynchronizer initialized")
    
    def register_agent(self, agent_name: str, callback: Optional[Callable] = None) -> None:
        """Register an agent for data synchronization"""
        with self.segment_lock:
            self.registered_agents[agent_name] = {
                "registered_at": datetime.now(),
                "callback": callback,
                "last_sync": None
            }
            
            # Create sync queue for agent
            self.sync_queues[agent_name] = asyncio.Queue()
            
            logger.info(f"Registered agent: {agent_name}")
    
    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent"""
        with self.segment_lock:
            if agent_name in self.registered_agents:
                del self.registered_agents[agent_name]
                
                # Remove subscriptions
                for symbol in list(self.agent_subscriptions.keys()):
                    self.agent_subscriptions[symbol].discard(agent_name)
                
                # Remove sync queue
                if agent_name in self.sync_queues:
                    del self.sync_queues[agent_name]
                
                logger.info(f"Unregistered agent: {agent_name}")
    
    def subscribe_agent_to_symbol(self, agent_name: str, symbol: str) -> None:
        """Subscribe an agent to symbol updates"""
        if agent_name not in self.registered_agents:
            logger.warning(f"Agent {agent_name} not registered")
            return
        
        self.agent_subscriptions[symbol].add(agent_name)
        logger.debug(f"Agent {agent_name} subscribed to {symbol}")
    
    async def sync_tick(self, symbol: str, tick_data: Dict[str, Any]) -> None:
        """Synchronize tick data across agents"""
        try:
            # Get or create shared segment for symbol
            segment = await self._get_or_create_segment(symbol)
            
            # Serialize data
            serialized_data = self._serialize_data(tick_data)
            
            # Write to shared memory
            await self._write_to_segment(segment, serialized_data)
            
            # Notify subscribed agents
            await self._notify_agents(symbol, tick_data)
            
            # Update statistics
            self.stats["total_syncs"] += 1
            self.stats["total_bytes_synced"] += len(serialized_data)
            
        except Exception as e:
            logger.error(f"Error syncing tick for {symbol}: {str(e)}")
            self.stats["sync_errors"] += 1
    
    async def get_shared_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data from shared memory"""
        try:
            segment = self.segments.get(symbol)
            if not segment or not segment.memory:
                return None
            
            # Read from shared memory
            data = await self._read_from_segment(segment)
            
            if data:
                return self._deserialize_data(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error reading shared data for {symbol}: {str(e)}")
            return None
    
    async def broadcast_data(self, data_type: str, data: Dict[str, Any]) -> None:
        """Broadcast data to all registered agents"""
        for agent_name in self.registered_agents:
            try:
                await self.sync_queues[agent_name].put({
                    "type": data_type,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error broadcasting to {agent_name}: {str(e)}")
    
    async def _get_or_create_segment(self, symbol: str) -> SharedDataSegment:
        """Get or create a shared memory segment for a symbol"""
        with self.segment_lock:
            if symbol in self.segments:
                return self.segments[symbol]
            
            # Check segment limit
            if len(self.segments) >= self.max_segments:
                # Remove oldest segment
                oldest_symbol = min(
                    self.segments.keys(),
                    key=lambda s: self.segments[s].last_update or datetime.min
                )
                await self._remove_segment(oldest_symbol)
            
            # Create new segment
            segment = SharedDataSegment(
                name=f"shagunintelligence_{symbol}_{datetime.now().timestamp()}",
                size=self.segment_size
            )
            
            try:
                # Create shared memory
                segment.memory = shared_memory.SharedMemory(
                    create=True,
                    size=segment.size
                )
                
                self.segments[symbol] = segment
                self.stats["active_segments"] = len(self.segments)
                
                logger.debug(f"Created shared segment for {symbol}")
                
            except Exception as e:
                logger.error(f"Error creating shared segment: {str(e)}")
                raise
            
            return segment
    
    async def _write_to_segment(self, segment: SharedDataSegment, data: bytes) -> None:
        """Write data to shared memory segment"""
        if not segment.memory:
            raise ValueError("Segment memory not initialized")
        
        if len(data) > segment.size:
            raise ValueError(f"Data size {len(data)} exceeds segment size {segment.size}")
        
        try:
            # Write size header (4 bytes)
            size_bytes = len(data).to_bytes(4, byteorder='big')
            
            # Write version (4 bytes)
            segment.version += 1
            version_bytes = segment.version.to_bytes(4, byteorder='big')
            
            # Write data
            segment.memory.buf[:4] = size_bytes
            segment.memory.buf[4:8] = version_bytes
            segment.memory.buf[8:8+len(data)] = data
            
            segment.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error writing to shared segment: {str(e)}")
            raise
    
    async def _read_from_segment(self, segment: SharedDataSegment) -> Optional[bytes]:
        """Read data from shared memory segment"""
        if not segment.memory:
            return None
        
        try:
            # Read size header
            size = int.from_bytes(segment.memory.buf[:4], byteorder='big')
            
            if size == 0 or size > segment.size:
                return None
            
            # Read version
            version = int.from_bytes(segment.memory.buf[4:8], byteorder='big')
            
            # Read data
            data = bytes(segment.memory.buf[8:8+size])
            
            return data
            
        except Exception as e:
            logger.error(f"Error reading from shared segment: {str(e)}")
            return None
    
    async def _remove_segment(self, symbol: str) -> None:
        """Remove a shared memory segment"""
        if symbol in self.segments:
            segment = self.segments[symbol]
            
            if segment.memory:
                try:
                    segment.memory.close()
                    segment.memory.unlink()
                except Exception as e:
                    logger.error(f"Error removing segment: {str(e)}")
            
            del self.segments[symbol]
            self.stats["active_segments"] = len(self.segments)
            
            logger.debug(f"Removed shared segment for {symbol}")
    
    async def _notify_agents(self, symbol: str, data: Dict[str, Any]) -> None:
        """Notify subscribed agents about data update"""
        subscribed_agents = self.agent_subscriptions.get(symbol, set())
        
        for agent_name in subscribed_agents:
            if agent_name not in self.registered_agents:
                continue
            
            agent_info = self.registered_agents[agent_name]
            
            # Update last sync time
            agent_info["last_sync"] = datetime.now()
            
            # Call callback if provided
            if agent_info["callback"]:
                try:
                    callback = agent_info["callback"]
                    if asyncio.iscoroutinefunction(callback):
                        await callback(symbol, data)
                    else:
                        callback(symbol, data)
                except Exception as e:
                    logger.error(f"Error in agent callback for {agent_name}: {str(e)}")
            
            # Add to sync queue
            if agent_name in self.sync_queues:
                try:
                    await self.sync_queues[agent_name].put({
                        "symbol": symbol,
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    })
                except asyncio.QueueFull:
                    logger.warning(f"Sync queue full for agent {agent_name}")
    
    def _serialize_data(self, data: Dict[str, Any]) -> bytes:
        """Serialize data for shared memory"""
        try:
            # Use pickle for efficient serialization
            return pickle.dumps(data)
        except Exception:
            # Fallback to JSON
            return json.dumps(data).encode('utf-8')
    
    def _deserialize_data(self, data: bytes) -> Dict[str, Any]:
        """Deserialize data from shared memory"""
        try:
            # Try pickle first
            return pickle.loads(data)
        except Exception:
            # Fallback to JSON
            return json.loads(data.decode('utf-8'))
    
    async def get_agent_sync_queue(self, agent_name: str) -> Optional[asyncio.Queue]:
        """Get sync queue for an agent"""
        return self.sync_queues.get(agent_name)
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        agent_stats = {}
        
        for agent_name, info in self.registered_agents.items():
            agent_stats[agent_name] = {
                "registered_at": info["registered_at"].isoformat(),
                "last_sync": info["last_sync"].isoformat() if info["last_sync"] else None,
                "subscriptions": len([s for s, agents in self.agent_subscriptions.items() 
                                    if agent_name in agents])
            }
        
        return {
            **self.stats,
            "registered_agents": len(self.registered_agents),
            "agent_stats": agent_stats,
            "sync_success_rate": (
                (self.stats["total_syncs"] - self.stats["sync_errors"]) / 
                self.stats["total_syncs"] * 100
                if self.stats["total_syncs"] > 0 else 100
            )
        }
    
    async def cleanup(self) -> None:
        """Clean up all shared memory segments"""
        logger.info("Cleaning up shared memory segments")
        
        for symbol in list(self.segments.keys()):
            await self._remove_segment(symbol)
        
        logger.info("Cleanup completed")