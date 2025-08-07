import json
import logging
import queue
import threading
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the system"""

    ANALYSIS_UPDATE = "analysis_update"
    TRADE_SIGNAL = "trade_signal"
    RISK_ALERT = "risk_alert"
    DECISION_MADE = "decision_made"
    EXECUTION_UPDATE = "execution_update"
    PERFORMANCE_UPDATE = "performance_update"
    SYSTEM_STATUS = "system_status"
    ERROR_REPORT = "error_report"
    ADAPTATION_NOTICE = "adaptation_notice"


class MessagePriority(Enum):
    """Message priority levels"""

    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1


@dataclass
class Message:
    """Represents a message in the system"""

    id: str
    type: MessageType
    priority: MessagePriority
    source: str  # Agent or component name
    target: str | None  # Specific target or None for broadcast
    content: dict[str, Any]
    timestamp: datetime
    requires_response: bool = False
    correlation_id: str | None = None  # For tracking related messages


class CommunicationHub:
    """Central hub for agent communication and coordination"""

    def __init__(self):
        # Message queues
        self.message_queue = queue.PriorityQueue()
        self.message_history = deque(maxlen=10000)

        # Subscribers
        self.subscribers: dict[MessageType, list[Callable]] = defaultdict(list)
        self.agent_subscriptions: dict[str, list[MessageType]] = defaultdict(list)

        # Message tracking
        self.message_counter = 0
        self.pending_responses: dict[str, Message] = {}

        # Communication channels
        self.channels: dict[str, queue.Queue] = {}

        # Statistics
        self.message_stats = defaultdict(
            lambda: {"sent": 0, "received": 0, "processing_time": 0}
        )

        # Processing thread
        self.is_running = True
        self.processor_thread = threading.Thread(
            target=self._process_messages, daemon=True
        )
        self.processor_thread.start()

        # Event log
        self.event_log = []

    def send_message(
        self,
        message_type: MessageType,
        source: str,
        content: dict[str, Any],
        priority: MessagePriority = MessagePriority.MEDIUM,
        target: str | None = None,
        requires_response: bool = False,
    ) -> str:
        """Send a message through the hub"""
        self.message_counter += 1
        message_id = (
            f"msg_{self.message_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

        message = Message(
            id=message_id,
            type=message_type,
            priority=priority,
            source=source,
            target=target,
            content=content,
            timestamp=datetime.now(),
            requires_response=requires_response,
        )

        # Add to queue (negative priority for max heap behavior)
        self.message_queue.put((-priority.value, message.timestamp, message))

        # Track if response required
        if requires_response:
            self.pending_responses[message_id] = message

        # Update statistics
        self.message_stats[source]["sent"] += 1

        logger.debug(
            f"Message {message_id} sent from {source} - Type: {message_type.value}"
        )

        return message_id

    def subscribe(
        self,
        message_type: MessageType,
        callback: Callable,
        agent_name: str | None = None,
    ) -> None:
        """Subscribe to a message type"""
        self.subscribers[message_type].append(callback)

        if agent_name:
            self.agent_subscriptions[agent_name].append(message_type)

        logger.info(
            f"Subscribed to {message_type.value} messages"
            f"{f' for agent {agent_name}' if agent_name else ''}"
        )

    def create_channel(self, channel_name: str) -> queue.Queue:
        """Create a dedicated communication channel"""
        if channel_name not in self.channels:
            self.channels[channel_name] = queue.Queue()
            logger.info(f"Created communication channel: {channel_name}")

        return self.channels[channel_name]

    def _process_messages(self) -> None:
        """Background message processor"""
        while self.is_running:
            try:
                # Get message from queue (wait up to 1 second)
                try:
                    _, _, message = self.message_queue.get(timeout=1)
                except queue.Empty:
                    continue

                start_time = datetime.now()

                # Process message
                self._deliver_message(message)

                # Record in history
                self.message_history.append(message)

                # Update processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                self.message_stats[message.source]["processing_time"] += processing_time

            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")

    def _deliver_message(self, message: Message) -> None:
        """Deliver message to appropriate recipients"""
        # If targeted message, deliver to specific channel
        if message.target and message.target in self.channels:
            self.channels[message.target].put(message)
            self.message_stats[message.target]["received"] += 1

        # Deliver to subscribers
        subscribers = self.subscribers.get(message.type, [])
        for callback in subscribers:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {str(e)}")

    def broadcast_decision(self, decision: dict[str, Any]) -> None:
        """Broadcast a trading decision to all agents"""
        self.send_message(
            MessageType.DECISION_MADE,
            source="coordinator",
            content={"decision": decision, "timestamp": datetime.now().isoformat()},
            priority=MessagePriority.HIGH,
        )

        # Log important event
        self.event_log.append(
            {
                "event": "decision_broadcast",
                "timestamp": datetime.now(),
                "details": {
                    "symbol": decision.get("symbol"),
                    "action": decision.get("action"),
                    "confidence": decision.get("confidence"),
                },
            }
        )

    def broadcast_execution(self, execution_result: dict[str, Any]) -> None:
        """Broadcast trade execution result"""
        self.send_message(
            MessageType.EXECUTION_UPDATE,
            source="trade_executor",
            content={
                "execution": execution_result,
                "timestamp": datetime.now().isoformat(),
            },
            priority=MessagePriority.HIGH,
        )

        # Log execution event
        self.event_log.append(
            {
                "event": "execution_broadcast",
                "timestamp": datetime.now(),
                "details": {
                    "status": execution_result.get("execution", {}).get("status"),
                    "symbol": execution_result.get("decision", {}).get("symbol"),
                },
            }
        )

    def send_risk_alert(
        self, alert_type: str, severity: str, details: dict[str, Any]
    ) -> None:
        """Send a risk alert to the system"""
        priority_map = {
            "critical": MessagePriority.CRITICAL,
            "high": MessagePriority.HIGH,
            "medium": MessagePriority.MEDIUM,
            "low": MessagePriority.LOW,
        }

        self.send_message(
            MessageType.RISK_ALERT,
            source="risk_manager",
            content={
                "alert_type": alert_type,
                "severity": severity,
                "details": details,
                "timestamp": datetime.now().isoformat(),
            },
            priority=priority_map.get(severity, MessagePriority.MEDIUM),
        )

        logger.warning(f"Risk alert sent: {alert_type} - Severity: {severity}")

    def request_analysis(
        self, agent: str, symbols: list[str], analysis_type: str
    ) -> str:
        """Request analysis from a specific agent"""
        return self.send_message(
            MessageType.ANALYSIS_UPDATE,
            source="coordinator",
            target=agent,
            content={
                "request_type": "analysis",
                "symbols": symbols,
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat(),
            },
            priority=MessagePriority.MEDIUM,
            requires_response=True,
        )

    def send_response(
        self, original_message_id: str, response_content: dict[str, Any], source: str
    ) -> None:
        """Send a response to a message that required one"""
        if original_message_id in self.pending_responses:
            original = self.pending_responses[original_message_id]

            # Send response
            self.send_message(
                original.type,
                source=source,
                target=original.source,
                content={
                    "response_to": original_message_id,
                    "original_request": original.content,
                    "response": response_content,
                },
                priority=MessagePriority.HIGH,
            )

            # Remove from pending
            del self.pending_responses[original_message_id]

    def get_message_statistics(self) -> dict[str, Any]:
        """Get communication statistics"""
        total_messages = sum(stats["sent"] for stats in self.message_stats.values())

        # Message type distribution
        type_distribution = defaultdict(int)
        for msg in list(self.message_history)[-1000:]:  # Last 1000 messages
            type_distribution[msg.type.value] += 1

        # Average processing time
        total_processing_time = sum(
            stats["processing_time"] for stats in self.message_stats.values()
        )
        avg_processing_time = (
            total_processing_time / total_messages if total_messages > 0 else 0
        )

        return {
            "total_messages": total_messages,
            "pending_responses": len(self.pending_responses),
            "active_channels": len(self.channels),
            "subscribers": {
                msg_type.value: len(subs) for msg_type, subs in self.subscribers.items()
            },
            "message_type_distribution": dict(type_distribution),
            "average_processing_time": avg_processing_time,
            "agent_statistics": dict(self.message_stats),
        }

    def get_recent_messages(
        self, message_type: MessageType | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get recent messages, optionally filtered by type"""
        messages = list(self.message_history)

        if message_type:
            messages = [m for m in messages if m.type == message_type]

        # Convert to dictionaries
        return [
            {
                "id": msg.id,
                "type": msg.type.value,
                "priority": msg.priority.value,
                "source": msg.source,
                "target": msg.target,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "requires_response": msg.requires_response,
            }
            for msg in messages[-limit:]
        ]

    def get_event_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent important events"""
        return self.event_log[-limit:]

    def broadcast_system_status(self, status: dict[str, Any]) -> None:
        """Broadcast system status update"""
        self.send_message(
            MessageType.SYSTEM_STATUS,
            source="coordinator",
            content={"status": status, "timestamp": datetime.now().isoformat()},
            priority=MessagePriority.LOW,
        )

    def broadcast_adaptation(self, adaptation_details: dict[str, Any]) -> None:
        """Broadcast strategy adaptation notice"""
        self.send_message(
            MessageType.ADAPTATION_NOTICE,
            source="learning_manager",
            content={
                "adaptation": adaptation_details,
                "timestamp": datetime.now().isoformat(),
            },
            priority=MessagePriority.MEDIUM,
        )

        # Log adaptation event
        self.event_log.append(
            {
                "event": "strategy_adaptation",
                "timestamp": datetime.now(),
                "details": adaptation_details,
            }
        )

    def export_communication_log(self, filepath: str) -> bool:
        """Export communication log for analysis"""
        try:
            log_data = {
                "messages": [
                    {
                        "id": msg.id,
                        "type": msg.type.value,
                        "priority": msg.priority.value,
                        "source": msg.source,
                        "target": msg.target,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                    }
                    for msg in list(self.message_history)[-5000:]  # Last 5000 messages
                ],
                "events": self.event_log,
                "statistics": self.get_message_statistics(),
                "export_timestamp": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(log_data, f, indent=2)

            logger.info(f"Communication log exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting communication log: {str(e)}")
            return False

    def shutdown(self) -> None:
        """Shutdown the communication hub"""
        logger.info("Shutting down communication hub")
        self.is_running = False

        if self.processor_thread:
            self.processor_thread.join()

        logger.info("Communication hub shutdown complete")
