"""
Coordinator Agent for Shagun Intelligence

This module provides the master orchestration agent that coordinates all other agents
in the trading system. It includes:
- Decision fusion from multiple agents
- Task delegation and management
- Trade approval workflow
- Learning and adaptation mechanisms
- Performance monitoring
- Inter-agent communication
"""

from .agent import AgentType, CoordinatorAgent, TradingOpportunity
from .communication_hub import CommunicationHub, Message, MessagePriority, MessageType
from .decision_fusion_engine import DecisionFusionEngine, FusedSignal, SignalStrength
from .learning_manager import Adaptation, AdaptationType, LearningEvent, LearningManager
from .performance_monitor import PerformanceMonitor, TradeMetrics
from .task_delegator import Task, TaskDelegator, TaskPriority, TaskStatus
from .trade_approval_workflow import (
    ApprovalRule,
    ApprovalStatus,
    RejectionReason,
    TradeApprovalWorkflow,
)

__all__ = [
    # Main agent
    "CoordinatorAgent",
    "AgentType",
    "TradingOpportunity",
    # Decision fusion
    "DecisionFusionEngine",
    "SignalStrength",
    "FusedSignal",
    # Task management
    "TaskDelegator",
    "Task",
    "TaskStatus",
    "TaskPriority",
    # Trade approval
    "TradeApprovalWorkflow",
    "ApprovalStatus",
    "RejectionReason",
    "ApprovalRule",
    # Learning
    "LearningManager",
    "LearningEvent",
    "Adaptation",
    "AdaptationType",
    # Performance
    "PerformanceMonitor",
    "TradeMetrics",
    # Communication
    "CommunicationHub",
    "Message",
    "MessageType",
    "MessagePriority",
]
