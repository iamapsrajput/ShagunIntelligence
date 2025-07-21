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

from .agent import CoordinatorAgent, AgentType, TradingOpportunity
from .decision_fusion_engine import DecisionFusionEngine, SignalStrength, FusedSignal
from .task_delegator import TaskDelegator, Task, TaskStatus, TaskPriority
from .trade_approval_workflow import (
    TradeApprovalWorkflow, 
    ApprovalStatus, 
    RejectionReason,
    ApprovalRule
)
from .learning_manager import LearningManager, LearningEvent, Adaptation, AdaptationType
from .performance_monitor import PerformanceMonitor, TradeMetrics
from .communication_hub import (
    CommunicationHub,
    Message,
    MessageType,
    MessagePriority
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
    "MessagePriority"
]