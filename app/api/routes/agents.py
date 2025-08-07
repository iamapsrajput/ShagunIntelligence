from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from app.core.auth import get_current_user
from app.models.user import User
from app.services.websocket_manager import websocket_broadcaster

router = APIRouter()


class AgentConfig(BaseModel):
    enabled: bool = True
    weight: float = Field(0.25, ge=0, le=1)
    parameters: dict[str, Any] = {}


class AgentStatus(BaseModel):
    agent_type: str
    status: str  # active, idle, error
    enabled: bool
    last_activity: datetime | None
    current_task: str | None
    performance_metrics: dict[str, Any]


class AgentAnalysis(BaseModel):
    agent_type: str
    analysis: dict[str, Any]
    confidence: float
    timestamp: datetime


@router.get("/status", response_model=dict[str, AgentStatus])
async def get_agents_status(
    request: Request, current_user: User = Depends(get_current_user)
):
    """Get status of all agents"""
    try:
        crew_manager = request.app.state.crew_manager
        agents_status = await crew_manager.get_all_agents_status()

        return {
            agent_type: AgentStatus(
                agent_type=agent_type,
                status=status.get("status", "unknown"),
                enabled=status.get("enabled", False),
                last_activity=status.get("last_activity"),
                current_task=status.get("current_task"),
                performance_metrics=status.get("performance_metrics", {}),
            )
            for agent_type, status in agents_status.items()
        }
    except Exception as e:
        logger.error(f"Error fetching agents status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_type}/status", response_model=AgentStatus)
async def get_agent_status(
    agent_type: str, request: Request, current_user: User = Depends(get_current_user)
):
    """Get status of a specific agent"""
    try:
        crew_manager = request.app.state.crew_manager
        status = await crew_manager.get_agent_status(agent_type)

        if not status:
            raise HTTPException(status_code=404, detail=f"Agent {agent_type} not found")

        return AgentStatus(
            agent_type=agent_type,
            status=status.get("status", "unknown"),
            enabled=status.get("enabled", False),
            last_activity=status.get("last_activity"),
            current_task=status.get("current_task"),
            performance_metrics=status.get("performance_metrics", {}),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching agent {agent_type} status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_type}/analysis", response_model=AgentAnalysis)
async def get_agent_analysis(
    agent_type: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    symbol: str | None = None,
):
    """Get latest analysis from a specific agent"""
    try:
        crew_manager = request.app.state.crew_manager

        # Get agent analysis
        if agent_type == "market":
            analysis = await crew_manager.get_market_analysis(symbol)
        elif agent_type == "technical":
            analysis = await crew_manager.get_technical_analysis(symbol)
        elif agent_type == "sentiment":
            analysis = await crew_manager.get_sentiment_analysis(symbol)
        elif agent_type == "risk":
            analysis = await crew_manager.get_risk_analysis(symbol)
        elif agent_type == "coordinator":
            analysis = await crew_manager.get_coordinator_decision(symbol)
        else:
            raise HTTPException(status_code=404, detail=f"Agent {agent_type} not found")

        return AgentAnalysis(
            agent_type=agent_type,
            analysis=analysis.get("analysis", {}),
            confidence=analysis.get("confidence", 0),
            timestamp=analysis.get("timestamp", datetime.utcnow()),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching agent {agent_type} analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{agent_type}/config")
async def update_agent_config(
    agent_type: str,
    config: AgentConfig,
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Update agent configuration"""
    try:
        crew_manager = request.app.state.crew_manager

        # Update agent configuration
        result = await crew_manager.update_agent_config(
            agent_type=agent_type,
            enabled=config.enabled,
            weight=config.weight,
            parameters=config.parameters,
        )

        if not result:
            raise HTTPException(status_code=404, detail=f"Agent {agent_type} not found")

        # Broadcast configuration update
        background_tasks.add_task(
            websocket_broadcaster.broadcast_agent_activity,
            {
                "agentId": f"agent_{agent_type}",
                "agentType": agent_type,
                "action": "config_update",
                "confidence": 1.0,
                "analysis": {
                    "enabled": config.enabled,
                    "weight": config.weight,
                    "parameters": config.parameters,
                },
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        logger.info(
            f"Agent {agent_type} configuration updated by {current_user.username}"
        )

        return {"agent_type": agent_type, "status": "updated", "config": config.dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent {agent_type} config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_type}/restart")
async def restart_agent(
    agent_type: str, request: Request, current_user: User = Depends(get_current_user)
):
    """Restart a specific agent"""
    try:
        crew_manager = request.app.state.crew_manager

        # Restart agent
        result = await crew_manager.restart_agent(agent_type)

        if not result:
            raise HTTPException(status_code=404, detail=f"Agent {agent_type} not found")

        logger.info(f"Agent {agent_type} restarted by {current_user.username}")

        return {
            "agent_type": agent_type,
            "status": "restarted",
            "message": f"Agent {agent_type} has been restarted successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting agent {agent_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/metrics")
async def get_agents_performance(
    request: Request,
    current_user: User = Depends(get_current_user),
    period: str = "1d",  # 1h, 1d, 1w, 1m
):
    """Get performance metrics for all agents"""
    try:
        crew_manager = request.app.state.crew_manager

        # Get performance metrics
        metrics = await crew_manager.get_agents_performance_metrics(period)

        return {
            "period": period,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error fetching agents performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activity/recent")
async def get_recent_activity(
    request: Request,
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    agent_type: str | None = None,
):
    """Get recent agent activities"""
    try:
        crew_manager = request.app.state.crew_manager

        # Get recent activities
        activities = await crew_manager.get_recent_agent_activities(
            limit=limit, agent_type=agent_type
        )

        return {
            "activities": activities,
            "count": len(activities),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error fetching recent activities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crew/execute")
async def execute_crew_task(
    task: dict[str, Any],
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Execute a custom crew task"""
    try:
        crew_manager = request.app.state.crew_manager

        # Validate task
        if "type" not in task:
            raise HTTPException(status_code=400, detail="Task type is required")

        # Execute task
        result = await crew_manager.execute_custom_task(task)

        # Log task execution
        logger.info(
            f"Custom crew task executed by {current_user.username}: {task['type']}"
        )

        return {
            "task_id": result.get("task_id"),
            "status": "executed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing crew task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decisions/history")
async def get_decision_history(
    request: Request,
    current_user: User = Depends(get_current_user),
    symbol: str | None = None,
    limit: int = 100,
):
    """Get agent decision history"""
    try:
        crew_manager = request.app.state.crew_manager

        # Get decision history
        decisions = await crew_manager.get_decision_history(symbol=symbol, limit=limit)

        return {
            "decisions": decisions,
            "count": len(decisions),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error fetching decision history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
