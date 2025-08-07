import os
from datetime import datetime

import psutil
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    system_info: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system information"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        system_info={
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        },
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes/Docker"""
    return {"status": "ready", "timestamp": datetime.utcnow()}
