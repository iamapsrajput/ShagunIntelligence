import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""

    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BACKGROUND = 1


@dataclass
class Task:
    """Represents a task to be executed by an agent"""

    id: str
    agent_type: Any  # AgentType enum
    method: str
    parameters: dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout: int = 300  # 5 minutes default
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: datetime | None = None
    completed_at: datetime | None = None
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    retries: int = 0
    max_retries: int = 3


class TaskDelegator:
    """Manages task delegation and execution across multiple agents"""

    def __init__(self, agents: dict[Any, Any], max_workers: int = 5):
        self.agents = agents
        self.max_workers = max_workers

        # Task management
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: dict[str, Task] = {}
        self.completed_tasks: list[Task] = []

        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_counter = 0
        self.is_running = True

        # Agent workload tracking
        self.agent_workload: dict[Any, int] = {agent: 0 for agent in agents}
        self.agent_performance: dict[Any, dict[str, Any]] = {
            agent: {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "average_execution_time": 0,
                "success_rate": 100.0,
            }
            for agent in agents
        }

        # Start task processor
        self.processor_thread = threading.Thread(
            target=self._process_tasks, daemon=True
        )
        self.processor_thread.start()

    def create_task(
        self,
        agent_type: Any,
        method: str,
        parameters: dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        timeout: int = 300,
    ) -> Task:
        """Create a new task for delegation"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        task = Task(
            id=task_id,
            agent_type=agent_type,
            method=method,
            parameters=parameters,
            priority=priority,
            timeout=timeout,
        )

        return task

    def submit_task(self, task: Task) -> str:
        """Submit a task for execution"""
        # Add to priority queue (negative priority for max heap behavior)
        self.task_queue.put((-task.priority.value, task.created_at, task))
        logger.info(f"Task {task.id} submitted for {task.agent_type}")

        return task.id

    def execute_task(self, task: Task) -> dict[str, Any]:
        """Execute a single task synchronously"""
        return self._execute_task_internal(task)

    def execute_parallel_tasks(self, tasks: list[Task]) -> list[dict[str, Any]]:
        """Execute multiple tasks in parallel"""
        futures = []
        results = []

        # Submit all tasks
        for task in tasks:
            future = self.executor.submit(self._execute_task_internal, task)
            futures.append((future, task))

        # Collect results
        for future, task in futures:
            try:
                result = future.result(timeout=task.timeout)
                results.append(result)
            except Exception as e:
                logger.error(f"Task {task.id} failed: {str(e)}")
                results.append(
                    {
                        "task_id": task.id,
                        "status": "failed",
                        "error": str(e),
                        "agent_type": task.agent_type,
                    }
                )

        return results

    def _process_tasks(self):
        """Background task processor"""
        while self.is_running:
            try:
                # Get task from queue (wait up to 1 second)
                try:
                    _, _, task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Check if agent is available
                if self._is_agent_available(task.agent_type):
                    # Submit for execution
                    future = self.executor.submit(self._execute_task_internal, task)

                    # Track active task
                    self.active_tasks[task.id] = task

                    # Handle completion asynchronously
                    future.add_done_callback(
                        lambda f, t=task: self._handle_task_completion(f, t)
                    )
                else:
                    # Re-queue task if agent is busy
                    self.task_queue.put((-task.priority.value, task.created_at, task))
                    time.sleep(0.1)  # Brief delay before retry

            except Exception as e:
                logger.error(f"Error in task processor: {str(e)}")

    def _execute_task_internal(self, task: Task) -> dict[str, Any]:
        """Internal method to execute a task"""
        try:
            # Update task status
            task.status = TaskStatus.ASSIGNED
            task.assigned_at = datetime.now()

            # Get agent
            agent = self.agents.get(task.agent_type)
            if not agent:
                raise ValueError(f"Agent {task.agent_type} not found")

            # Update workload
            self.agent_workload[task.agent_type] += 1

            # Update task status
            task.status = TaskStatus.IN_PROGRESS

            # Execute task method
            start_time = time.time()

            # Get method from agent
            method = getattr(agent, task.method, None)
            if not method:
                raise AttributeError(
                    f"Method {task.method} not found on agent {task.agent_type}"
                )

            # Execute with timeout
            result = self._execute_with_timeout(method, task.parameters, task.timeout)

            execution_time = time.time() - start_time

            # Update task
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result

            # Update performance metrics
            self._update_agent_performance(task.agent_type, True, execution_time)

            return {
                "task_id": task.id,
                "status": "success",
                "agent_type": task.agent_type,
                "data": result,
                "execution_time": execution_time,
            }

        except Exception as e:
            logger.error(f"Task {task.id} execution failed: {str(e)}")

            # Update task
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()

            # Update performance metrics
            self._update_agent_performance(task.agent_type, False, 0)

            # Check if we should retry
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                self.task_queue.put((-task.priority.value, task.created_at, task))
                logger.info(f"Retrying task {task.id} (attempt {task.retries})")

            return {
                "task_id": task.id,
                "status": "failed",
                "agent_type": task.agent_type,
                "error": str(e),
            }

        finally:
            # Update workload
            self.agent_workload[task.agent_type] -= 1

    def _execute_with_timeout(
        self, method: Callable, parameters: dict[str, Any], timeout: int
    ) -> Any:
        """Execute a method with timeout"""
        future = self.executor.submit(method, **parameters)

        try:
            return future.result(timeout=timeout)
        except Exception as e:
            future.cancel()
            raise TimeoutError(
                f"Task execution timed out after {timeout} seconds"
            ) from e

    def _handle_task_completion(self, future: Future, task: Task):
        """Handle task completion callback"""
        # Remove from active tasks
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]

        # Add to completed tasks
        self.completed_tasks.append(task)

        # Keep only recent completed tasks
        if len(self.completed_tasks) > 1000:
            self.completed_tasks = self.completed_tasks[-1000:]

    def _is_agent_available(self, agent_type: Any) -> bool:
        """Check if an agent is available for new tasks"""
        # Simple workload-based availability
        max_concurrent_tasks = 3
        return self.agent_workload.get(agent_type, 0) < max_concurrent_tasks

    def _update_agent_performance(
        self, agent_type: Any, success: bool, execution_time: float
    ):
        """Update agent performance metrics"""
        metrics = self.agent_performance[agent_type]

        if success:
            metrics["tasks_completed"] += 1

            # Update average execution time
            total_tasks = metrics["tasks_completed"]
            current_avg = metrics["average_execution_time"]
            metrics["average_execution_time"] = (
                current_avg * (total_tasks - 1) + execution_time
            ) / total_tasks
        else:
            metrics["tasks_failed"] += 1

        # Update success rate
        total = metrics["tasks_completed"] + metrics["tasks_failed"]
        if total > 0:
            metrics["success_rate"] = (metrics["tasks_completed"] / total) * 100

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get status of a specific task"""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task.status.value,
                "agent_type": task.agent_type,
                "created_at": task.created_at,
                "assigned_at": task.assigned_at,
            }

        # Check completed tasks
        for task in self.completed_tasks:
            if task.id == task_id:
                return {
                    "task_id": task_id,
                    "status": task.status.value,
                    "agent_type": task.agent_type,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at,
                    "result": task.result,
                    "error": task.error,
                }

        return None

    def get_agent_status(self) -> dict[str, Any]:
        """Get status of all agents"""
        status = {}

        for agent_type, agent in self.agents.items():
            status[agent_type.value] = {
                "available": agent is not None,
                "current_workload": self.agent_workload.get(agent_type, 0),
                "performance": self.agent_performance.get(agent_type, {}),
            }

        return status

    def get_workload_summary(self) -> dict[str, Any]:
        """Get overall workload summary"""
        return {
            "pending_tasks": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "agent_workloads": dict(self.agent_workload),
            "total_capacity": self.max_workers,
        }

    def reassign_task(self, task_id: str, new_agent_type: Any) -> bool:
        """Reassign a task to a different agent"""
        # Find task in active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]

            # Cancel current execution if in progress
            if task.status == TaskStatus.IN_PROGRESS:
                logger.warning(f"Cannot reassign task {task_id} while in progress")
                return False

            # Update agent type
            task.agent_type = new_agent_type
            task.status = TaskStatus.PENDING

            # Re-queue task
            self.task_queue.put((-task.priority.value, task.created_at, task))

            logger.info(f"Task {task_id} reassigned to {new_agent_type}")
            return True

        return False

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or active task"""
        # Check if task is active
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]

            if task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
                task.status = TaskStatus.CANCELLED
                del self.active_tasks[task_id]
                self.completed_tasks.append(task)

                logger.info(f"Task {task_id} cancelled")
                return True
            else:
                logger.warning(
                    f"Cannot cancel task {task_id} with status {task.status}"
                )
                return False

        return False

    def shutdown(self):
        """Shutdown the task delegator"""
        logger.info("Shutting down task delegator")

        self.is_running = False

        # Wait for processor thread
        if self.processor_thread:
            self.processor_thread.join()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("Task delegator shutdown complete")
