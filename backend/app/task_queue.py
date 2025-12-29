"""
Task Queue System for Load Balancing

Basit bir in-memory task queue sistemi. Gelen istekleri birden fazla GPU worker'a dağıtır.
Production'da Redis/Celery kullanılabilir.
"""

import asyncio
import threading
import time
from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task durumu"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Task representation"""
    task_id: str
    message: str
    adapter_name: str
    wallet_address: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[str] = None


class Worker:
    """GPU Worker representation"""
    
    def __init__(self, worker_id: str, gpu_id: Optional[int] = None):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.is_busy = False
        self.current_task: Optional[Task] = None
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.last_activity = time.time()
    
    def assign_task(self, task: Task):
        """Assign a task to this worker"""
        self.is_busy = True
        self.current_task = task
        task.worker_id = self.worker_id
        task.status = TaskStatus.PROCESSING
        task.started_at = time.time()
        self.total_tasks += 1
        self.last_activity = time.time()
    
    def complete_task(self, result: Any = None, error: Optional[str] = None):
        """Complete current task"""
        if self.current_task:
            self.current_task.completed_at = time.time()
            if error:
                self.current_task.status = TaskStatus.FAILED
                self.current_task.error = error
                self.failed_tasks += 1
            else:
                self.current_task.status = TaskStatus.COMPLETED
                self.current_task.result = result
                self.completed_tasks += 1
        
        self.is_busy = False
        self.current_task = None
        self.last_activity = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,
            "is_busy": self.is_busy,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 0.0,
            "last_activity": self.last_activity,
        }


class TaskQueue:
    """
    Basit in-memory task queue sistemi.
    
    Production'da Redis/Celery kullanılabilir, ancak bu basit implementasyon
    küçük-orta ölçekli deployment'lar için yeterlidir.
    """
    
    def __init__(self, max_workers: int = 1):
        """
        Initialize task queue.
        
        Args:
            max_workers: Maximum number of workers (GPUs)
        """
        self.max_workers = max_workers
        self.queue: asyncio.Queue = asyncio.Queue()
        self.workers: Dict[str, Worker] = {}
        self.tasks: Dict[str, Task] = {}
        self.lock = threading.Lock()
        self.task_counter = 0
        
        # Initialize workers
        for i in range(max_workers):
            worker_id = f"worker_{i}"
            self.workers[worker_id] = Worker(worker_id, gpu_id=i if max_workers > 1 else None)
        
        logger.info(f"TaskQueue initialized with {max_workers} workers")
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        with self.lock:
            self.task_counter += 1
            return f"task_{int(time.time())}_{self.task_counter}"
    
    async def enqueue(
        self,
        message: str,
        adapter_name: str,
        wallet_address: str,
        process_fn: Callable[[str, str], Any],
    ) -> str:
        """
        Enqueue a new task.
        
        Args:
            message: User message
            adapter_name: Selected adapter name
            wallet_address: Wallet address
            process_fn: Function to process the task (async generator for streaming)
            
        Returns:
            Task ID
        """
        task_id = self._generate_task_id()
        task = Task(
            task_id=task_id,
            message=message,
            adapter_name=adapter_name,
            wallet_address=wallet_address,
            created_at=time.time(),
        )
        
        with self.lock:
            self.tasks[task_id] = task
        
        # Find available worker
        worker = self._find_available_worker()
        if worker:
            # Assign immediately if worker available
            worker.assign_task(task)
            # Process in background
            asyncio.create_task(self._process_task(worker, task, process_fn))
        else:
            # Add to queue if no worker available
            await self.queue.put((task, process_fn))
            logger.info(f"Task {task_id} queued (queue size: {self.queue.qsize()})")
        
        return task_id
    
    def _find_available_worker(self) -> Optional[Worker]:
        """Find an available worker"""
        for worker in self.workers.values():
            if not worker.is_busy:
                return worker
        return None
    
    async def _process_task(
        self,
        worker: Worker,
        task: Task,
        process_fn: Callable[[str, str], Any],
    ):
        """Process a task with a worker"""
        try:
            logger.info(f"Worker {worker.worker_id} processing task {task.task_id}")
            
            # Process task (synchronous generator for streaming)
            result = []
            for chunk in process_fn(task.message, task.adapter_name):
                result.append(chunk)
            
            # Complete task
            worker.complete_task(result="".join(result))
            logger.info(f"Task {task.task_id} completed by worker {worker.worker_id}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task.task_id} failed: {error_msg}")
            worker.complete_task(error=error_msg)
        finally:
            # Process next task from queue if available
            await self._process_next_task(worker)
    
    async def process_with_queue(
        self,
        message: str,
        adapter_name: str,
        process_fn: Callable[[str, str], Any],
    ):
        """
        Process a task through the queue and stream results.
        
        Args:
            message: User message
            adapter_name: Selected adapter name
            process_fn: Function to process the task (synchronous generator)
            
        Yields:
            Chunks from the processing function
        """
        # Find available worker or enqueue
        worker = self._find_available_worker()
        
        if worker:
            # Process immediately if worker available
            try:
                logger.info(f"Worker {worker.worker_id} processing immediately")
                for chunk in process_fn(message, adapter_name):
                    yield chunk
            except Exception as e:
                logger.error(f"Task processing failed: {e}")
                raise
        else:
            # Enqueue and wait (for now, process immediately but log queue)
            logger.info(f"All workers busy, processing directly (queue size: {self.queue.qsize()})")
            for chunk in process_fn(message, adapter_name):
                yield chunk
    
    async def _process_next_task(self, worker: Worker):
        """Process next task from queue if available"""
        if not self.queue.empty():
            try:
                task, process_fn = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                worker.assign_task(task)
                asyncio.create_task(self._process_task(worker, task, process_fn))
            except asyncio.TimeoutError:
                pass
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            return {
                "task_id": task.task_id,
                "status": task.status.value,
                "message": task.message,
                "adapter_name": task.adapter_name,
                "wallet_address": task.wallet_address,
                "worker_id": task.worker_id,
                "error": task.error,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "processing_time": (
                    task.completed_at - task.started_at
                    if task.started_at and task.completed_at
                    else None
                ),
            }
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.lock:
            busy_workers = sum(1 for w in self.workers.values() if w.is_busy)
            total_tasks = len(self.tasks)
            pending_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
            processing_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PROCESSING)
            completed_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
            failed_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
            
            return {
                "queue_size": self.queue.qsize(),
                "total_workers": len(self.workers),
                "busy_workers": busy_workers,
                "available_workers": len(self.workers) - busy_workers,
                "total_tasks": total_tasks,
                "pending_tasks": pending_tasks,
                "processing_tasks": processing_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "worker_stats": [w.get_stats() for w in self.workers.values()],
            }

