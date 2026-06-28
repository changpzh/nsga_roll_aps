"""
排程过程核心状态数据结构
所有解码算法共用的通用状态和结果定义
示例业务背景：机械加工车间，工艺类型1=车削、2=铣削、3=磨削
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
from datetime import datetime

from .base_types import MachineId, WorkerId, JobId, OperationId


@dataclass
class SchedulingTrackers:
    """
    解码过程全局状态跟踪器
    所有累计状态集中管理，避免函数间传递零散变量
    """
    # ============================== 机器状态 ==============================
    machine_last_end_time_dict: Dict[MachineId, datetime] = field(default_factory=dict)
    machine_previous_technology_type_dict: Dict[MachineId, int] = field(default_factory=dict)
    machine_workloads_dict: Dict[MachineId, List[float]] = field(default_factory=lambda: defaultdict(list))

    # ============================== 工人状态 ==============================
    worker_task_intervals_dict: Dict[WorkerId, List[Tuple[datetime, datetime]]] = field(default_factory=lambda: defaultdict(list))
    worker_task_ends_heap_dict: Dict[WorkerId, List[datetime]] = field(default_factory=lambda: defaultdict(list))
    worker_workloads_dict: Dict[WorkerId, List[float]] = field(default_factory=lambda: defaultdict(list))

    # ============================== 工件状态 ==============================
    job_last_operation_end_time_dict: Dict[JobId, datetime] = field(default_factory=dict)
    job_op_finish_time_dict: Dict[OperationId, datetime] = field(default_factory=dict)

    # ============================== 全局累计指标 ==============================
    total_changeover_time: float = 0.0
    total_wip_wait_time: float = 0.0
    total_operation_count: int = 0
    worker_switch_count: int = 0
    machine_total_available_hour: Dict[MachineId, float] = field(default_factory=dict)
    machine_total_process_hour: Dict[MachineId, float] = field(default_factory=dict)
    overdue_job_count: int = 0
    total_overdue_penalty: float = 0.0


@dataclass
class OperationSchedulingResult:
    """
    单个操作的调度结果
    """
    operation_id: OperationId = OperationId(0)
    job_id: JobId = JobId("")
    operation_index_in_job: int = -1
    machine_id: MachineId = MachineId(-1)
    worker_id: WorkerId = WorkerId(-1)
    start_time: datetime = field(default_factory=datetime.min)
    end_time: datetime = field(default_factory=datetime.min)
    actual_processing_time: float = 0.0
    technology_type: int = -1
    is_frozen: bool = False
    is_manual_locked: bool = False
    operation_metadata: Optional[Any] = None