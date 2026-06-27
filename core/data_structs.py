# core/data_structs.py
from dataclasses import dataclass
from typing import Dict, List, Optional,Tuple
from datetime import datetime, date


@dataclass
class WorkerMeta:
    worker_id: int
    available: bool
    rest_day: List[date]
    tech_speed_ratio: Dict[int, float]


@dataclass
class ResourceGroup:
    group_id: int
    group_name: str
    machine_id_list: List[int]
    worker_id_list: List[int]
    worker_max_parallel: int = 2

@dataclass
class MachineMeta:
    machine_id: int
    available: bool
    planned_daily_hour: float
    changeover_time_map: dict


@dataclass
class JobMeta:
    job_id: int
    op_id_list: List[int]
    priority: str
    due_warn_time: float
    due_contract_time: float
    base_weight: float
    quantity: int
    due_delivery_date: date
    due_delivery_time: float

@dataclass
class ManualLockAssign:
    op_global_id: int
    business_op_id: str
    business_op_no: str
    fixed_machine_id: int
    fixed_worker_id: int
    lock_machine: bool = True
    lock_worker: bool = False
    operator: str = ""
    lock_reason: str = ""
    lock_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None

    def __post_init__(self):
        if self.lock_time is None:
            self.lock_time = datetime.now()
        if self.last_update_time is None:
            self.last_update_time = datetime.now()

    def to_dict(self) -> dict:
        return {
            "op_global_id": self.op_global_id,
            "fixed_machine_id": self.fixed_machine_id,
            "fixed_worker_id": self.fixed_worker_id,
            "lock_machine": self.lock_machine,
            "lock_worker": self.lock_worker,
            "operator": self.operator,
            "lock_reason": self.lock_reason,
            "lock_time": self.lock_time.isoformat() if self.lock_time else None,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None
        }

    @staticmethod
    def from_dict(data: dict):
        lock = ManualLockAssign(
            op_global_id=data["op_global_id"],
            fixed_machine_id=data["fixed_machine_id"],
            fixed_worker_id=data["fixed_worker_id"],
            lock_machine=data["lock_machine"],
            lock_worker=data["lock_worker"],
            operator=data.get("operator", ""),
            lock_reason=data.get("lock_reason", "")
        )
        if data.get("lock_timestamp"):
            lock.lock_timestamp = datetime.fromisoformat(data["lock_timestamp"])
        if data.get("last_update_time"):
            lock.last_update_time = datetime.fromisoformat(data["last_update_time"])
        return lock


@dataclass
class OperationMeta:
    op_global_id: int
    op_status: int
    business_op_id: str
    business_op_no: str
    op_name: str
    op_content: str
    belong_job_id: int
    resource_group_id: int
    resource_group_name: str
    process_time: float
    op_index_in_job: int
    op_quantity: int
    material_ready_time: float = 0.0
    op_tech_type: int = 0
    op_lock_info: Optional[ManualLockAssign] = None

    def __post_init__(self):
        if self.op_lock_info is None:
            self.op_lock_info = ManualLockAssign(
                op_global_id=self.op_global_id,
                fixed_machine_id=-1,
                fixed_worker_id=-1,
                lock_machine=False,
                lock_worker=False,
                operator="",
                lock_reason=""
            )

@dataclass
class ShiftSegment:
    """单个班次时间段（允许跨午夜）"""
    shift_name: str          # 班次名称，如 "白班", "夜班"
    start_hour: float        # 开始小时（0~24），如 20.0
    end_hour: float          # 结束小时（0~24），如 1.0 表示次日凌晨1点

    def to_natural_day_segments(self) -> List[Tuple[str, float, float]]:
        """
        将可能跨午夜的班次拆解为自然日（0~24）内的段，返回 [(名称, 开始, 结束), ...]
        例如：夜班 (20.0, 1.0) → [("夜班", 20.0, 24.0), ("夜班", 0.0, 1.0)]
        """
        if self.start_hour < self.end_hour:
            return [(self.shift_name, self.start_hour, self.end_hour)]
        else:
            segments = []
            if self.start_hour < 24.0:
                segments.append((self.shift_name, self.start_hour, 24.0))
            if self.end_hour > 0.0:
                segments.append((self.shift_name, 0.0, self.end_hour))
            return segments