from typing import Dict, List, Set, Optional
from datetime import datetime, date, time, timedelta
from config.settings import *
from core.calendar import ShiftCalendar
from core.data_structs import *
from utils.log_utils import get_logger

logger = get_logger(__name__)


class ProductionStateManager:
    def __init__(self):
        self.job_meta_dict: Dict[str, JobMeta] = {}
        self.op_meta_dict: Dict[int, OperationMeta] = {}
        self.resource_group_dict: Dict[int, ResourceGroup] = {}
        self.work_calendar: Optional[ShiftCalendar] = None
        self.operation_id_to_job_id: dict[int, str] = {}
        self.operation_status_dict: Dict[int, int] = {}
        self.machine_meta_dict: Dict[int, MachineMeta] = {}
        self.worker_meta_dict: Dict[int, WorkerMeta] = {}
        self.last_pareto_solutions: List[dict] = []
        self.manual_lock_dict: Dict[int, ManualLockAssign] = {}
        self.current_system_time: datetime = datetime.min
        self.last_schedule_result: Dict[int, dict] = {}

    # ===================== 时间相关 =====================

    def get_valid_start_time(self, ideal_start: datetime) -> datetime:
        return self.work_calendar.next_work_start(ideal_start)

    def calculate_actual_work_end_time(self, start_time: datetime, work_duration: float) -> datetime:
        return self.work_calendar.add_work_hours(start_time, work_duration)

    def work_hours_between(self, start_time: datetime, end_time: datetime) -> float:
        return self.work_calendar.work_hours_between(start_time, end_time)

    def datetime_to_iso(self, dt: datetime) -> str:
        return dt.isoformat()

    def is_workday(self, dt: date) -> bool:
        return self.work_calendar.is_workday(dt)

    # ===================== 系统时间管理 =====================

    def advance_system_time(self, hours: float):
        if hours < 0:
            raise ValueError("系统时间不能倒退")
        self.current_system_time += timedelta(hours=hours)
        freeze = self.current_system_time + timedelta(hours=PLAN_FROZEN_HORIZON_HOURS)
        print(f"系统时间已推进至：{self.current_system_time.isoformat()}")
        print(f"当前计划冻结区间：开工时间 < {freeze.isoformat()}")

    def set_system_time(self, dt: datetime):
        if self.work_calendar and dt < self.work_calendar.base_zero:
            raise ValueError("系统时间不能早于基准零点")
        self.current_system_time = dt
        freeze = dt + timedelta(hours=PLAN_FROZEN_HORIZON_HOURS) if self.work_calendar else dt
        print(f"系统时间已设置为：{dt.isoformat()}")
        print(f"当前计划冻结区间：开工时间 < {freeze.isoformat()}")

    # ===================== 工序管理 =====================

    def get_optimizable_operation_ids(self) -> List[int]:
        return [op_idx for op_idx, status in self.operation_status_dict.items() if status == OP_STATUS_OPTIMIZABLE]

    def get_resource_group_by_op(self, op_id: int) -> ResourceGroup:
        op_meta = self.op_meta_dict[op_id]
        return self.resource_group_dict[op_meta.resource_group_id]

    def get_new_job_ratio(self, old_job_ids: Set[int]) -> float:
        current_jobs = set(self.job_meta_dict.keys())
        new_jobs = current_jobs - old_job_ids
        if len(current_jobs) == 0:
            return 0.0
        return len(new_jobs) / len(current_jobs)

    def calc_delivery_overdue_penalty(self, finish_time: datetime, job_meta: JobMeta) -> float:
        delivery_t = job_meta.due_delivery_time
        weight = job_meta.base_weight
        penalty = 0.0
        if finish_time > delivery_t:
            delta = (finish_time - delivery_t).total_seconds() / 3600.0
            if job_meta.priority in ["urgent"]:
                penalty = weight * DELIVERY_OVERDUE_COEFFICIENT * (delta ** 2)
            else:
                penalty = weight * DELIVERY_OVERDUE_COEFFICIENT * delta
        return penalty

    # ===================== 手动锁定管理 =====================

    def add_manual_lock(self, lock_info: ManualLockAssign):
        lock_info.last_update_time = datetime.now()
        self.manual_lock_dict[lock_info.op_global_id] = lock_info

    def remove_manual_lock(self, op_id: int):
        if op_id in self.manual_lock_dict:
            del self.manual_lock_dict[op_id]

    def is_op_manual_locked(self, op_id: int) -> bool:
        return op_id in self.manual_lock_dict

    def get_lock_info(self, op_id: int) -> Optional[ManualLockAssign]:
        return self.manual_lock_dict.get(op_id)

    def export_all_manual_lock(self) -> List[dict]:
        return [lock.to_dict() for lock in self.manual_lock_dict.values()]

    def load_manual_lock_from_db(self, db_data_list: List[dict]):
        self.manual_lock_dict.clear()
        for item in db_data_list:
            lock_obj = ManualLockAssign.from_dict(item)
            self.manual_lock_dict[lock_obj.op_global_id] = lock_obj

    # ===================== 资源可用性 =====================

    def is_machine_available(self, machine_id: int) -> bool:
        machine = self.machine_meta_dict.get(machine_id)
        return machine and machine.available

    def is_worker_available(self, worker_id: int) -> bool:
        worker = self.worker_meta_dict.get(worker_id)
        return worker and worker.available

    def get_available_machines(self, resource_group_machine_id_list: List[int]) -> List[int]:
        return [m_id for m_id in resource_group_machine_id_list if
                (machine := self.machine_meta_dict.get(m_id)) and machine.available]

    def get_available_workers(self, resource_group_worker_id_list: List[int]) -> List[int]:
        return [w_id for w_id in resource_group_worker_id_list if
                (worker := self.worker_meta_dict.get(w_id)) and worker.available]

    # ===================== 缓存与加载 =====================

    def cache_schedule_result(self, schedule_detail: List[dict]):
        self.last_schedule_result = {}
        for item in schedule_detail:
            self.last_schedule_result[item["op_id"]] = {
                "start_time": item["start_time"],
                "machine_id": item["machine_id"],
                "worker_id": item["worker_id"],
                "sequence": item["op_id"]
            }
        print(f"已缓存本次调度结果，共 {len(schedule_detail)} 道工序")

    def get_next_workday_start_time(self, current_datetime: datetime = None) -> datetime:
        if current_datetime is None:
            current_datetime = datetime.now()
        candidate_date = current_datetime.date() + timedelta(days=1)
        while not self.work_calendar.is_workday(candidate_date):
            candidate_date += timedelta(days=1)
        cal = self.work_calendar
        segs = cal._get_segments(candidate_date)
        if not segs:
            return datetime.combine(candidate_date, time(0, 0))
        first_seg = segs[0]
        for sh in cal._get_shifts(candidate_date):
            if sh.name == "白班" and sh.segments:
                first_seg = sh.segments[0]
                break
        return datetime.combine(candidate_date, first_seg.start)

    def load_shift_data_from_db(self, base_date: date, shift_configs: dict) -> ShiftCalendar:
        weekly_shifts: Dict[int, List[Shift]] = {}
        special_shifts: Dict[date, List[Shift]] = {}

        for wd_str, shift_list in shift_configs.get("weekly_shifts", {}).items():
            wd = int(wd_str)
            shifts = []
            for sh_data in shift_list:
                segments = tuple(
                    ShiftSegment(start=time.fromisoformat(seg["start"]), end=time.fromisoformat(seg["end"]))
                    for seg in sh_data["segments"]
                )
                shifts.append(Shift(name=sh_data["name"], segments=segments))
            weekly_shifts[wd] = shifts

        for dt_str, shift_list in shift_configs.get("special_shifts", {}).items():
            dt = date.fromisoformat(dt_str)
            shifts = []
            for sh_data in shift_list:
                segments = tuple(
                    ShiftSegment(start=time.fromisoformat(seg["start"]), end=time.fromisoformat(seg["end"]))
                    for seg in sh_data["segments"]
                )
                shifts.append(Shift(name=sh_data["name"], segments=segments))
            special_shifts[dt] = shifts

        self.work_calendar = ShiftCalendar(
            base_date=base_date,
            weekly_shifts=weekly_shifts,
            special_shifts=special_shifts
        )
        if self.current_system_time == datetime.min:
            self.current_system_time = self.work_calendar.base_zero
        return self.work_calendar