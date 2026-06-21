from typing import Dict, List, Set, Tuple
from datetime import timedelta
from config.settings import *
from core.calendar import WorkCalendar
from core.data_structs import *
import numpy as np

class ProductionStateManager:
    def __init__(self):
        # 所有工件的元数据字典 | 核心：获取订单交期、优先级、工艺路线 | 例子：{1001: JobMeta(订单号SO20260617001, 交期2026-06-20, 优先级1)}
        self.job_meta_dict: Dict[int, JobMeta] = {}
        # 所有工序的元数据字典 | 核心：获取工序加工时间、工艺类型、所属资源组 | 例子：{100101: OperationMeta(工序名粗车外圆, 基础加工时间2.5, 工艺类型1)}
        self.op_meta_dict: Dict[int, OperationMeta] = {}
        # 所有资源组的配置字典 | 核心：获取某类工序可用的机器和工人列表 | 例子：{1: ResourceGroup(名称车削组, 机器[1,2], 工人[101,102], 工人最大并行1)}
        self.resource_group_dict: Dict[int, ResourceGroup] = {}
        # 全局工作日历 | 核心：计算合法的开始/结束时间，排除休息和节假日 | 例子：WorkCalendar(基准日期2026-06-17 08:00, 班次8-12,13-17, 周末休息)
        self.work_calendar: Optional[WorkCalendar] = None
        # 操作ID到工件ID的快速反向映射 | 核心优化：O(1)查找某道工序属于哪个工件 | 例子：{100101: 1001, 100102: 1001}
        self.operation_id_to_job_id: dict[int, int] = {}
        # 所有工序的当前状态 | 核心：区分已完成/进行中/未开始，仅未开始工序参与排程 | 例子：{100101:0, 100102:1, 100103:2} (键=操作ID, 0=未开始,1=进行中,2=已完成)
        self.operation_status_dict: Dict[int, int] = {}
        # 所有机器的信息 | 核心： | 例子：
        self.machine_meta_dict: Dict[int, MachineMeta] = {}
        # 所有工人的信息 | 核心：| 例子：
        self.worker_meta_dict: Dict[int, WorkerMeta] = {}
        # 上一次优化得到的帕累托解集 | 核心：滚动排程时复用历史解，大幅提高优化速度 | 例子：[{"op_sequence": [...], "resource_assign": [...], "fitness": [...]}]
        self.last_pareto_solutions: List[dict] = []
        # 用户手动锁定的操作配置 | 核心：算法不能修改锁定操作的资源分配和时间 | 例子：{100101: ManualLockAssign(lock_machine=True, fixed_machine_id=1)}
        self.manual_lock_dict: Dict[int, ManualLockAssign] = {}
        # 当前系统时间 | 距离WorkCalendar基准base_date的浮点小时数，只代表生产推进的相对工时刻度 | 例子：0.0=2026-06-17 08:00, 8.0=2026-06-17 16:00
        self.current_system_time: float = 0.0
        # 上一次的完整排程结果 | 核心：冻结区间内的操作直接复用历史结果，保证计划稳定性 | 例子：{100101: {"start_time":0.0, "end_time":2.0, "machine_id":1}}
        self.last_schedule_result: Dict[int, dict] = {}
        # 设备每日计划工作时长,默认12个小时
        self.machine_planned_daily_hour: float = 12.0

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

    # def get_machine_overload_penalty(self, machine_id: int, total_load: float):
    #     machine_meta = self.machine_meta_dict.get(machine_id)
    #     if machine_meta is None:
    #         return 0.0
    #     if total_load > machine_meta.planned_daily_hour:
    #         return (total_load - machine_meta.planned_daily_hour) * OVERLOAD_PENALTY_COEFFICIENT
    #     return 0.0

    def calc_delivery_overdue_penalty(self, finish_time: float, job_meta: JobMeta) -> float:
        delivery_t = job_meta.due_delivery_time
        weight = job_meta.base_weight
        penalty = 0.0
        if finish_time > delivery_t:
            delta = finish_time - delivery_t
            if job_meta.priority in ["urgent"]:
                penalty = weight * DELIVERY_OVERDUE_COEFFICIENT * (delta ** 2)  # 平方惩罚
            else:
                penalty = weight * DELIVERY_OVERDUE_COEFFICIENT * delta # 线性惩罚

        return penalty

    def refresh_production_status(self):
        pass

    def add_manual_lock(self, lock_info: ManualLockAssign):
        if not lock_info.lock_machine and not lock_info.lock_worker:
            pass
        lock_info.last_update_time = datetime.now()
        self.manual_lock_dict[lock_info.op_global_id] = lock_info

    def remove_manual_lock(self, op_id: int):
        if op_id in self.manual_lock_dict:
            del self.manual_lock_dict[op_id]

    def is_op_manual_locked(self, op_id: int) -> bool:
        return op_id in self.manual_lock_dict

    def get_lock_info(self, op_id: int) -> ManualLockAssign | None:
        return self.manual_lock_dict.get(op_id, None)

    def export_all_manual_lock(self) -> List[dict]:
        return [lock.to_dict() for lock in self.manual_lock_dict.values()]

    def load_manual_lock_from_db(self, db_data_list: List[dict]):
        self.manual_lock_dict.clear()
        for item in db_data_list:
            lock_obj = ManualLockAssign.from_dict(item)
            self.manual_lock_dict[lock_obj.op_global_id] = lock_obj

    def insert_new_order(self, job_id: int, priority: str, warn_due: float, due_contract_time: float, base_weight: float,
                         op_info_list: List[dict]) -> List[int]:
        new_job = JobMeta(
            job_id=job_id,
            op_id_list=[],
            priority=priority,
            due_warn_time=warn_due,
            due_contract_time=due_contract_time,
            base_weight=base_weight
        )
        self.job_meta_dict[job_id] = new_job
        start_op_id = len(self.op_meta_dict)
        new_op_ids = []
        for op_index, op_item in enumerate(op_info_list):
            op = OperationMeta(
                op_global_id=start_op_id + op_index,
                business_op_id=op_item["business_op_id"],
                business_op_no=op_item["business_op_no"],
                op_name=op_item["op_name"],
                op_content=op_item["op_content"],
                belong_job_id=job_id,
                resource_group_id=op_item["resource_group_id"],
                resource_group_name=op_item["resource_group_name"],
                process_time=op_item["process_time"],
                op_index_in_job=op_index,
                material_ready_time=op_item.get("material_ready_time", 0.0),
                op_tech_type=op_item.get("op_tech_type", 0)
            )
            self.op_meta_dict[op.op_global_id] = op
            self.operation_id_to_job_id[op.op_global_id] = job_id
            self.operation_status_dict.append(OP_STATUS_OPTIMIZABLE)
            new_op_ids.append(op.op_global_id)
        new_job.op_id_list = new_op_ids
        return new_op_ids

    def refresh_optimizable_operation_pool(self):
        self.get_optimizable_operation_ids()

    def is_work_day(self, day_num:int) -> bool:
        cal = self.work_calendar
        return cal.is_workday(day_num) if cal else True

    def get_valid_start_time(self, ideal_start: float) -> float:
        cal = self.work_calendar
        return cal.get_valid_start_time_skip_holidays(ideal_start) if cal else ideal_start

    def calculate_actual_work_end_time(self, start_time: float, work_duration: float) -> float:
        cal = self.work_calendar
        if cal is None:
            return start_time + work_duration
        return cal.calculate_actual_work_end_time_skip_holidays(start_time, work_duration)

    def relative_hour_to_datetime(self, relative_hour: float) -> datetime:
        if self.work_calendar is None:
            return datetime.now()
        delta = timedelta(hours=relative_hour)
        return datetime.combine(self.work_calendar.base_date, datetime.min.time()) + delta

    def relative_hour_to_iso(self, relative_hour: float) -> str:
        return self.relative_hour_to_datetime(relative_hour).isoformat()

    def advance_system_time(self, hours: float):
        if hours < 0:
            raise ValueError("系统时间不能倒退")
        self.current_system_time += hours
        print(f"系统时间已推进：{self.current_system_time:.1f} 小时")
        print(f"当前计划冻结区间：开工时间 < {self.current_system_time + PLAN_FROZEN_HORIZON:.1f} 小时")

    def set_system_time(self, relative_hour: float):
        if relative_hour < 0:
            raise ValueError("系统时间不能为负数")
        self.current_system_time = relative_hour
        print(f"系统时间已设置为：{self.current_system_time:.1f} 小时")
        print(f"当前计划冻结区间：开工时间 < {self.current_system_time + PLAN_FROZEN_HORIZON:.1f} 小时")

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
        return  [w_id for w_id in resource_group_worker_id_list if
                 (worker := self.worker_meta_dict.get(w_id)) and worker.available]

    def get_datetime_to_relative_hours(self, target_datetime: datetime) -> float:
        cal = self.work_calendar
        return cal.datetime_to_relative_hour(target_datetime)

    def get_schedule_total_work_days_horizon(self, max_delivery_time: float) -> int:
        """
        返回本次排程的总天数
        计算方式：取所有订单的最大交付日期与基准日期的天数差，向上取整
        """
        if not self.job_meta_dict:
            return 1  # 没有订单时默认返回1天

        # 转换为天数（向上取整，确保覆盖整个排程周期）
        daily_work_hours = self.work_calendar.daily_work_hours
        total_days = int(np.ceil(max_delivery_time / daily_work_hours))

        total_work_days = 0
        base_date = self.work_calendar.base_date.date()
        # 遍历排程周期内的每一天
        for day_offset in range(total_days):
            current_date = base_date + timedelta(days=day_offset)
            # 只有工作日才计算工时
            if self.work_calendar.is_workday(current_date):
                total_work_days += 1

        # 至少返回1天
        return max(1, total_work_days)

    def get_schedule_total_work_hours_horizon(self, max_delivery_time: float, planned_daily_hour: float = 12.0) -> float:
        """
        返回本次排程周期内所有可用设备的总可用工时
        自动排除周末和法定节假日，最后一天按实际需要的工时计算
        """
        # 转换为天数（向上取整，确保覆盖整个排程周期）
        full_days = int(np.floor(max_delivery_time / planned_daily_hour))
        last_day_remaining_hours = max_delivery_time % planned_daily_hour

        total_work_hours = 0.0
        base_date = self.work_calendar.base_date.date()

        # 计算full_days天的总可用工时（完整工作日）
        for day_offset in range(full_days):
            current_date = base_date + timedelta(days=day_offset)
            # 只有工作日才计算工时
            if self.work_calendar.is_workday(current_date):
                total_work_hours += planned_daily_hour

        total_work_hours += last_day_remaining_hours

        return total_work_hours