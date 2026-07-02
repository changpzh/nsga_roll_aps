from datetime import datetime, date, time, timedelta
from typing import Set
import json
import os
import hashlib

from config.settings import JOB_PRIORITY_WEIGHT
from core.state_manager import ProductionStateManager
from core.data_structs import ResourceGroup, MachineMeta, WorkerMeta, OperationMeta, JobMeta, ManualLockAssign, Shift, ShiftSegment
from core.work_calendar import ShiftCalendar

from utils.log_utils import get_logger
logger = get_logger(__name__)


def load_production_json(json_path: str = "data/test_data.json") -> dict:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_file_dir, json_path)
    print(f"full_path: {full_path} ")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"数据文件不存在: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_day_end_datetime(state_manager: ProductionStateManager, target_date: date) -> datetime:
    cal = state_manager.work_calendar
    if cal is None:
        return datetime.combine(target_date, time(0, 0))
    segs = cal._get_segments(target_date)
    if not segs:
        return datetime.combine(target_date, time(0, 0))
    last_seg = segs[-1]
    end_dt = datetime.combine(target_date, last_seg.end)
    if last_seg.end <= last_seg.start:
        end_dt += timedelta(days=1)
    return end_dt


def _generate_stable_op_id(job_id: str, business_op_id: str, used_ids: Set[int] = None) -> int:
    """
    生成稳定工序ID，带冲突检测
    :param used_ids: 已使用的ID集合，用于冲突校验
    """
    unique_str = f"{job_id}:{business_op_id}"
    hash_bytes = hashlib.sha256(unique_str.encode()).digest()[:8]
    op_id = int.from_bytes(hash_bytes, 'big')

    # 冲突检测：若已存在，向后偏移直到找到空位
    if used_ids is not None:
        offset = 1
        while op_id in used_ids:
            op_id = int.from_bytes(hash_bytes, 'big') + offset
            offset += 1
        used_ids.add(op_id)

    return op_id


def build_test_production_data(state_manager: ProductionStateManager, json_path: str):
    sm = state_manager
    data = load_production_json(json_path)

    # ========== 1. 工作日历 ==========
    base_date = sm.current_system_time.date() if sm.current_system_time else date.today()

    weekly_shifts = {}
    for wd_str, shift_list in data.get("weekly_shifts", {}).items():
        wd = int(wd_str)
        shifts = []
        for sh_data in shift_list:
            segments = tuple(
                ShiftSegment(start=time.fromisoformat(seg["start"]), end=time.fromisoformat(seg["end"]))
                for seg in sh_data["segments"]
            )
            shifts.append(Shift(name=sh_data["name"], segments=segments))
        weekly_shifts[wd] = shifts

    special_shifts = {}
    for dt_str, shift_list in data.get("special_shifts", {}).items():
        dt = date.fromisoformat(dt_str)
        shifts = []
        for sh_data in shift_list:
            segments = tuple(
                ShiftSegment(start=time.fromisoformat(seg["start"]), end=time.fromisoformat(seg["end"]))
                for seg in sh_data["segments"]
            )
            shifts.append(Shift(name=sh_data["name"], segments=segments))
        special_shifts[dt] = shifts

    sm.work_calendar = ShiftCalendar(
        base_date=base_date,
        weekly_shifts=weekly_shifts,
        special_shifts=special_shifts
    )

    # ========== 2. 资源组 ==========
    sm.resource_group_dict.clear()
    for rg_data in data.get("resource_groups", []):
        rg = ResourceGroup(
            group_id=rg_data["group_id"],
            group_name=rg_data["group_name"],
            machine_id_list=rg_data["machine_id_list"],
            worker_id_list=rg_data["worker_id_list"],
            worker_max_parallel=rg_data["worker_max_parallel"]
        )
        sm.resource_group_dict[rg.group_id] = rg

    # ========== 3. 机床换型时间映射 ==========
    changeover_raw = data.get("machine_changeover_map", {})
    change_map = {}
    for mid_str, inner in changeover_raw.items():
        mid = int(mid_str)
        inner_conv = {int(k): v for k, v in inner.items()}
        change_map[mid] = inner_conv

    # ========== 4. 机床信息 ==========
    sm.machine_meta_dict.clear()
    for mid_str, m_info in data.get("machine_dict", {}).items():
        mid = int(mid_str)
        sm.machine_meta_dict[mid] = MachineMeta(
            machine_id=mid,
            available=m_info["available"],
            planned_daily_hour=m_info.get("planned_daily_hour", 12.0),
            changeover_time_map=change_map
        )

    # ========== 5. 工人信息 ==========
    sm.worker_meta_dict.clear()
    for wid_str, w_info in data.get("worker_dict", {}).items():
        wid = int(wid_str)
        rest_dates = [date.fromisoformat(ds) for ds in w_info.get("rest_day", [])]
        speed_raw = w_info.get("tech_speed_ratio", {})
        speed_conv = {int(k): v for k, v in speed_raw.items()}
        sm.worker_meta_dict[wid] = WorkerMeta(
            worker_id=wid,
            available=w_info["available"],
            rest_day=rest_dates,
            tech_speed_ratio=speed_conv
        )

    # ========== 6. 订单+工序 ==========
    all_job_op_map = {}
    sm.op_meta_dict.clear()
    sm.operation_id_to_job_id.clear()
    sm.job_meta_dict.clear()

    for job_data in data.get("job_list", []):
        jid = job_data["job_id"]
        op_info_list = job_data["op_info_list"]
        op_id_list = []

        for op_index, op_item in enumerate(op_info_list):
            business_op_id = op_item["business_op_id"]
            global_op_id = _generate_stable_op_id(jid, business_op_id)

            lock_dict = op_item.get("op_lock_info", {})
            lock_time_str = lock_dict.get("lock_time")
            lock_time = datetime.strptime(lock_time_str, "%Y-%m-%d %H:%M:%S") if (lock_time_str and lock_time_str.strip()) else None
            lock_obj = ManualLockAssign(
                op_global_id=global_op_id,
                business_op_id=op_item.get("business_op_id", ""),
                business_op_no=op_item.get("business_op_no", ""),
                fixed_machine_id=lock_dict.get("fixed_machine_id", -1),
                fixed_worker_id=lock_dict.get("fixed_worker_id", -1),
                lock_machine=lock_dict.get("lock_machine", False),
                lock_worker=lock_dict.get("lock_worker", False),
                operator=lock_dict.get("operator", ""),
                lock_reason=lock_dict.get("lock_reason", ""),
                lock_time=lock_time,
                last_update_time=lock_time
            )
            if lock_obj.lock_machine or lock_obj.lock_worker:
                sm.add_manual_lock(lock_obj)

            mat_ready_raw = op_item.get("material_ready_time", 0.0)
            if mat_ready_raw and mat_ready_raw > 0:
                material_ready_dt = sm.work_calendar.base_zero + timedelta(hours=mat_ready_raw)
            else:
                material_ready_dt = sm.work_calendar.base_zero

            op_meta = OperationMeta(
                op_global_id=global_op_id,
                op_status=op_item["op_status"],
                business_op_id=op_item.get("business_op_id", ""),
                business_op_no=op_item.get("business_op_no", ""),
                op_name=op_item.get("op_name", ""),
                op_content=op_item.get("op_content", ""),
                belong_job_id=jid,
                resource_group_id=op_item["resource_group_id"],
                resource_group_name=op_item.get("resource_group_name", ""),
                process_time=op_item["process_time"],
                op_index_in_job=op_index,
                op_quantity=op_item.get("op_quantity", 1),
                material_ready_time=material_ready_dt,
                op_tech_type=op_item.get("op_tech_type", 0),
                op_lock_info=lock_obj,
                size_factor=op_item.get("size_factor", 1.0),
                min_batch_size=op_item.get("min_batch_size", 0),
                mergeable=op_item.get("mergeable", False)
            )
            sm.op_meta_dict[global_op_id] = op_meta
            sm.operation_id_to_job_id[global_op_id] = jid
            op_id_list.append(global_op_id)

        all_job_op_map[jid] = op_id_list
        base_weight = JOB_PRIORITY_WEIGHT.get(job_data["priority"], 1.0)

        delivery_raw = job_data.get("due_delivery_date")
        due_delivery_date = None
        due_delivery_time = sm.work_calendar.base_zero
        if delivery_raw:
            due_delivery_date = date.fromisoformat(delivery_raw)
            due_delivery_time = _get_day_end_datetime(sm, due_delivery_date)

        job_meta = JobMeta(
            job_id=jid,
            op_id_list=op_id_list,
            priority=job_data["priority"],
            due_warn_time=job_data.get("due_warn_time", 0),
            due_contract_time=job_data.get("due_contract_time", 0),
            base_weight=base_weight,
            quantity=job_data.get("quantity", 1),
            due_delivery_date=due_delivery_date,
            due_delivery_time=due_delivery_time
        )
        sm.job_meta_dict[jid] = job_meta

    sm.operation_status_dict = {op_id: op_meta.op_status for op_id, op_meta in sm.op_meta_dict.items()}

    last_op_id = max(sm.op_meta_dict.keys()) if sm.op_meta_dict else 0
    print(f"JSON数据集加载构建完成｜总订单{len(sm.job_meta_dict)} 总工序{last_op_id}")
    print(f"all_job_op_map: {all_job_op_map}")
    return all_job_op_map


def build_production_data_from_dict(state_manager: ProductionStateManager, data: dict):
    sm = state_manager

    if "base_date" in data:
        base_date = datetime.strptime(data["base_date"], "%Y-%m-%d").date()
        weekly_shifts = {}
        for wd_str, shift_list in data.get("weekly_shifts", {}).items():
            wd = int(wd_str)
            shifts = []
            for sh_data in shift_list:
                segments = tuple(
                    ShiftSegment(start=time.fromisoformat(s["start"]), end=time.fromisoformat(s["end"]))
                    for s in sh_data["segments"]
                )
                shifts.append(Shift(name=sh_data["name"], segments=segments))
            weekly_shifts[wd] = shifts

        special_shifts = {}
        for dt_str, shift_list in data.get("special_shifts", {}).items():
            dt = date.fromisoformat(dt_str)
            shifts = []
            for sh_data in shift_list:
                segments = tuple(
                    ShiftSegment(start=time.fromisoformat(s["start"]), end=time.fromisoformat(s["end"]))
                    for s in sh_data["segments"]
                )
                shifts.append(Shift(name=sh_data["name"], segments=segments))
            special_shifts[dt] = shifts

        sm.work_calendar = ShiftCalendar(base_date=base_date, weekly_shifts=weekly_shifts, special_shifts=special_shifts)

    for rg_data in data.get("resource_groups", []):
        rg = ResourceGroup(
            group_id=rg_data["group_id"], group_name=rg_data["group_name"],
            machine_id_list=rg_data["machine_id_list"], worker_id_list=rg_data["worker_id_list"],
            worker_max_parallel=rg_data["worker_max_parallel"]
        )
        sm.resource_group_dict[rg.group_id] = rg

    for mid_str, m_info in data.get("machine_dict", {}).items():
        mid = int(mid_str)
        if mid in sm.machine_meta_dict:
            sm.machine_meta_dict[mid].available = m_info["available"]
        else:
            changeover_raw = data.get("machine_changeover_map", {})
            change_map = {int(k): {int(ik): iv for ik, iv in v.items()} for k, v in changeover_raw.items()}
            sm.machine_meta_dict[mid] = MachineMeta(
                machine_id=mid, available=m_info["available"],
                planned_daily_hour=m_info.get("planned_daily_hour", 12.0),
                changeover_time_map=change_map,
                standard_capacity=m_info.get("standard_capacity", 0)
            )

    for wid_str, w_info in data.get("worker_dict", {}).items():
        wid = int(wid_str)
        rest_dates = [date.fromisoformat(ds) for ds in w_info.get("rest_day", [])]
        speed_conv = {int(k): v for k, v in w_info.get("tech_speed_ratio", {}).items()}
        if wid in sm.worker_meta_dict:
            sm.worker_meta_dict[wid].available = w_info["available"]
            sm.worker_meta_dict[wid].rest_day = rest_dates
            sm.worker_meta_dict[wid].tech_speed_ratio = speed_conv
        else:
            sm.worker_meta_dict[wid] = WorkerMeta(worker_id=wid, available=w_info["available"],
                                                   rest_day=rest_dates, tech_speed_ratio=speed_conv)

    for job_data in data.get("job_list", []):
        jid = job_data["job_id"]
        op_info_list = job_data["op_info_list"]
        if jid not in sm.job_meta_dict:
            op_id_list = []
            for op_index, op_item in enumerate(op_info_list):
                global_op_id = _generate_stable_op_id(jid, op_item["business_op_id"])
                op_id_list.append(global_op_id)
                if global_op_id not in sm.op_meta_dict:
                    lock_dict = op_item.get("op_lock_info", {})
                    lock_time_str = lock_dict.get("lock_time")
                    lock_time = datetime.strptime(lock_time_str, "%Y-%m-%d %H:%M:%S") if (lock_time_str and lock_time_str.strip()) else None
                    lock_obj = ManualLockAssign(
                        op_global_id=global_op_id,
                        business_op_id=op_item.get("business_op_id", ""),
                        business_op_no=op_item.get("business_op_no", ""),
                        fixed_machine_id=lock_dict.get("fixed_machine_id", -1),
                        fixed_worker_id=lock_dict.get("fixed_worker_id", -1),
                        lock_machine=lock_dict.get("lock_machine", False),
                        lock_worker=lock_dict.get("lock_worker", False),
                        operator=lock_dict.get("operator", ""),
                        lock_reason=lock_dict.get("lock_reason", ""),
                        lock_time=lock_time, last_update_time=lock_time
                    )
                    mat_ready_raw = op_item.get("material_ready_time", 0.0)
                    material_ready_dt = sm.work_calendar.base_zero + timedelta(hours=mat_ready_raw) if (mat_ready_raw and mat_ready_raw > 0) else sm.work_calendar.base_zero
                    op_meta = OperationMeta(
                        op_global_id=global_op_id, op_status=op_item["op_status"],
                        business_op_id=op_item.get("business_op_id", ""),
                        business_op_no=op_item.get("business_op_no", ""),
                        op_name=op_item.get("op_name", ""), op_content=op_item.get("op_content", ""),
                        belong_job_id=jid, resource_group_id=op_item["resource_group_id"],
                        resource_group_name=op_item.get("resource_group_name", ""),
                        process_time=op_item["process_time"], op_index_in_job=op_index,
                        op_quantity=op_item.get("op_quantity", 1),
                        material_ready_time=material_ready_dt,
                        op_tech_type=op_item.get("op_tech_type", 0), op_lock_info=lock_obj
                    )
                    sm.op_meta_dict[global_op_id] = op_meta
                    sm.operation_id_to_job_id[global_op_id] = jid

            base_weight = JOB_PRIORITY_WEIGHT.get(job_data["priority"], 1.0)
            delivery_raw = job_data.get("due_delivery_date")
            due_delivery_date = date.fromisoformat(delivery_raw) if delivery_raw else None
            due_delivery_time = _get_day_end_datetime(sm, due_delivery_date) if due_delivery_date else sm.work_calendar.base_zero
            sm.job_meta_dict[jid] = JobMeta(
                job_id=jid, op_id_list=op_id_list, priority=job_data["priority"],
                due_warn_time=job_data.get("due_warn_time", 0),
                due_contract_time=job_data.get("due_contract_time", 0),
                base_weight=base_weight, quantity=job_data.get("quantity", 1),
                due_delivery_date=due_delivery_date, due_delivery_time=due_delivery_time
            )

    sm.operation_status_dict = {op_id: meta.op_status for op_id, meta in sm.op_meta_dict.items()}