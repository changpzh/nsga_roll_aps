# data/test_dataset.py
from datetime import datetime, date
import json
import os
import config as cfg
from config.settings import JOB_PRIORITY_WEIGHT, SHIFT_CONFIG_FILE, BASE_DATE, SHIFT_DATA_SOURCE
from core.state_manager import ProductionStateManager
from core.data_structs import ResourceGroup, MachineMeta, WorkerMeta, ShiftSegment
from core.calendar import WorkCalendar
from core.data_structs import OperationMeta, JobMeta, ManualLockAssign
import hashlib

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


def _load_shift_configs_from_json(shift_config_path: str = None) -> list:
    """从 JSON 文件加载班次配置（独立于主数据文件）"""
    if shift_config_path is None:
        shift_config_path = SHIFT_CONFIG_FILE
    # 如果是相对路径，基于项目根目录
    if not os.path.isabs(shift_config_path):
        from config.settings import PROJECT_ROOT
        shift_config_path = PROJECT_ROOT / shift_config_path
    if os.path.exists(shift_config_path):
        with open(shift_config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def build_test_production_data(state_manager: ProductionStateManager, json_path: str = "test_data.json"):
    """
    从 JSON 构建测试数据（适配新日历架构）
    班次配置从独立 JSON 文件加载，与主数据解耦
    """
    sm = state_manager
    data = load_production_json(json_path)

    # ========== 1. 工作日历（新架构：从独立班次配置文件加载） ==========
    cal_cfg = data.get("calendar_config", {})
    base_date = datetime.strptime(cal_cfg.get("base_date", str(BASE_DATE)), "%Y-%m-%d").date()

    # 加载班次配置（优先使用独立配置文件，若不存在则从主数据中读取）
    shift_configs = _load_shift_configs_from_json()
    if not shift_configs:
        # 兼容旧版：如果主数据中有 shift_configs 字段，则使用它
        shift_configs = data.get("shift_configs", [])
    # 如果仍然为空，使用默认配置（周一至周五白班8-12,14-18）
    if not shift_configs:
        shift_configs = _get_default_shift_configs()

    # 使用新方法初始化日历
    sm.load_shift_data_from_db(base_date=base_date, shift_configs=shift_configs)

    # 处理旧格式中的特殊日期（兼容：如果主数据中有 special_date_work_map，转换为 special_rules）
    if "special_date_work_map" in data:
        _convert_old_special_dates(sm, data["special_date_work_map"])

    # ========== 2. 资源组（不变） ==========
    for rg_data in data["resource_groups"]:
        rg = ResourceGroup(
            group_id=rg_data["group_id"],
            group_name=rg_data["group_name"],
            machine_id_list=rg_data["machine_id_list"],
            worker_id_list=rg_data["worker_id_list"],
            worker_max_parallel=rg_data["worker_max_parallel"]
        )
        sm.resource_group_dict[rg.group_id] = rg

    # ========== 3. 机床换型时间映射（不变） ==========
    changeover_raw = data.get("machine_changeover_map", {})
    change_map = {}
    for mid_str, inner in changeover_raw.items():
        mid = int(mid_str)
        inner_conv = {int(k): v for k, v in inner.items()}
        change_map[mid] = inner_conv

    # ========== 4. 机床信息（不变） ==========
    sm.machine_meta_dict.clear()
    for mid_str, m_info in data["machine_dict"].items():
        mid = int(mid_str)
        sm.machine_meta_dict[mid] = MachineMeta(
            machine_id=mid,
            available=m_info["available"],
            planned_daily_hour=m_info["planned_daily_hour"],
            changeover_time_map=change_map
        )

    # ========== 5. 工人信息（不变） ==========
    sm.worker_meta_dict.clear()
    for wid_str, w_info in data["worker_dict"].items():
        wid = int(wid_str)
        rest_dates = [datetime.strptime(ds, "%Y-%m-%d").date() for ds in w_info["rest_day"]]
        speed_raw = w_info["tech_speed_ratio"]
        speed_conv = {int(k): v for k, v in speed_raw.items()}
        w_meta = WorkerMeta(
            worker_id=wid,
            available=w_info["available"],
            rest_day=rest_dates,
            tech_speed_ratio=speed_conv
        )
        sm.worker_meta_dict[wid] = w_meta

    # ========== 6. 订单+工序（交期计算适配新日历） ==========
    all_job_op_map = {}
    sm.op_meta_dict.clear()
    sm.operation_id_to_job_id.clear()
    sm.job_meta_dict.clear()

    for job_data in data["job_list"]:
        jid = job_data["job_id"]
        op_info_list = job_data["op_info_list"]
        op_id_list = []

        for op_index, op_item in enumerate(op_info_list):
            business_op_id = op_item["business_op_id"]
            global_op_id = _generate_stable_op_id(jid, business_op_id)

            # 解析锁定信息
            lock_dict = op_item.get("op_lock_info", {})
            lock_time_str = lock_dict.get("lock_time")
            lock_time = datetime.strptime(lock_dict["lock_time"], "%Y-%m-%d %H:%M:%S") if (lock_time_str and lock_time_str.strip()) else None
            lock_obj = ManualLockAssign(
                op_global_id=global_op_id,
                business_op_id=op_item["business_op_id"],
                business_op_no=op_item["business_op_no"],
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

            op_meta = OperationMeta(
                op_global_id=global_op_id,
                op_status=op_item["op_status"],
                business_op_id=op_item["business_op_id"],
                business_op_no=op_item["business_op_no"],
                op_name=op_item["op_name"],
                op_content=op_item["op_content"],
                belong_job_id=jid,
                resource_group_id=op_item["resource_group_id"],
                resource_group_name=op_item["resource_group_name"],
                process_time=op_item["process_time"],
                op_index_in_job=op_index,
                op_quantity=op_item["op_quantity"],
                material_ready_time=op_item["material_ready_time"],
                op_tech_type=op_item["op_tech_type"],
                op_lock_info=lock_obj
            )
            sm.op_meta_dict[global_op_id] = op_meta
            sm.operation_id_to_job_id[global_op_id] = jid
            op_id_list.append(global_op_id)

        all_job_op_map[jid] = op_id_list
        base_weight = JOB_PRIORITY_WEIGHT[job_data["priority"]]

        # ===== 交期计算（适配新日历：使用当天末班结束时间） =====
        delivery_raw = job_data.get("due_delivery_date")
        due_delivery_date = None
        due_delivery_time = 0.0
        if delivery_raw:
            due_delivery_date = datetime.strptime(delivery_raw, "%Y-%m-%d").date()
            # 获取该日期的末班结束时间（相对小时）
            due_delivery_time = _get_day_end_relative_hour(sm, due_delivery_date)

        job_meta = JobMeta(
            job_id=jid,
            op_id_list=op_id_list,
            priority=job_data["priority"],
            due_warn_time=job_data["due_warn_time"],
            due_contract_time=job_data["due_contract_time"],
            base_weight=base_weight,
            quantity=job_data["quantity"],
            due_delivery_date=due_delivery_date,
            due_delivery_time=due_delivery_time
        )
        sm.job_meta_dict[jid] = job_meta

    sm.operation_status_dict = {
        op_id: op_meta.op_status
        for op_id, op_meta in sm.op_meta_dict.items()
    }

    # 获取最后生成的全局工序ID（用于日志）
    last_op_id = max(sm.op_meta_dict.keys()) if sm.op_meta_dict else 0
    print(f"JSON数据集加载构建完成｜总订单{len(sm.job_meta_dict)} 总工序{last_op_id}")
    print(f"all_job_op_map: {all_job_op_map}")
    return all_job_op_map


def _get_day_end_relative_hour(state_manager: ProductionStateManager, target_date: date) -> float:
    """
    获取指定日期的末班结束时间（相对小时）
    如果当天无班次，返回 0.0
    """
    cal = state_manager.work_calendar
    if cal is None:
        return 0.0
    # 获取该日期的所有班次段
    segments = cal.get_segments_for_date(target_date)
    if not segments:
        return 0.0
    # 取最后一个段的结束时间
    last_end_hour = segments[-1][2]  # 三元组 (name, start, end)
    return cal._combine_relative_hour(target_date, last_end_hour)


def _get_default_shift_configs() -> list:
    """返回默认班次配置（周一至周五白班 8-12, 14-18）"""
    return [
        # 周一至周五：白班 8-12, 14-18
        {"day_type": "weekly", "weekday": 0, "shift_name": "白班", "start": 8.0, "end": 12.0},
        {"day_type": "weekly", "weekday": 0, "shift_name": "白班", "start": 14.0, "end": 18.0},
        {"day_type": "weekly", "weekday": 1, "shift_name": "白班", "start": 8.0, "end": 12.0},
        {"day_type": "weekly", "weekday": 1, "shift_name": "白班", "start": 14.0, "end": 18.0},
        {"day_type": "weekly", "weekday": 2, "shift_name": "白班", "start": 8.0, "end": 12.0},
        {"day_type": "weekly", "weekday": 2, "shift_name": "白班", "start": 14.0, "end": 18.0},
        {"day_type": "weekly", "weekday": 3, "shift_name": "白班", "start": 8.0, "end": 12.0},
        {"day_type": "weekly", "weekday": 3, "shift_name": "白班", "start": 14.0, "end": 18.0},
        {"day_type": "weekly", "weekday": 4, "shift_name": "白班", "start": 8.0, "end": 12.0},
        {"day_type": "weekly", "weekday": 4, "shift_name": "白班", "start": 14.0, "end": 18.0},
        # 周六日休息
    ]


def _convert_old_special_dates(state_manager: ProductionStateManager, old_special_map: dict):
    """
    将旧格式的 special_date_work_map 转换为新架构的 special_rules
    旧格式: { "2026-06-22": True/False }  True=上班, False=休息
    新格式: special_rules[date] = [ShiftSegment, ...]  空列表=休息
    """
    cal = state_manager.work_calendar
    if cal is None:
        return
    weekly_raw = cal._weekly_natural_segments
    for date_str, is_work in old_special_map.items():
        dt = date.fromisoformat(date_str)
        if dt in cal._special_natural_segments:
            logger.warning(f"特殊日期 {date_str} 已有配置，旧格式转换将覆盖原有班次定义")
        if is_work:
            wd = dt.weekday()
            seg_list = []
            if wd in weekly_raw:
                for name, s, e in weekly_raw[wd]:
                    seg_list.append(ShiftSegment(shift_name=name, start_hour=s, end_hour=e))
            cal._special_natural_segments[dt] = seg_list
        else:
            cal._special_natural_segments[dt] = []


def _generate_stable_op_id(job_id: int, business_op_id: str) -> int:
    """用 SHA256 生成跨进程稳定的工序ID"""
    unique_str = f"{job_id}:{business_op_id}"
    hash_bytes = hashlib.sha256(unique_str.encode()).digest()[:8]
    return int.from_bytes(hash_bytes, 'big')


def build_production_data_from_dict(state_manager: ProductionStateManager, data: dict):
    """
    从字典数据刷新 state_manager 的所有生产数据（适配新日历架构）
    """
    sm = state_manager

    # ========== 1. 工作日历（新架构） ==========
    cal_cfg = data.get("calendar_config")
    if cal_cfg:
        base_date = datetime.strptime(cal_cfg["base_date"], "%Y-%m-%d").date()
        # 检查是否需要重建日历
        if sm.work_calendar is None or sm.work_calendar.base_date != base_date:
            # 加载班次配置
            shift_configs = _load_shift_configs_from_json()
            if not shift_configs:
                shift_configs = data.get("shift_configs", [])
            if not shift_configs:
                shift_configs = _get_default_shift_configs()
            sm.load_shift_data_from_db(base_date=base_date, shift_configs=shift_configs)

    # ========== 2. 资源组（不变） ==========
    rg_data_list = data.get("resource_groups")
    if rg_data_list:
        for rg_data in rg_data_list:
            rg = ResourceGroup(
                group_id=rg_data["group_id"],
                group_name=rg_data["group_name"],
                machine_id_list=rg_data["machine_id_list"],
                worker_id_list=rg_data["worker_id_list"],
                worker_max_parallel=rg_data["worker_max_parallel"]
            )
            sm.resource_group_dict[rg.group_id] = rg

    # ========== 3. 机床状态（不变） ==========
    machine_dict = data.get("machine_dict")
    if machine_dict:
        for mid_str, m_info in machine_dict.items():
            mid = int(mid_str)
            if mid in sm.machine_meta_dict:
                sm.machine_meta_dict[mid].available = m_info["available"]
            else:
                changeover_raw = data.get("machine_changeover_map", {})
                change_map = {}
                for k, v in changeover_raw.items():
                    change_map[int(k)] = {int(ik): iv for ik, iv in v.items()}
                sm.machine_meta_dict[mid] = MachineMeta(
                    machine_id=mid,
                    available=m_info["available"],
                    planned_daily_hour=m_info.get("planned_daily_hour", 12.0),
                    changeover_time_map=change_map
                )

    # ========== 4. 工人状态（不变） ==========
    worker_dict = data.get("worker_dict")
    if worker_dict:
        for wid_str, w_info in worker_dict.items():
            wid = int(wid_str)
            rest_dates = [datetime.strptime(ds, "%Y-%m-%d").date() for ds in w_info.get("rest_day", [])]
            speed_raw = w_info.get("tech_speed_ratio", {})
            speed_conv = {int(k): v for k, v in speed_raw.items()}
            if wid in sm.worker_meta_dict:
                sm.worker_meta_dict[wid].available = w_info["available"]
                sm.worker_meta_dict[wid].rest_day = rest_dates
                sm.worker_meta_dict[wid].tech_speed_ratio = speed_conv
            else:
                sm.worker_meta_dict[wid] = WorkerMeta(
                    worker_id=wid,
                    available=w_info["available"],
                    rest_day=rest_dates,
                    tech_speed_ratio=speed_conv
                )

    # ========== 5. 订单工序状态（适配新日历） ==========
    job_list = data.get("job_list")
    if job_list:
        for job_data in job_list:
            jid = job_data["job_id"]
            op_info_list = job_data["op_info_list"]

            if jid not in sm.job_meta_dict:
                op_id_list = []
                for op_index, op_item in enumerate(op_info_list):
                    business_op_id = op_item["business_op_id"]
                    global_op_id = _generate_stable_op_id(jid, business_op_id)
                    op_id_list.append(global_op_id)

                    if global_op_id not in sm.op_meta_dict:
                        lock_dict = op_item.get("op_lock_info", {})
                        lock_time_str = lock_dict.get("lock_time")
                        lock_time = datetime.strptime(lock_dict["lock_time"], "%Y-%m-%d %H:%M:%S") if (lock_time_str and lock_time_str.strip()) else None
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
                            material_ready_time=op_item.get("material_ready_time", 0.0),
                            op_tech_type=op_item.get("op_tech_type", 0),
                            op_lock_info=lock_obj
                        )
                        sm.op_meta_dict[global_op_id] = op_meta
                        sm.operation_id_to_job_id[global_op_id] = jid

                base_weight = JOB_PRIORITY_WEIGHT.get(job_data["priority"], 1.0)
                delivery_raw = job_data.get("due_delivery_date")
                due_delivery_date = datetime.strptime(delivery_raw, "%Y-%m-%d").date() if delivery_raw else None
                due_delivery_time = 0.0
                if due_delivery_date and sm.work_calendar:
                    due_delivery_time = _get_day_end_relative_hour(sm, due_delivery_date)

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
            else:
                for op_index, op_item in enumerate(op_info_list):
                    business_op_id = op_item["business_op_id"]
                    global_op_id = _generate_stable_op_id(sm, jid, business_op_id)
                    if global_op_id in sm.op_meta_dict:
                        # 已存在仅刷新状态
                        sm.op_meta_dict[global_op_id].op_status = op_item["op_status"]
                    else:
                        # 订单已存在、当前是新增工序，完整新建OP元数据
                        lock_dict = op_item.get("op_lock_info", {})
                        lock_time_str = lock_dict.get("lock_time")
                        lock_time = datetime.strptime(lock_dict["lock_time"], "%Y-%m-%d %H:%M:%S") if (
                                    lock_time_str and lock_time_str.strip()) else None
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
                            material_ready_time=op_item.get("material_ready_time", 0.0),
                            op_tech_type=op_item.get("op_tech_type", 0),
                            op_lock_info=lock_obj
                        )
                        sm.op_meta_dict[global_op_id] = op_meta
                        sm.operation_id_to_job_id[global_op_id] = jid
                        if lock_obj.lock_machine or lock_obj.lock_worker:
                            sm.add_manual_lock(lock_obj)

    sm.operation_status_dict = {
        op_id: op_meta.op_status
        for op_id, op_meta in sm.op_meta_dict.items()
    }