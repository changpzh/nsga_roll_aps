from datetime import datetime, date
import json
import os
import config as cfg
from config.settings import JOB_PRIORITY_WEIGHT
from core.state_manager import ProductionStateManager
from core.data_structs import ResourceGroup, MachineMeta, WorkerMeta
from core.calendar import WorkCalendar
from core.data_structs import OperationMeta, JobMeta, ManualLockAssign


def load_production_json(json_path: str = "data/test_data2.json") -> dict:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接json完整路径
    full_path = os.path.join(current_file_dir, json_path)
    print(f"full_path: {full_path} ")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"数据文件不存在: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_test_production_data(state_manager: ProductionStateManager, json_path: str = "test_data2.json"):
    sm = state_manager
    data = load_production_json(json_path)

    # 1、工作日历
    cal_cfg = data["calendar_config"]
    base_date = datetime.strptime(cal_cfg["base_date"], "%Y-%m-%d").date()
    sm.work_calendar = WorkCalendar(
        base_date=base_date,
        default_daily_work_start=cal_cfg["default_daily_work_start"],
        default_daily_work_end=cal_cfg["default_daily_work_end"]
    )
    sm.work_calendar.special_date_work_map = cfg.DATE_WORK_MAP
    sm.work_calendar.set_week_rule(cal_cfg["week_work_rule"])
    sm.work_calendar.add_calendar_item(date.fromisoformat("2026-06-22"),True, 8.0, 12.0)
    sm.work_calendar.add_calendar_item(date.fromisoformat("2026-06-22"), True, 14.0, 20.0)

    # 2、资源组
    for rg_data in data["resource_groups"]:
        rg = ResourceGroup(
            group_id=rg_data["group_id"],
            group_name=rg_data["group_name"],
            machine_id_list=rg_data["machine_id_list"],
            worker_id_list=rg_data["worker_id_list"],
            worker_max_parallel=rg_data["worker_max_parallel"]
        )
        sm.resource_group_dict[rg.group_id] = rg

    # 3、机床换型时间映射,将字符型id转换为数字型id
    changeover_raw = data["machine_changeover_map"]
    change_map = {}
    for mid_str, inner in changeover_raw.items():
        mid = int(mid_str)
        inner_conv = {int(k): v for k, v in inner.items()}
        change_map[mid] = inner_conv

    # 4、机床信息
    sm.machine_meta_dict.clear()
    for mid_str, m_info in data["machine_dict"].items():
        mid = int(mid_str)
        sm.machine_meta_dict[mid] = MachineMeta(
            machine_id=mid,
            available=m_info["available"],
            planned_daily_hour=m_info["planned_daily_hour"],
            changeover_time_map=change_map
        )

    # 5、工人信息：构建 WorkerMeta
    sm.worker_meta_dict.clear()
    for wid_str, w_info in data["worker_dict"].items():
        wid = int(wid_str)
        # 休息日转date列表
        rest_dates = [datetime.strptime(ds, "%Y-%m-%d").date() for ds in w_info["rest_day"]]
        # 工艺速度倍率key转int
        speed_raw = w_info["tech_speed_ratio"]
        speed_conv = {int(k): v for k, v in speed_raw.items()}

        w_meta = WorkerMeta(
            worker_id=wid,
            available=w_info["available"],
            rest_day=rest_dates,
            tech_speed_ratio=speed_conv
        )
        sm.worker_meta_dict[wid] = w_meta

    # 6、订单+工序
    global_op_id = 0
    all_job_op_map = {}
    sm.op_meta_dict.clear()
    sm.operation_id_to_job_id.clear()
    sm.job_meta_dict.clear()

    for job_data in data["job_list"]:
        jid = job_data["job_id"]
        op_info_list = job_data["op_info_list"]
        op_id_list = []

        for op_index, op_item in enumerate(op_info_list):
            # 解析锁定信息为 ManualLockAssign 对象
            lock_dict = op_item["op_lock_info"]
            lock_time_str = lock_dict.get("lock_time")
            lock_time = datetime.strptime(lock_dict["lock_time"], "%Y-%m-%d %H:%M:%S") if (lock_time_str and lock_time_str.strip()) else None
            lock_obj = ManualLockAssign(
                op_global_id=global_op_id,
                business_op_id=op_item["business_op_id"],
                business_op_no=op_item["business_op_no"],
                fixed_machine_id=lock_dict["fixed_machine_id"],
                fixed_worker_id=lock_dict["fixed_worker_id"],
                lock_machine=lock_dict["lock_machine"],
                lock_worker=lock_dict["lock_worker"],
                operator=lock_dict["operator"],
                lock_reason=lock_dict["lock_reason"],
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
            global_op_id += 1

        all_job_op_map[jid] = op_id_list
        base_weight = JOB_PRIORITY_WEIGHT[job_data["priority"]]

        # 交货期解析，兼容null
        delivery_raw = job_data.get("due_delivery_date")
        due_delivery_date = None
        if delivery_raw is not None:
            due_delivery_date = datetime.strptime(delivery_raw, "%Y-%m-%d").date()
        # 通过交货期日期，算出due_contract_time
        due_delivery_time = 0.0
        if due_delivery_date is not None:
            due_delivery_datetime = datetime.combine(due_delivery_date, datetime.min.time()).replace(hour=int(state_manager.work_calendar.daily_work_end))
            due_delivery_time = state_manager.get_datetime_to_relative_hours(due_delivery_datetime)
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

    print(f"JSON数据集加载构建完成｜总订单{len(sm.job_meta_dict)} 总工序{global_op_id}")
    print(f"all_job_op_map: {all_job_op_map}")
    return all_job_op_map