from datetime import datetime
import config as cfg
from config.settings import JOB_PRIORITY_WEIGHT
from core.state_manager import ProductionStateManager
from core.data_structs import ResourceGroup, MachineTechParam, WorkerSkillInfo
from core.calendar import WorkCalendar
from core.data_structs import OperationMeta


def build_test_production_data(state_manager: ProductionStateManager):
    sm = state_manager
    sm.work_calendar = WorkCalendar(
        base_date=datetime(2026, 6, 15).date(),
        default_work_start=8.0,
        default_work_end=20.0
    )
    sm.work_calendar.date_work_map = cfg.DATE_WORK_MAP
    sm.work_calendar.set_week_rule([0, 1, 2, 3, 4])

    rg1 = ResourceGroup(group_id=1, group_name="数控车组", machine_id_list=[0, 1], worker_id_list=[10, 11],
                        worker_max_parallel=2)
    rg2 = ResourceGroup(group_id=2, group_name="铣削组", machine_id_list=[2, 3], worker_id_list=[12, 13],
                        worker_max_parallel=2)
    rg3 = ResourceGroup(group_id=3, group_name="磨床组", machine_id_list=[4], worker_id_list=[14, 15], worker_max_parallel=2)
    sm.resource_group_dict[1] = rg1
    sm.resource_group_dict[2] = rg2
    sm.resource_group_dict[3] = rg3

    change_map = {
        0: {0: 0, 1: 4.5, 2: 5.0},
        1: {0: 6.2, 1: 0, 2: 4.8},
        2: {0: 5.2, 1: 4.9, 2: 0}
    }
    for mid in [0, 1, 2, 3, 4]:
        tech_param = MachineTechParam(machine_id=mid, max_work_hour=120.0, changeover_time_map=change_map)
        sm.machine_tech_dict[mid] = tech_param

    worker_skill_list = [
        WorkerSkillInfo(worker_id=10, tech_speed_ratio={0: 1.0, 1: 0.85}),
        WorkerSkillInfo(worker_id=11, tech_speed_ratio={0: 1.1, 1: 0.9}),
        WorkerSkillInfo(worker_id=12, tech_speed_ratio={0: 0.95, 1: 1.05}),
        WorkerSkillInfo(worker_id=13, tech_speed_ratio={0: 1.2, 1: 0.8}),
        WorkerSkillInfo(worker_id=14, tech_speed_ratio={0: 1.0, 1: 1.0}),
    ]
    for skill in worker_skill_list:
        sm.worker_skill_dict[skill.worker_id] = skill

    sm.machine_available_dict = {
        0: True,
        1: True,
        2: True,
        3: True,
        4: True,
        5: True
    }

    sm.worker_available_dict = {
        10: True,
        11: True,
        12: True,
        13: True,
        14: True,
        15: True
    }

    job_info = [
        {
            "job_id": 1,
            "priority": "urgent",
            "quantity": 2,  # 新增：订单总投产数量2件
            "due_warn_time": 110.0,
            "due_contract_time": 120.0,
            "op_info_list": [
                {
                    "business_op_id": "OP1001",
                    "op_status": cfg.OP_STATUS_FINISHED,
                    "business_op_no": "5",
                    "op_name": "粗车外圆",
                    "op_content": "粗车工件外圆至φ50mm",
                    "resource_group_id": 1,
                    "resource_group_name": "数控车组",
                    "process_time": 12.0,
                    "op_tech_type": 0,
                    "op_quantity": 2  # 新增：本工序加工2件，和订单总量一致
                },
                {
                    "business_op_id": "OP1002",
                    "op_status": cfg.OP_STATUS_RUNNING,
                    "business_op_no": "10",
                    "op_name": "精车外圆",
                    "op_content": "精车外圆至φ49.8±0.02mm",
                    "resource_group_id": 1,
                    "resource_group_name": "数控车组",
                    "process_time": 8.0,
                    "op_tech_type": 1,
                    "op_quantity": 2
                },
                {
                    "business_op_id": "OP1003",
                    "op_status": cfg.OP_STATUS_OPTIMIZABLE,
                    "business_op_no": "15",
                    "op_name": "端面铣削",
                    "op_content": "铣削工件两端面保证总长100mm",
                    "resource_group_id": 2,
                    "resource_group_name": "铣削组",
                    "process_time": 15.0,
                    "op_tech_type": 2,
                    "op_quantity": 2
                },
            ]
        },
        {
            "job_id": 2,
            "priority": "high",
            "quantity": 3,  # 订单总产量3
            "due_warn_time": 160.0,
            "due_contract_time": 180.0,
            "op_info_list": [
                {
                    "business_op_id": "OP2001", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "5", "op_name": "平面铣削",
                    "op_content": "铣削基准平面",
                    "resource_group_id": 2, "resource_group_name": "铣削组", "process_time": 10.0, "op_tech_type": 2,
                    "op_quantity": 3
                },
                {"business_op_id": "OP2002", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "10", "op_name": "外圆磨削",
                 "op_content": "磨削外圆至φ30±0.01mm",
                 "resource_group_id": 3,"resource_group_name": "磨床组", "process_time": 18.0, "op_tech_type": 0,
                 "op_quantity": 3},
                {"business_op_id": "OP2003", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "15", "op_name": "键槽铣削",
                 "op_content": "铣削8mm宽键槽",
                 "resource_group_id": 2, "resource_group_name": "铣削组", "process_time": 12.0, "op_tech_type": 2,
                 "op_quantity": 3},
                {"business_op_id": "OP2004", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "20", "op_name": "钻孔",
                 "op_content": "钻φ10mm通孔",
                 "resource_group_id": 1, "resource_group_name": "数控车组", "process_time": 6.0, "op_tech_type": 0,
                 "op_quantity": 3},
                {"business_op_id": "OP2005", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "25", "op_name": "内孔磨削",
                 "op_content": "磨削内孔至φ20±0.01mm",
                 "resource_group_id": 3, "resource_group_name": "磨床组", "process_time": 16.0, "op_tech_type": 0,
                 "op_quantity": 3},
            ]
        },
        {
            "job_id": 3,
            "priority": "normal",
            "quantity": 4,
            "due_warn_time": 185.0,
            "due_contract_time": 200.0,
            "op_info_list": [
                {"business_op_id": "OP3001", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "5", "op_name": "粗车毛坯",
                 "op_content": "粗车毛坯去除余量",
                 "resource_group_id": 1, "resource_group_name": "数控车组", "process_time": 14.0, "op_tech_type": 0,
                 "op_quantity": 4},
                {"business_op_id": "OP3002", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "10", "op_name": "精铣成型面",
                 "op_content": "精铣复杂成型面",
                 "resource_group_id": 2, "resource_group_name": "铣削组", "process_time": 16.0, "op_tech_type": 2,
                 "op_quantity": 4},
            ]
        },
        {
            "job_id": 4,
            "priority": "low",
            "quantity": 2,
            "due_warn_time": 260.0,
            "due_contract_time": 300.0,
            "op_info_list": [
                {"business_op_id": "OP4001", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "5", "op_name": "平面磨削",
                 "op_content": "磨削基准平面保证平面度0.02",
                 "resource_group_id": 3, "resource_group_name": "磨床组", "process_time": 12.0, "op_tech_type": 0,
                 "op_quantity": 2},
                {"business_op_id": "OP4002", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "10", "op_name": "车外螺纹",
                 "op_content": "车M20×2外螺纹",
                 "resource_group_id": 1, "resource_group_name": "数控车组", "process_time": 10.0, "op_tech_type": 1,
                 "op_quantity": 2},
                {"business_op_id": "OP4003", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "15", "op_name": "铣退刀槽",
                 "op_content": "铣削6mm宽退刀槽",
                 "resource_group_id": 2, "resource_group_name": "铣削组", "process_time": 8.0, "op_tech_type": 2,
                 "op_quantity": 2},
                {"business_op_id": "OP4004", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "20", "op_name": "内孔磨削",
                 "op_content": "磨削内孔至φ15±0.01mm",
                 "resource_group_id": 3, "resource_group_name": "磨床组", "process_time": 14.0, "op_tech_type": 0,
                 "op_quantity": 2},
            ]
        },
        {
            "job_id": 5,
            "priority": "high",
            "quantity": 3,
            "due_warn_time": 135.0,
            "due_contract_time": 150.0,
            "op_info_list": [
                {"business_op_id": "OP5001", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "5", "op_name": "铣上下平面",
                 "op_content": "铣削上下平面保证平行度0.03",
                 "resource_group_id": 2, "resource_group_name": "铣削组", "process_time": 11.0, "op_tech_type": 2,
                 "op_quantity": 3},
                {"business_op_id": "OP5002", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "10", "op_name": "外圆磨削",
                 "op_content": "磨削外圆至φ40±0.01mm",
                 "resource_group_id": 3, "resource_group_name": "磨床组", "process_time": 15.0, "op_tech_type": 0,
                 "op_quantity": 3},
                {"business_op_id": "OP5003", "op_status": cfg.OP_STATUS_OPTIMIZABLE,"business_op_no": "15", "op_name": "精车端面",
                 "op_content": "精车端面保证总长80mm",
                 "resource_group_id": 1, "resource_group_name": "数控车组", "process_time": 9.0, "op_tech_type": 1,
                 "op_quantity": 3},
            ]
        },
    ]

    global_op_id = 0
    all_job_op_map = {}
    for job_data in job_info:
        jid = job_data["job_id"]
        op_info_list = job_data["op_info_list"]
        op_id_list = []
        for op_index, op_item in enumerate(op_info_list):
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
                material_ready_time=op_item.get("material_ready_time", 0.0),
                op_tech_type=op_item.get("op_tech_type", 0)
            )
            sm.op_meta_dict[global_op_id] = op_meta
            sm.operation_id_to_job_id[global_op_id] = jid
            op_id_list.append(global_op_id)
            global_op_id += 1
        all_job_op_map[jid] = op_id_list
        base_weight = JOB_PRIORITY_WEIGHT[job_data["priority"]]
        from core.data_structs import JobMeta
        job_meta = JobMeta(
            job_id=jid,
            op_id_list=op_id_list,
            priority=job_data["priority"],
            due_warn_time=job_data["due_warn_time"],
            due_contract_time=job_data["due_contract_time"],
            quantity=job_data["quantity"],
            base_weight=base_weight
        )
        sm.job_meta_dict[jid] = job_meta

    total_global_op = global_op_id

    sm.operation_status_dict = {
        op_id: op_meta.op_status
        for op_id, op_meta in sm.op_meta_dict.items()
    }

    print(f"业务格式数据集构建完成｜总订单{len(sm.job_meta_dict)} 总工序{total_global_op}")
    return all_job_op_map
