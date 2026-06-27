# main.py
import numpy as np
from core.state_manager import ProductionStateManager
from data.test_dataset import build_test_production_data
from trigger.rolling_trigger import RollingScheduleTrigger
from core.data_structs import ManualLockAssign
from core.nsga2_operator import nsga2_rolling_schedule
import core.base_ga as base_ga
from visual.plot_gantt import plot_pareto_front, plot_machine_gantt, plot_worker_gantt, plot_operation_gantt
from utils.log_utils import get_logger
import config as cfg
from datetime import timedelta, datetime, date
from config import settings
import json


# 全局日志初始化（仅在main.py执行一次）
logger = get_logger(__name__)


if __name__ == "__main__":
    # ============================================================
    # 【初始化滚动排程】加载test_data1.json数据
    # ============================================================
    np.random.seed(40)
    sm = ProductionStateManager()
    all_job_op_map = build_test_production_data(sm, "test_data.json")
    all_job_ids = list(all_job_op_map.keys())
    trigger = RollingScheduleTrigger(sm)

    print("\n" + "=" * 70)
    print("【第一次全量调度】系统初始时间：0.0小时")
    print("=" * 70)

    # 强制从下一个工作日白班开始排程
    ref_dt = datetime(2026, 6, 24, 10, 0)
    next_workday_start = sm.get_next_workday_start_time(current_datetime=ref_dt)
    sm.set_system_time(next_workday_start)

    print(f"get_next_workday_start_time返回值: {next_workday_start}")
    print(f'next_start_datetime:{sm.relative_hour_from_base_to_first_start_datetime(next_workday_start)}')
    print(f'_base_work_zero_datetime{sm.work_calendar._base_work_zero_datetime}')
    print(f"对应实际日期: {sm.relative_hour_from_base_to_real_datetime(next_workday_start)}")
    print(f"2026-06-24是否为工作日: {sm.work_calendar.is_workday(date(2026, 6, 24))}")


    pareto_set, final_fits, pareto_idx_list = nsga2_rolling_schedule(sm, all_job_ids)

    print(f"\n帕累托最优解集数量：{len(pareto_set)}")
    weight = [0.30, 0.10, 0.20, 0.20, 0.10, 0.05, 0.05]
    best_chrom, best_fit = base_ga.select_optimal_solution_by_weight(pareto_set, final_fits, pareto_idx_list, weight)

    target_name = [
        "逾期订单总数",
        "订单逾期总惩罚成本",
        "最大完工时间",
        "设备整体闲置率",
        "设备负荷不均衡度",
        "人员负荷不均衡度",
        "加权在制品等待总时长"
    ]
    print(f"\n【最优方案多维指标】")
    for name, val in zip(target_name, best_fit):
        print(f"  {name}: {val:.2f}")

    _, schedule_detail = base_ga.decode_chromosome(best_chrom, sm)
    sm.cache_schedule_result(schedule_detail)

    print(f"\n【排产明细前10条】")
    for item in schedule_detail[:10]:
        print(
            f"工序{item['op_id']:2d} | 订单{item['job_id']:d}-{item['business_op_no']:2s} | "
            f"机床{item['machine_id']:d} 工人{item['worker_id']:d} | "
            f"开始{item['start_time']:5.1f} 结束{item['end_time']:5.1f} | "
            f"锁定:{str(item['is_manual_locked']):5s} 冻结:{str(item['is_frozen']):5s}"
        )

    print("排程结果所有工序的start_time:", [round(item["start_time"], 1) for item in schedule_detail])
    print("当前系统时间:", sm.current_system_time)

    print("\n正在生成初始调度图表...")
    plot_pareto_front([final_fits[i] for i in pareto_idx_list])
    plot_machine_gantt(schedule_detail, sm)
    plot_worker_gantt(schedule_detail, sm)
    plot_operation_gantt(schedule_detail, sm)


'''
    # ============================================================
    # 【第二次滚动排程】模拟生产推进了 13 小时，加载test_data3.json数据
    # ============================================================
    print("\n" + "=" * 70)
    print("【第二次滚动排程】模拟生产已推进 13.0 小时")
    print("=" * 70)

    sm.advance_system_time(12.0)

    new_all_job_op_map = build_test_production_data(sm, json_path="test_data3.json")
    new_all_job_ids = list(new_all_job_op_map.keys())

    print(f"\n新数据集订单数：{len(new_all_job_ids)}，工序总数：{len(sm.op_meta_dict)}")
    print(f"上次排程缓存工序数：{len(sm.last_schedule_result)}")

    pareto_set_2, final_fits_2, pareto_idx_list_2 = nsga2_rolling_schedule(sm, new_all_job_ids)

    print(f"\n第二次排程 - 帕累托最优解集数量：{len(pareto_set_2)}")
    best_chrom_2, best_fit_2 = base_ga.select_optimal_solution_by_weight(
        pareto_set_2, final_fits_2, pareto_idx_list_2, weight
    )

    print(f"\n【第二次排程 - 最优方案多维指标】")
    for name, val in zip(target_name, best_fit_2):
        print(f"  {name}: {val:.2f}")

    _, schedule_detail_2 = base_ga.decode_chromosome(best_chrom_2, sm)
    sm.cache_schedule_result(schedule_detail_2)

    frozen_count = sum(1 for item in schedule_detail_2 if item["is_frozen"])
    changed_count = len(schedule_detail_2) - frozen_count
    print(f"\n【滚动排程对比】")
    print(f"  总工序数：{len(schedule_detail_2)}")
    print(f"  冻结工序（沿用上次）：{frozen_count}")
    print(f"  重排工序：{changed_count}")

    # 打印冻结/重排明细（前30条）
    print(f"\n【第二次排产明细 - 含冻结标记】")
    for item in schedule_detail_2[:30]:
        # 判断状态标签
        if item["is_frozen"]:
            tag = "🔒冻结"
        elif item["is_manual_locked"]:
            tag = "🔧锁定"
        else:
            tag = "🔄重排"

        # 获取工序状态
        op_status = item.get("op_status", -1)
        status_icon = cfg.OP_STATUS_MAP.get(op_status, "❓未知")

        print(
            f"{tag} {status_icon} | 工序{item['op_id']} | "
            f"订单{item['job_id']}-{item['business_op_no']:>3s} | "
            f"机床{item['machine_id']} 工人{item['worker_id']} | "
            f"开始{item['start_time']:6.1f} 结束{item['end_time']:6.1f}"
        )

    print("\n正在生成滚动排程图表...")
    plot_pareto_front([final_fits_2[i] for i in pareto_idx_list_2])
    plot_machine_gantt(schedule_detail_2, sm)
    plot_worker_gantt(schedule_detail_2, sm)
    plot_operation_gantt(schedule_detail_2, sm)


    # 插单演示代码（放开注释即可运行）
    # print("\n" + "=" * 70)
    # print("【紧急插单重排】模拟生产已进行12小时")
    # print("=" * 70)
    # sm.advance_system_time(12.0)
    # insert_job_config = {
    #     "job_id": 6,
    #     "priority": "urgent",
    #     "warn_due": 95.0,
    #     "due_contract_time": 108.0,
    #     "base_weight": JOB_PRIORITY_WEIGHT["urgent"],
    #     "op_info_list": [
    #         {
    #             "business_op_id": "OP6001",
    #             "business_op_no": "5",
    #             "op_name": "铣上下平面",
    #             "op_content": "铣削上下平面保证平行度0.03",
    #             "resource_group_id": 2,
    #             "resource_group_name": "铣削组",
    #             "process_time": 11.0,
    #             "op_tech_type": 2
    #         },
    #         {
    #             "business_op_id": "OP6002",
    #             "business_op_no": "10",
    #             "op_name": "外圆磨削",
    #             "op_content": "磨削外圆至φ40±0.01mm",
    #             "resource_group_id": 3,
    #             "resource_group_name": "磨床组",
    #             "process_time": 15.0,
    #             "op_tech_type": 0
    #         },
    #         {
    #             "business_op_id": "OP6003",
    #             "business_op_no": "15",
    #             "op_name": "精车端面",
    #             "op_content": "精车端面保证总长80mm",
    #             "resource_group_id": 1,
    #             "resource_group_name": "数控车组",
    #             "process_time": 9.0,
    #             "op_tech_type": 1
    #         },
    #     ]
    # }
    # pareto_set_new, final_fits_new, pareto_idx_new = trigger.trigger_by_event("new_order", insert_job_config)
    # if pareto_set_new is not None and len(pareto_set_new) > 0:
    #     best_chrom_new, best_fit_new = select_optimal_solution(pareto_set_new, final_fits_new, pareto_idx_new)
    #     print(f"\n【插单后最优方案核心指标】")
    #     print(f"  最大完工时间: {best_fit_new[0]:.2f} 小时")
    #     print(f"  订单加权逾期成本: {best_fit_new[1]:.2f}")
    #     _, new_schedule_detail = decode_chromosome(best_chrom_new, sm)
    #     sm.cache_schedule_result(new_schedule_detail)
    #     plot_pareto_front([final_fits_new[i] for i in pareto_idx_new])
    #     plot_machine_gantt(new_schedule_detail, sm)
    #     plot_worker_gantt(new_schedule_detail, sm)
    #     plot_operation_gantt(new_schedule_detail, sm)
    # else:
    #     print("插单重调度无有效帕累托解集")
'''


def init_calendar(state_manager: ProductionStateManager):
    # 根据配置加载班次数据
    if settings.SHIFT_DATA_SOURCE == 'json_file':
        with open(settings.SHIFT_CONFIG_FILE, 'r', encoding='utf-8') as f:
            shift_configs = json.load(f)
    else:
        # 从数据库加载（需自行实现查询）
        # shift_configs = fetch_from_db(settings.SHIFT_CONFIG_TABLE)
        raise NotImplementedError("数据库班次加载方式需自行实现")

    state_manager.load_shift_data_from_db(
        base_date=settings.BASE_DATE,
        shift_configs=shift_configs
    )