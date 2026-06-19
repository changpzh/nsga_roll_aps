import numpy as np
from core.state_manager import ProductionStateManager
from data.test_dataset import build_test_production_data
from trigger.rolling_trigger import RollingScheduleTrigger
from core.data_structs import ManualLockAssign
from core.nsga2_operator import nsga2_rolling_schedule
import core.base_ga as base_ga
from visual.plot_gantt import plot_pareto_front, plot_machine_gantt, plot_worker_gantt, plot_operation_gantt
from utils.log_utils import get_logger

# 全局日志初始化（仅在main.py执行一次）
logger = get_logger(__name__)


if __name__ == "__main__":
    np.random.seed(40)
    sm = ProductionStateManager()
    all_job_op_map = build_test_production_data(sm)
    all_job_ids = list(all_job_op_map.keys())

    db_lock_data = sm.export_all_manual_lock()
    print("人工锁定配置可入库数据样例：", db_lock_data[:1])

    trigger = RollingScheduleTrigger(sm)

    print("\n" + "=" * 70)
    print("【第一次全量调度】系统初始时间：0.0小时")
    print("=" * 70)

    sm.set_system_time(0.0)
    pareto_set, final_fits, pareto_idx_list = nsga2_rolling_schedule(sm, all_job_ids)
    # # 替换为NSGA3调用
    # pareto, fits, idx = nsga3_rolling_schedule(state_manager, reorder_job_seq, divisions=3)

    print(f"\n帕累托最优解集数量：{len(pareto_set)}")

    best_chrom, best_fit = base_ga.select_optimal_solution(pareto_set, final_fits, pareto_idx_list)

    target_name = [
        "逾期订单总数",
        "订单逾期总惩罚成本",
        "最大完工时间(含超负荷惩罚)",
        "设备整体闲置率",
        "设备总换型时间",
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

    print("\n正在生成初始调度图表...")
    plot_pareto_front([final_fits[i] for i in pareto_idx_list])
    plot_machine_gantt(schedule_detail, sm)
    plot_worker_gantt(schedule_detail, sm)
    plot_operation_gantt(schedule_detail, sm)

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