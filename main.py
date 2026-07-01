import numpy as np
import random
from core.state_manager import ProductionStateManager
from data.test_dataset import build_test_production_data
from trigger.rolling_trigger import RollingScheduleTrigger
from core.nsga2_operator import nsga2_rolling_schedule
import core.base_ga as base_ga
from visual.plot_gantt import plot_pareto_front, plot_machine_gantt, plot_worker_gantt, plot_operation_gantt, print_topsis_sorted_pareto_table
from utils.log_utils import get_logger
from datetime import timedelta, datetime, date
from core.multi_criteria_decision import TopsisAllMinEvaluator


logger = get_logger(__name__)


if __name__ == "__main__":
    SEED = 60
    random.seed(SEED)
    np.random.seed(SEED)
    sm = ProductionStateManager()
    current_weight = [0.45, 0.10, 0.20, 0.10, 0.05, 0.05, 0.05]
    sm.topsis_weight = current_weight
    topsis_evaluator = TopsisAllMinEvaluator(decimal_reserve=6)
    all_job_op_map = build_test_production_data(sm, "test_data3.json")
    all_job_ids = list(all_job_op_map.keys())
    trigger = RollingScheduleTrigger(sm)

    print("\n" + "=" * 70)
    print("【第一次全量调度】系统初始时间：基准零点")
    print("=" * 70)

    ref_dt = datetime(2026, 7, 1, 10, 0)
    # ref_dt = datetime.now()
    next_workday_start = sm.get_next_workday_start_time(current_datetime=ref_dt)
    sm.set_system_time(next_workday_start)

    pareto_set, final_fits, pareto_idx_list = nsga2_rolling_schedule(sm, all_job_ids)

    print(f"\n帕累托最优解集数量：{len(pareto_set)}")

    best_chrom, best_fit, sorted_pareto_list, sorted_fit_list= base_ga.select_optimal_solution_by_weight(
        pareto_set=pareto_set,
        all_pop_fits=final_fits,
        pareto_index_list=pareto_idx_list,
        weight=sm.topsis_weight,
        topsis_evaluator=topsis_evaluator
    )
    print_topsis_sorted_pareto_table(
        sorted_pareto_list=sorted_pareto_list,
        sorted_fit_list=sorted_fit_list
    )

    target_name = [
        "逾期订单总数",
        "订单逾期总惩罚成本",
        "最大完工时间",
        "设备整体闲置率",
        "设备负荷不均衡度",
        "人员负荷不均衡度",
        "加权在制品等待总时长"
    ]
    print("\n" + "=" * 130)
    print(f"{'【全部帕累托解集 + TOPSIS综合排名明细】':^130}")
    print("=" * 130)



    print(f"\n【最优方案多维指标】")
    for name, val in zip(target_name, best_fit):
        print(f"  {name}: {val:.2f}")

    _, best_schedule_detail = base_ga.decode_chromosome(best_chrom, sm)
    sm.cache_schedule_result(best_schedule_detail)

    print(f"\n【排产明细】")
    for item in best_schedule_detail:
        print(
            f"工序{item['op_id']:2d} | 订单{item['job_id']}-{item['business_op_no']:2s} | "
            f"机床{item['machine_id']:d} 工人{item['worker_id']:d} | "
            f"开始{item['start_time'].isoformat()} 结束{item['end_time'].isoformat()} | "
            f"锁定:{str(item['is_manual_locked']):5s} 冻结:{str(item['is_frozen']):5s}"
        )

    print("排程结果所有工序的start_time:", [item["start_time"].isoformat() for item in best_schedule_detail])
    print("当前系统时间:", sm.current_system_time.isoformat())

    print("\n正在生成初始调度图表...")
    plot_pareto_front([final_fits[i] for i in pareto_idx_list])
    plot_machine_gantt(best_schedule_detail, sm)
    plot_worker_gantt(best_schedule_detail, sm)
    plot_operation_gantt(best_schedule_detail, sm)