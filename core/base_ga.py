import numpy as np
import heapq
import copy
from typing import List, Dict, Tuple, Any,Deque
from datetime import datetime, timedelta
from collections import deque
from models.base_types import MachineId, WorkerId, OperationId, JobId
from models.scheduling_state import SchedulingTrackers, OperationSchedulingResult
from core.state_manager import ProductionStateManager
from utils.common_utils import select_best_worker, select_best_machine
import config.settings as cfg
from config.settings import (
    WORKER_SWITCH_COST, WIP_WEIGHT_COEFFICIENT, POPULATION_SIZE,
    ROLLING_HISTORY_SEED_RATIO, ROLLING_HEURISTIC_SEED_RATIO, ROLLING_PERTURB_RATE,
    PLAN_FROZEN_HORIZON_HOURS
)
from core.multi_criteria_decision import TopsisAllMinEvaluator

import time
from utils.log_utils import get_logger

logger = get_logger(__name__)

class ConvergenceMonitor:
    def __init__(
        self,
        window_size: int = 10,
        rel_tol: float = 0.005
    ):
        self.window_size = window_size
        self.rel_tol = rel_tol
        self.history_window: Deque[List[float]] = deque(maxlen=window_size)
        self.full_history: List[List[float]] = []

    def add_generation_best(self, fit_vec: List[float]):
        self.history_window.append(fit_vec.copy())
        self.full_history.append(fit_vec.copy())

    def is_converged(self) -> bool:
        if len(self.history_window) < self.window_size:
            return False
        arr = np.array(self.history_window)
        mean_vals = np.mean(arr, axis=0)
        max_vals = np.max(arr, axis=0)
        min_vals = np.min(arr, axis=0)

        for idx in range(arr.shape[1]):
            mean_val = mean_vals[idx]
            if abs(mean_val) < 1e-9:
                continue
            relative_range = (max_vals[idx] - min_vals[idx]) / abs(mean_val)
            if relative_range > self.rel_tol:
                return False
        return True

    def get_history(self) -> List[List[float]]:
        return self.full_history

def init_single_chromosome(reorder_job_seq: List[int], state_manager: ProductionStateManager, shuffle_free: bool = True) -> dict:
    target_ops = []
    all_optimizable_ops_set = set(state_manager.get_optimizable_operation_ids())

    for job_id in reorder_job_seq:
        job_op_list = [op for op, jid in state_manager.operation_id_to_job_id.items() if jid == job_id]
        valid_ops = [op for op in job_op_list if op in all_optimizable_ops_set]
        valid_ops.sort(key=lambda x: int(state_manager.op_meta_dict[x].business_op_no))
        target_ops.extend(valid_ops)

    frozen_boundary = state_manager.current_system_time + timedelta(hours=PLAN_FROZEN_HORIZON_HOURS)
    frozen_ops = []
    unfrozen_old_ops = []
    new_ops = []

    for op_id in target_ops:
        if op_id in state_manager.last_schedule_result:
            last_start = state_manager.last_schedule_result[op_id]["start_time"]
            if last_start < frozen_boundary:
                frozen_ops.append(op_id)
            else:
                unfrozen_old_ops.append(op_id)
        else:
            new_ops.append(op_id)

    frozen_ops.sort(key=lambda x: state_manager.last_schedule_result[x]["start_time"])
    free_ops = unfrozen_old_ops + new_ops

    job_op_map: Dict[int, List[int]] = {}
    for op_id in free_ops:
        jid = state_manager.operation_id_to_job_id[op_id]
        if jid not in job_op_map:
            job_op_map[jid] = []
        job_op_map[jid].append(op_id)

    job_id_list = list(job_op_map.keys())
    if shuffle_free:
        np.random.shuffle(job_id_list)

    free_sequence = []
    for jid in job_id_list:
        free_sequence.extend(job_op_map[jid])

    op_sequence = frozen_ops + free_sequence

    job_prev_machine_map: Dict[int, MachineId] = {}
    resource_assign = []
    for op_id in op_sequence:
        if op_id in frozen_ops:
            mid = state_manager.last_schedule_result[op_id]["machine_id"]
            wid = state_manager.last_schedule_result[op_id]["worker_id"]
            resource_assign.append((mid, wid))
            job_prev_machine_map[state_manager.op_meta_dict[op_id].belong_job_id] = mid
            continue

        selected_machine_id, selected_worker_id = _select_machine_and_worker(
            op_id, op_sequence, resource_assign, state_manager
        )
        resource_assign.append((selected_machine_id, selected_worker_id))

    op_sequence, resource_assign = deduplicate_sequence(op_sequence, resource_assign)

    return {"op_sequence": op_sequence, "resource_assign": resource_assign}


def init_mixed_population(reorder_job_seq: List[int], state_manager: ProductionStateManager) -> List[dict]:
    pop_size = POPULATION_SIZE
    history_num = int(pop_size * ROLLING_HISTORY_SEED_RATIO)
    heuristic_num = int(pop_size * ROLLING_HEURISTIC_SEED_RATIO)
    random_num = pop_size - history_num - heuristic_num
    population = []

    old_job_set = set()
    for chrom in state_manager.last_pareto_solutions:
        for op in chrom["op_sequence"]:
            job_id = state_manager.operation_id_to_job_id[op]
            old_job_set.add(job_id)
    new_job_ratio = state_manager.get_new_job_ratio(old_job_set)
    if new_job_ratio > 0.3:
        history_num = int(pop_size * 0.3)
        heuristic_num = int(pop_size * 0.25)
        random_num = pop_size - history_num - heuristic_num

    history_candidates = state_manager.last_pareto_solutions
    if len(history_candidates) > 0:
        for _ in range(history_num):
            base_chrom = copy.deepcopy(np.random.choice(history_candidates))
            trimmed_chrom = trim_historical_chromosome(base_chrom, state_manager)
            new_history_seed = perturb_historical_chromosome(trimmed_chrom, state_manager)
            population.append(new_history_seed)
    else:
        for _ in range(history_num):
            random_chrom = init_single_chromosome(reorder_job_seq, state_manager)
            population.append(random_chrom)

    for _ in range(heuristic_num):
        heuristic_chromosome = generate_heuristic_chromosome(reorder_job_seq, state_manager)
        population.append(heuristic_chromosome)

    for _ in range(random_num):
        random_chrom = init_single_chromosome(reorder_job_seq, state_manager)
        population.append(random_chrom)

    return population


def perturb_historical_chromosome(trimmed_chrom: dict, state_manager: ProductionStateManager) -> dict:
    chrom = copy.deepcopy(trimmed_chrom)
    opt_ops = set(state_manager.get_optimizable_operation_ids())
    op_seq = chrom["op_sequence"]
    assign = chrom["resource_assign"]
    total_op_len = len(op_seq)

    freeze_boundary = state_manager.current_system_time + timedelta(hours=PLAN_FROZEN_HORIZON_HOURS)
    last_schedule = state_manager.last_schedule_result

    perturb_num = max(1, int(total_op_len * ROLLING_PERTURB_RATE))
    perturb_indexes = np.random.choice(range(total_op_len), perturb_num, replace=False)
    for idx in perturb_indexes:
        op_id = op_seq[idx]
        if (op_id not in opt_ops
            or state_manager.is_op_manual_locked(op_id))\
            or (op_id in last_schedule and last_schedule[op_id]["start_time"] < freeze_boundary):
            continue
        jid = state_manager.operation_id_to_job_id[op_id]
        job_meta = state_manager.job_meta_dict[jid]
        if job_meta.priority in ["high", "urgent"]:
            if np.random.random() > 0.4:
                continue
        new_machine, new_worker = _random_select_resources(op_id, state_manager)
        assign[idx] = (new_machine, new_worker)
    chrom["resource_assign"] = assign
    return chrom


def generate_heuristic_chromosome(reorder_job_seq: List[int], state_manager: ProductionStateManager) -> dict:
    job_sort_info = []
    for j_id in reorder_job_seq:
        meta = state_manager.job_meta_dict[j_id]
        job_sort_info.append((-meta.base_weight, meta.due_delivery_time, j_id))
    job_sort_info.sort()
    sorted_job_ids = [item[2] for item in job_sort_info]
    return init_single_chromosome(sorted_job_ids, state_manager, shuffle_free=False)


def trim_historical_chromosome(old_chrom: dict, state_manager: ProductionStateManager) -> dict:
    new_chrom = copy.deepcopy(old_chrom)
    opt_ops = set(state_manager.get_optimizable_operation_ids())
    new_op_sequence = []
    new_resource_assign = []
    old_seq = old_chrom["op_sequence"]
    old_assign = old_chrom["resource_assign"]

    op_to_pos = {op: idx for idx, op in enumerate(old_seq)}
    processed_ops = set()

    freeze_boundary = state_manager.current_system_time + timedelta(hours=PLAN_FROZEN_HORIZON_HOURS)
    last_schedule = state_manager.last_schedule_result

    for op in old_seq:
        status = state_manager.operation_status_dict.get(op, -1)
        if status in [cfg.OP_STATUS_FINISHED, cfg.OP_STATUS_RUNNING]:
            processed_ops.add(op)
            continue
        if op in last_schedule:
            op_start_time = last_schedule[op]["start_time"]
            if op_start_time < freeze_boundary:
                new_op_sequence.append(op)
                new_resource_assign.append(old_assign[op_to_pos[op]])
                processed_ops.add(op)

    for op in old_seq:
        if op in processed_ops or op not in opt_ops:
            continue
        if state_manager.is_op_manual_locked(op):
            lock_cfg = state_manager.get_lock_info(op)
            machine = lock_cfg.fixed_machine_id if lock_cfg.lock_machine else old_assign[op_to_pos[op]][0]
            worker = lock_cfg.fixed_worker_id if lock_cfg.lock_worker else old_assign[op_to_pos[op]][1]
            new_op_sequence.append(op)
            new_resource_assign.append((machine, worker))
        else:
            new_op_sequence.append(op)
            new_resource_assign.append(old_assign[op_to_pos[op]])
        processed_ops.add(op)

    new_ops = [op for op in opt_ops if op not in processed_ops]
    new_ops.sort(
        key=lambda x: (state_manager.operation_id_to_job_id[x], int(state_manager.op_meta_dict[x].business_op_no)))
    for op in new_ops:
        rg = state_manager.get_resource_group_by_op(op)
        available_machines = state_manager.get_available_machines(rg.machine_id_list)
        available_workers = state_manager.get_available_workers(rg.worker_id_list)
        new_op_sequence.append(op)
        new_resource_assign.append(_random_resource(available_machines, available_workers))
        processed_ops.add(op)

    new_op_sequence, new_resource_assign = deduplicate_sequence(new_op_sequence, new_resource_assign)
    new_chrom["op_sequence"] = new_op_sequence
    new_chrom["resource_assign"] = new_resource_assign
    return new_chrom


def decode_chromosome(chromosome: Dict[str, Any], state_manager: ProductionStateManager) -> Tuple[List[float], List[dict]]:
    """
    通用多目标排程染色体解码函数
    输入：染色体（操作序列+资源分配）、生产状态管理器、配置参数
    输出：多维适应度向量 + 详细排程结果列表
    支持批量拆分：若工序数量超过设备容量，自动均摊拆分为多个子批次
    """
    # --------------------------
    # 1. 输入合法性验证
    # --------------------------
    _validate_chromosome_input(chromosome, state_manager)

    operation_sequence = chromosome["op_sequence"]
    resource_assignment = chromosome["resource_assign"]

    # ===== 去重 + 裁剪超长 =====
    operation_sequence, resource_assignment = deduplicate_sequence(operation_sequence, resource_assignment)

    # --------------------------
    # 2. 初始化全局状态（只调用一次）
    # --------------------------
    trackers = _initialize_tracking_structures(state_manager)
    schedule_detail = []

    # --------------------------
    # 3. 逐操作解码核心流程
    # --------------------------
    for idx, operation_id in enumerate(operation_sequence):
        # 步骤1：获取并验证资源分配
        res = resource_assignment[idx]
        if res is None:
            # 兜底：随机分配可用资源
            rg = state_manager.get_resource_group_by_op(operation_id)
            avail_machines = state_manager.get_available_machines(rg.machine_id_list)
            avail_workers = state_manager.get_available_workers(rg.worker_id_list)
            if not avail_machines and not avail_workers:
                raise ValueError(f"工序{operation_id}无可用资源")
            raw_machine_id, raw_worker_id = _random_resource(avail_machines, avail_workers)
        else:
            raw_machine_id, raw_worker_id = res

        # 统一转为 NewType
        machine_id = MachineId(int(raw_machine_id))
        worker_id = WorkerId(int(raw_worker_id))

        # 步骤2：应用手动锁定规则
        machine_id, worker_id = _apply_manual_locks(
            operation_id, machine_id, worker_id, trackers, state_manager
        )

        # 步骤3：确保资源有效（处理-1和不可用资源）
        machine_id, worker_id = _ensure_valid_resource(
            operation_id, machine_id, worker_id, trackers, state_manager
        )

        # 步骤4：执行单操作调度计算（返回列表，支持批量拆分）
        batch_results = _schedule_single_operation(
            operation_id, machine_id, worker_id, trackers, state_manager, cfg
        )

        # 步骤5：将子批次结果加入排程记录
        for batch_res in batch_results:
            schedule_detail.append(_build_schedule_record(batch_res, state_manager))

    # --------------------------
    # 4. 计算最终适应度向量
    # --------------------------
    fitness_vector = _compute_fitness_vector(trackers, state_manager, cfg)
    return fitness_vector, schedule_detail


def fast_non_dominated_sorting(pop_fits: List[List[float]]) -> Tuple[List[List[int]], List[int]]:
    n = len(pop_fits)
    dominated: List[List[int]] = [[] for _ in range(n)]
    dom_count: List[int] = [0] * n
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _is_dominates(pop_fits[p], pop_fits[q]):
                dominated[p].append(q)
            elif _is_dominates(pop_fits[q], pop_fits[p]):
                dom_count[p] += 1
    fronts, rank = _extract_fronts(dominated, dom_count)
    return fronts, rank


def pox_crossover(p1: dict, p2: dict, state_manager: ProductionStateManager) -> Tuple[dict, dict]:
    seq1, assign1 = p1["op_sequence"], p1["resource_assign"]
    seq2, assign2 = p2["op_sequence"], p2["resource_assign"]

    frozen_boundary = state_manager.current_system_time + timedelta(hours=PLAN_FROZEN_HORIZON_HOURS)
    frozen_ops = _get_frozen_operations(seq1, state_manager, frozen_boundary)

    if len(frozen_ops) == len(seq1):
        return copy.deepcopy(p1), copy.deepcopy(p2)

    free_ops1 = _extract_free_operations(seq1, frozen_ops)
    free_ops2 = _extract_free_operations(seq2, frozen_ops)
    child1_free, child2_free = _pox_crossover_free_sequences(free_ops1, free_ops2, state_manager)

    frozen_set = set(frozen_ops)
    child1_free = [op for op in child1_free if op not in frozen_set]
    child2_free = [op for op in child2_free if op not in frozen_set]
    child1_free = deduplicate_op_list(child1_free)
    child2_free = deduplicate_op_list(child2_free)

    c1_res = rebuild_resource_assign(child1_free, state_manager, frozen_boundary)
    c2_res = rebuild_resource_assign(child2_free, state_manager, frozen_boundary)

    child1_seq = frozen_ops + child1_free
    child2_seq = frozen_ops + child2_free

    child1_assign = []
    child2_assign = []
    for fid in frozen_ops:
        s = state_manager.last_schedule_result[fid]
        child1_assign.append((s["machine_id"], s["worker_id"]))
        child2_assign.append((s["machine_id"], s["worker_id"]))

    child1_assign.extend(c1_res)
    child2_assign.extend(c2_res)
    child1_seq, child1_assign = deduplicate_sequence(child1_seq, child1_assign)
    child2_seq, child2_assign = deduplicate_sequence(child2_seq, child2_assign)

    return (
        {"op_sequence": child1_seq, "resource_assign": child1_assign},
        {"op_sequence": child2_seq, "resource_assign": child2_assign},
    )


def mutate_chromosome(chrom: dict, state_manager: ProductionStateManager) -> dict:
    def is_operation_frozen(op_id: int, frozen_boundary, state_manager: ProductionStateManager) -> bool:
        last_result = state_manager.last_schedule_result.get(op_id)
        if last_result is None:
            return False
        last_start_time = last_result.get("start_time")
        if last_start_time is None:
            return False
        return last_start_time < frozen_boundary

    new_chrom = copy.deepcopy(chrom)
    op_seq = new_chrom["op_sequence"]
    resource_assign = new_chrom["resource_assign"]
    frozen_boundary = state_manager.current_system_time + timedelta(hours=PLAN_FROZEN_HORIZON_HOURS)
    for idx, raw_op_id in enumerate(op_seq):
        op_id = int(raw_op_id)
        if is_operation_frozen(op_id, frozen_boundary, state_manager):
            continue
        if np.random.random() > cfg.MUTATION_RATE:
            continue
        selected_machine_id, selected_worker_id = _select_machine_and_worker(
            op_id, op_seq, resource_assign, state_manager
        )
        resource_assign[idx] = (selected_machine_id, selected_worker_id)
    new_chrom["resource_assign"] = resource_assign
    return new_chrom


def select_optimal_solution_by_weight(
    pareto_set: List[dict],
    all_pop_fits: List[List[float]],
    pareto_index_list: List[int],
    weight: List[float],
    # 新增入参：外部传入已初始化好的TOPSIS对象
    topsis_evaluator: TopsisAllMinEvaluator
) -> Tuple[dict, List[float], List[dict], List[List[float]]]:
    """
    基于TOPSIS(全指标越小越优)从帕累托解集中选取综合最优方案
    替换原线性加权求和择优逻辑，调用封装好的TOPSIS evaluate方法完成多准则评价排序
    :param pareto_set: 帕累托解集，元素为个体字典，顺序与pareto_index_list一一对应
    :param all_pop_fits: 整个种群所有个体的适应度向量列表，二维结构 [个体下标][各目标函数值]
    :param pareto_index_list: 帕累托个体在总种群中的下标集合
    :param weight: 各目标指标评价权重
    :param topsis_evaluator: 外部提前初始化完成的TOPSIS评价实例，全局复用
    :return: tuple(
        最优帕累托个体字典,
        该个体对应的适应度向量,
        排序后的帕累托子集列表（排序越靠前TOPSIS名次越好）,
        排序后帕累托解集对应的适应度二维列表
    )
    """
    pareto_fit_matrix = []
    for pop_idx in pareto_index_list:
        fit_vector = all_pop_fits[pop_idx]
        pareto_fit_matrix.append(fit_vector)

    # 修正：完整接收TOPSIS三个返回值，不需要原始排名列表用 _ 占位丢弃
    best_subset_idx, sorted_idx_by_rank= topsis_evaluator.evaluate(
        input_data=pareto_fit_matrix,
        weight=weight
    )

    best_real_pop_index = pareto_index_list[best_subset_idx]
    optimal_chromosome = pareto_set[best_subset_idx]
    optimal_fit_vector = all_pop_fits[best_real_pop_index]

    # 利用排序下标，生成【按名次从第1名往后排好的帕累托解集】
    sorted_pareto_list = [pareto_set[idx] for idx in sorted_idx_by_rank]
    sorted_fit_list = [all_pop_fits[pareto_index_list[idx]] for idx in sorted_idx_by_rank]

    # 四元组返回
    return optimal_chromosome, optimal_fit_vector, sorted_pareto_list, sorted_fit_list


def evaluate_population_fitness(population: List[Dict[str, Any]], state_manager: ProductionStateManager) -> List[List[float]]:
    fitness_list = []
    for chrom in population:
        fit_value, _ = decode_chromosome(chrom, state_manager)
        fitness_list.append(fit_value)
    return fitness_list


# ===================== 内部函数 =====================

def _validate_chromosome_input(chromosome: Dict[str, Any], state_manager: Any) -> None:
    if "op_sequence" not in chromosome or "resource_assign" not in chromosome:
        raise ValueError("染色体必须包含'op_sequence'和'resource_assign'")
    op_sequence = chromosome["op_sequence"]
    resource_assign = chromosome["resource_assign"]
    if len(op_sequence) != len(resource_assign):
        raise ValueError(f"操作序列长度不匹配")
    for op_id in op_sequence:
        if op_id not in state_manager.op_meta_dict:
            raise ValueError(f"未知的操作ID: {op_id}")


def _initialize_tracking_structures(state_manager: Any) -> SchedulingTrackers:
    trackers = SchedulingTrackers()
    current_time = state_manager.current_system_time

    for resource_group in state_manager.resource_group_dict.values():
        for machine_id in resource_group.machine_id_list:
            typed_machine_id = MachineId(machine_id)
            trackers.machine_last_end_time_dict[typed_machine_id] = current_time
            trackers.machine_previous_technology_type_dict[typed_machine_id] = -1
            trackers.machine_total_available_hour[typed_machine_id] = 0.0
            trackers.machine_total_process_hour[typed_machine_id] = 0.0

        for worker_id in resource_group.worker_id_list:
            typed_worker_id = WorkerId(worker_id)
            trackers.worker_task_intervals_dict[typed_worker_id] = []
            trackers.worker_task_ends_heap_dict[typed_worker_id] = []
            heapq.heappush(trackers.worker_task_ends_heap_dict[typed_worker_id], current_time)

    return trackers


def _apply_manual_locks(operation_id, original_machine_id, original_worker_id, trackers, state_manager):
    if not state_manager.is_op_manual_locked(operation_id):
        return original_machine_id, original_worker_id

    lock_config = state_manager.manual_lock_dict[operation_id]
    machine_id = original_machine_id
    worker_id = original_worker_id
    rg = state_manager.get_resource_group_by_op(operation_id)

    if lock_config.lock_machine and rg.machine_id_list:
        machine_id = MachineId(lock_config.fixed_machine_id)
    if lock_config.lock_worker and rg.worker_id_list:
        worker_id = WorkerId(lock_config.fixed_worker_id)

    if lock_config.lock_worker and not lock_config.lock_machine and rg.machine_id_list:
        if not state_manager.is_machine_available(original_machine_id):
            available_machines = state_manager.get_available_machines(rg.machine_id_list)
            if not available_machines:
                raise ValueError(f"操作[{operation_id}] 无可用机器")
            machine_id = select_best_machine(available_machines, trackers)

    if lock_config.lock_machine and not lock_config.lock_worker and rg.worker_id_list:
        if not state_manager.is_worker_available(original_worker_id):
            available_workers = state_manager.get_available_workers(rg.worker_id_list)
            if not available_workers:
                raise ValueError(f"操作[{operation_id}] 无可用工人")
            worker_id = select_best_worker(available_workers, trackers)

    return machine_id, worker_id


def _ensure_valid_resource(operation_id, machine_id, worker_id, trackers, state_manager):
    resource_group = state_manager.get_resource_group_by_op(operation_id)

    if resource_group.machine_id_list:
        if machine_id == MachineId(-1) or not state_manager.is_machine_available(machine_id):
            available_machines = state_manager.get_available_machines(resource_group.machine_id_list)
            if not available_machines:
                raise ValueError(f"操作[{operation_id}] 无可用机器")
            machine_id = select_best_machine(available_machines, trackers)

    if resource_group.worker_id_list:
        if worker_id == WorkerId(-1) or not state_manager.is_worker_available(worker_id):
            available_workers = state_manager.get_available_workers(resource_group.worker_id_list)
            if not available_workers:
                raise ValueError(f"操作[{operation_id}] 无可用工人")
            worker_id = select_best_worker(available_workers, trackers)

    return machine_id, worker_id


def _schedule_single_operation(
        operation_id: int,
        machine_id: MachineId,
        worker_id: WorkerId,
        trackers: SchedulingTrackers,
        state_manager: Any,
        config: Dict[str, Any]
) -> List[OperationSchedulingResult]:
    """
    单个操作的调度计算核心，支持批量拆分。
    返回一个或多个 OperationSchedulingResult（按子批顺序）。
    """
    operation_metadata = state_manager.op_meta_dict[operation_id]
    job_id = operation_metadata.belong_job_id
    technology_type = operation_metadata.op_tech_type
    rg = state_manager.get_resource_group_by_op(operation_id)

    # --------------------------
    # 1. 计算实际加工时间（速度系数）
    # --------------------------
    base_processing_time = operation_metadata.process_time
    speed_ratio = 1.0
    if worker_id != WorkerId(-1) and rg.worker_id_list:
        worker_meta = state_manager.worker_meta_dict.get(worker_id)
        if worker_meta and technology_type in worker_meta.tech_speed_ratio:
            speed_ratio = worker_meta.tech_speed_ratio[technology_type]

    # --------------------------
    # 2. 计算理想最早开始时间（基础约束）
    # --------------------------
    machine_available_time = state_manager.current_system_time
    if machine_id != MachineId(-1) and rg.machine_id_list:
        machine_available_time = trackers.machine_last_end_time_dict[machine_id]

    job_available_time = trackers.job_last_operation_end_time_dict.get(job_id, state_manager.current_system_time)
    material_ready_time = operation_metadata.material_ready_time
    if material_ready_time is None or material_ready_time == datetime.min:
        material_ready_time = state_manager.current_system_time

    # 工艺前驱约束
    current_biz_no = int(operation_metadata.business_op_no)
    all_job_ops = [oid for oid, meta in state_manager.op_meta_dict.items() if meta.belong_job_id == job_id]
    op_no_map = {int(state_manager.op_meta_dict[oid].business_op_no): oid for oid in all_job_ops}
    sorted_nos = sorted(op_no_map.keys())
    for idx, no in enumerate(sorted_nos):
        if no == current_biz_no and idx > 0:
            prev_op_id = op_no_map[sorted_nos[idx - 1]]
            prev_finish_time = trackers.job_op_finish_time_dict.get(prev_op_id)
            if prev_finish_time is not None:
                job_available_time = max(job_available_time, prev_finish_time)
            break

    ideal_start_time = max(machine_available_time, job_available_time, material_ready_time)

    # --------------------------
    # 3. 工人并行约束（仅当需要工人时）
    # --------------------------
    if worker_id != WorkerId(-1) and rg.worker_id_list:
        max_parallel = rg.worker_max_parallel
        heap = trackers.worker_task_ends_heap_dict[worker_id]
        while heap and heap[0] <= ideal_start_time:
            heapq.heappop(heap)
        if len(heap) >= max_parallel:
            ideal_start_time = max(ideal_start_time, heap[0])

    # --------------------------
    # 4. 机器换型时间
    # --------------------------
    if machine_id != MachineId(-1) and rg.machine_id_list:
        previous_tech = trackers.machine_previous_technology_type_dict[machine_id]
        if previous_tech != -1 and previous_tech != technology_type:
            changeover_map = state_manager.machine_meta_dict[machine_id].changeover_time_map
            changeover_time = changeover_map.get(previous_tech, {}).get(technology_type, 0.0)
            trackers.total_changeover_time += changeover_time
            changeover_end = state_manager.calculate_actual_work_end_time(machine_available_time, changeover_time)
            ideal_start_time = max(ideal_start_time, changeover_end)
        trackers.machine_previous_technology_type_dict[machine_id] = technology_type

    # --------------------------
    # 5. 班次日历约束（第一个子批开始时间）
    # --------------------------
    first_batch_start = state_manager.get_valid_start_time(ideal_start_time)

    # --------------------------
    # 6. 计算拆分批次
    # --------------------------
    total_qty = operation_metadata.op_quantity
    machine_meta = state_manager.machine_meta_dict.get(machine_id)
    capacity = get_actual_batch_capacity(machine_meta, operation_metadata)
    batch_sizes = split_into_batches(total_qty, capacity)

    # --------------------------
    # 7. 逐批次计算时间
    # --------------------------
    frozen_time_limit = state_manager.current_system_time + timedelta(hours=PLAN_FROZEN_HORIZON_HOURS)
    results = []
    current_start = first_batch_start

    for batch_idx, batch_qty in enumerate(batch_sizes):
        batch_process_time = round(base_processing_time * batch_qty * speed_ratio, 1)
        batch_end = state_manager.calculate_actual_work_end_time(current_start, batch_process_time)
        is_frozen = current_start < frozen_time_limit

        batch_result = OperationSchedulingResult(
            operation_id=OperationId(operation_id),
            job_id=JobId(str(job_id)),
            operation_index_in_job=operation_metadata.op_index_in_job,
            machine_id=machine_id,
            worker_id=worker_id,
            start_time=current_start,
            end_time=batch_end,
            actual_processing_time=batch_process_time,
            technology_type=technology_type,
            is_frozen=is_frozen,
            is_manual_locked=state_manager.is_op_manual_locked(operation_id),
            operation_metadata=operation_metadata,
            batch_index=batch_idx,
            batch_total=len(batch_sizes)
        )
        results.append(batch_result)

        # 更新跟踪器，为下一子批准备
        _update_trackers(batch_result, trackers)
        current_start = batch_end

    return results


def _update_trackers(result: OperationSchedulingResult, trackers: SchedulingTrackers) -> None:
    job_id = result.job_id
    machine_id = result.machine_id
    worker_id = result.worker_id
    start_time = result.start_time
    end_time = result.end_time
    actual_processing_time = result.actual_processing_time

    if job_id in trackers.job_last_operation_end_time_dict:
        previous_end = trackers.job_last_operation_end_time_dict[job_id]
        if previous_end < start_time:
            trackers.total_wip_wait_time += (start_time - previous_end).total_seconds() / 3600.0

    if machine_id != MachineId(-1):
        trackers.machine_last_end_time_dict[machine_id] = end_time
        trackers.machine_workloads_dict[machine_id].append(actual_processing_time)
        trackers.machine_total_process_hour[machine_id] += actual_processing_time

    if worker_id != WorkerId(-1):
        trackers.worker_task_intervals_dict[worker_id].append((start_time, end_time))
        heapq.heappush(trackers.worker_task_ends_heap_dict[worker_id], end_time)
        trackers.worker_workloads_dict[worker_id].append(actual_processing_time)

    trackers.job_last_operation_end_time_dict[job_id] = end_time
    trackers.job_op_finish_time_dict[result.operation_id] = end_time
    trackers.total_operation_count += 1


def _build_schedule_record(result: OperationSchedulingResult, state_manager: Any) -> Dict[str, Any]:
    op_meta = result.operation_metadata
    op_id_int = int(result.operation_id)
    return {
        "op_id": op_id_int,
        "op_status": op_meta.op_status,
        "job_id": result.job_id,
        "job_op_index": result.operation_index_in_job,
        "business_op_id": op_meta.business_op_id,
        "business_op_no": op_meta.business_op_no,
        "op_name": op_meta.op_name,
        "op_content": op_meta.op_content,
        "resource_group_id": op_meta.resource_group_id,
        "resource_group_name": op_meta.resource_group_name,
        "machine_id": int(result.machine_id),
        "worker_id": int(result.worker_id),
        "start_time": result.start_time,
        "end_time": result.end_time,
        "real_start_time": result.start_time.isoformat(),
        "real_end_time": result.end_time.isoformat(),
        "batch_index": result.batch_index,
        "batch_total": result.batch_total,
        "tech_type": result.technology_type,
        "real_process_time": result.actual_processing_time,
        "is_frozen": result.is_frozen,
        "is_manual_locked": result.is_manual_locked,
    }


def _compute_fitness_vector(trackers: SchedulingTrackers, state_manager: Any, config: Dict[str, Any]) -> List[float]:
    """
    单条排程方案适应度向量计算函数
    共7个最小化优化目标，完成工时统计、逾期罚金、设备利用率、负荷均衡、在制品等待成本核算
    全部时长类指标统一换算为【自然天】口径，抹平原始量级差距，适配NSGA-II寻优与后续TOPSIS择优
    :param trackers: 排程过程追踪器，存储各工序完工时刻、设备工时、人员工时、在制品时长等中间数据
    :param state_manager: 全局生产状态管理器，包含订单基础信息、工作日历、逾期罚金计算方法
    :param config: 配置字典，读取在制品权重系数等全局参数
    :return: list[float] 7维适应度向量，依次为：
        [逾期订单数量, 总逾期惩罚成本, 总完工时间(天), 设备闲置率(%), 设备负荷不均衡方差, 人员负荷不均衡方差, 在制品等待总成本]
    """
    # ---------------------- 1. 统计逾期订单数量 + 累加总逾期罚金 ----------------------
    overdue_count = 0
    total_overdue_penalty = 0.0
    # 遍历所有订单最终完工时刻，判断是否超交付期
    for job_id, finish_time in trackers.job_last_operation_end_time_dict.items():
        job_meta = state_manager.job_meta_dict.get(job_id)
        if not job_meta:
            continue
        # 订单实际完工时间晚于要求交付期，计算单笔逾期罚金并累加
        if finish_time > job_meta.due_delivery_time:
            penalty = state_manager.calc_delivery_overdue_penalty(finish_time, job_meta)
            total_overdue_penalty += penalty
            overdue_count += 1

    # ---------------------- 2. 计算总完工时间Makespan（单位：天） ----------------------
    if trackers.job_last_operation_end_time_dict:
        max_end = max(trackers.job_last_operation_end_time_dict.values())
        # 总秒数 ÷3600→小时，再÷24转为自然天
        makespan = (max_end - state_manager.work_calendar.base_zero).total_seconds() / (3600.0 * 24.0)
    else:
        makespan = 0.0

    # ---------------------- 3. 计算整体设备闲置率（百分比形式） ----------------------
    total_all_available_time = 0.0
    total_all_idle_time = 0.0
    current_time = state_manager.current_system_time

    # 遍历每台有效设备，统计可用总工时、闲置总工时
    for mid in trackers.machine_last_end_time_dict.keys():
        if mid == MachineId(-1):
            continue
        proc_h = trackers.machine_total_process_hour[mid]
        max_end_dt = max(trackers.job_last_operation_end_time_dict.values()) if trackers.job_last_operation_end_time_dict else current_time
        machine_available_h = state_manager.work_hours_between(current_time, max_end_dt)
        idle_h = max(0.0, machine_available_h - proc_h)
        total_all_available_time += machine_available_h
        total_all_idle_time += idle_h

    # 闲置率 = 总闲置工时 / 总可用工时 * 100，转为百分比，避免0~1小数值量级过小
    if total_all_available_time > 1e-9:
        overall_equipment_idle_rate = round(total_all_idle_time / total_all_available_time * 100, 2)
    else:
        overall_equipment_idle_rate = 0.0

    # ---------------------- 4. 设备负荷不均衡方差（工时转天计算） ----------------------
    machine_total_loads = []
    for mid, loads in trackers.machine_workloads_dict.items():
        if mid != MachineId(-1):
            total_hours = sum(loads)
            total_days = total_hours / 24.0  # 单台设备总加工工时 小时→天
            machine_total_loads.append(total_days)
    # 多设备负荷方差，方差越大代表设备负荷越不均衡；仅≥2台设备才有统计意义，否则置0
    machine_unbalance = round(np.var(machine_total_loads), 2) if len(machine_total_loads) > 1 else 0.0

    # ---------------------- 5. 人员负荷不均衡方差（工时转天计算） ----------------------
    worker_total_loads = []
    for wid, loads in trackers.worker_workloads_dict.items():
        if wid != WorkerId(-1):
            total_hours = sum(loads)
            total_days = total_hours / 24.0  # 单个人员总加工工时 小时→天
            worker_total_loads.append(total_days)
    # 多人员负荷方差，方差越大代表人员劳逸分配越不均衡；仅≥2人时有统计意义，否则置0
    worker_unbalance = round(np.var(worker_total_loads), 2) if len(worker_total_loads) > 1 else 0.0

    # ---------------------- 6. 在制品等待总成本 ----------------------
    # 在制品总等待时长(天） × 单位天权重系数，换算为等待成本
    wip_cost = trackers.total_wip_wait_time / 24.0 * cfg.WIP_WEIGHT_COEFFICIENT

    # 7维适应度向量返回，所有时长类统一天口径，缓解原始量纲悬殊，适配Min-Max归一TOPSIS排序
    return [
        overdue_count,                # 指标1：逾期订单数量（越小越优）
        total_overdue_penalty,        # 指标2：订单总逾期惩罚成本（越小越优）
        makespan,                     # 指标3：全局总完工时间(天)（越小越优）
        overall_equipment_idle_rate, # 指标4：设备整体闲置率(%)（越小越优）
        machine_unbalance,           # 指标5：设备负荷不均衡方差（越小越优）
        worker_unbalance,            # 指标6：人员负荷不均衡方差（越小越优）
        wip_cost                     # 指标7：在制品等待总成本（越小越优）
    ]


def _get_frozen_operations(seq, state_manager, frozen_boundary):
    frozen = []
    last_schedule = state_manager.last_schedule_result
    for op_id in seq:
        if op_id in last_schedule and last_schedule[op_id]["start_time"] < frozen_boundary:
            frozen.append(op_id)
    return frozen


def _extract_free_operations(seq, frozen_ops):
    return seq[len(frozen_ops):]


def _pox_crossover_free_sequences(free_ops1, free_ops2, state_manager):
    op_to_job = state_manager.operation_id_to_job_id
    job_set = set(op_to_job[op] for op in free_ops1 + free_ops2)
    job_list = list(job_set)
    if len(job_list) <= 1:
        return free_ops1.copy(), free_ops2.copy()
    split_idx = np.random.randint(1, len(job_list))
    group1 = set(job_list[:split_idx])
    group2 = set(job_list[split_idx:])
    child1_free = []
    child2_free = []
    for op in free_ops1:
        if op_to_job[op] in group1:
            child1_free.append(int(op))
    for op in free_ops2:
        if op_to_job[op] in group2:
            child1_free.append(int(op))
    for op in free_ops2:
        if op_to_job[op] in group1:
            child2_free.append(int(op))
    for op in free_ops1:
        if op_to_job[op] in group2:
            child2_free.append(int(op))
    return child1_free, child2_free


def _random_select_resources(op_id, state_manager):
    rg = state_manager.get_resource_group_by_op(op_id)
    available_machines = state_manager.get_available_machines(rg.machine_id_list)
    available_workers = state_manager.get_available_workers(rg.worker_id_list)
    return _random_resource(available_machines, available_workers)


def _select_machine_and_worker(op_id, op_sequence, resource_assign, state_manager):
    rg = state_manager.get_resource_group_by_op(op_id)
    if not rg.machine_id_list and not rg.worker_id_list:
        return -1, -1
    if not rg.machine_id_list:
        available_workers = state_manager.get_available_workers(rg.worker_id_list)
        worker_id = int(np.random.choice(available_workers)) if available_workers else -1
        return -1, worker_id
    if not rg.worker_id_list:
        reuse_mid = try_get_prev_same_group_machine(op_id, op_sequence, resource_assign, state_manager)
        if reuse_mid is not None:
            return int(reuse_mid), -1
        available_machines = state_manager.get_available_machines(rg.machine_id_list)
        machine_id = int(np.random.choice(available_machines)) if available_machines else -1
        return machine_id, -1
    reuse_mid = try_get_prev_same_group_machine(op_id, op_sequence, resource_assign, state_manager)
    available_workers = state_manager.get_available_workers(rg.worker_id_list)
    if reuse_mid is not None:
        return int(reuse_mid), int(np.random.choice(available_workers) if available_workers else -1)
    else:
        return _random_select_resources(op_id, state_manager)


def _is_dominates(candidate, other):
    all_less_or_equal = all(c <= o for c, o in zip(candidate, other))
    any_strictly_better = any(c < o for c, o in zip(candidate, other))
    return all_less_or_equal and any_strictly_better


def _extract_fronts(dominated, dom_count):
    n = len(dom_count)
    rank = [0] * n
    fronts = [[]]
    for p in range(n):
        if dom_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    front_idx = 0
    while front_idx < len(fronts):
        next_fronts = []
        for p in fronts[front_idx]:
            for q in dominated[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    rank[q] = front_idx + 1
                    next_fronts.append(q)
        front_idx += 1
        if next_fronts:
            fronts.append(next_fronts)
    return fronts, rank


def try_get_prev_same_group_machine(current_op_id, op_sequence, resource_assign, state_manager):
    op_meta = state_manager.op_meta_dict[current_op_id]
    job_id = op_meta.belong_job_id
    current_rg_id = op_meta.resource_group_id
    current_biz_no = int(op_meta.business_op_no)
    all_job_ops = [oid for oid, meta in state_manager.op_meta_dict.items() if meta.belong_job_id == job_id]
    op_no_map = {}
    for oid in all_job_ops:
        no_val = int(state_manager.op_meta_dict[oid].business_op_no)
        op_no_map[no_val] = oid
    sorted_no_list = sorted(op_no_map.keys())
    target_prev_no = None
    for idx, no in enumerate(sorted_no_list):
        if no == current_biz_no:
            if idx > 0:
                target_prev_no = sorted_no_list[idx - 1]
            break
    if target_prev_no is None:
        return None
    prev_op_id = op_no_map[target_prev_no]
    prev_op_meta = state_manager.op_meta_dict[prev_op_id]
    if prev_op_meta.resource_group_id != current_rg_id:
        return None
    try:
        prev_idx = op_sequence.index(prev_op_id)
        prev_machine, _ = resource_assign[prev_idx]
    except (ValueError, TypeError):
        return None
    rg = state_manager.get_resource_group_by_op(current_op_id)
    if prev_machine not in rg.machine_id_list:
        return None
    if not state_manager.is_machine_available(prev_machine):
        return None
    return prev_machine


def rebuild_resource_assign(new_op_seq, state_manager, frozen_boundary):
    total_len = len(new_op_seq)
    resource_assign = [None] * total_len
    op_index_map = {op: idx for idx, op in enumerate(new_op_seq)}

    def is_op_frozen(op_id):
        res = state_manager.last_schedule_result.get(op_id)
        if res is None or res.get("start_time") is None:
            return False
        return res["start_time"] < frozen_boundary

    job_op_dict: Dict[int, List[int]] = {}
    for op_id in new_op_seq:
        jid = state_manager.op_meta_dict[op_id].belong_job_id
        if jid not in job_op_dict:
            job_op_dict[jid] = []
        job_op_dict[jid].append(op_id)

    for job_id, op_list in job_op_dict.items():
        op_list_sorted = sorted(op_list, key=lambda x: int(state_manager.op_meta_dict[x].business_op_no))
        for op_id in op_list_sorted:
            pos = op_index_map[op_id]
            if is_op_frozen(op_id):
                mid = state_manager.last_schedule_result[op_id]["machine_id"]
                wid = state_manager.last_schedule_result[op_id]["worker_id"]
                resource_assign[pos] = (mid, wid)
                continue
            selected_machine_id, selected_worker_id = _select_machine_and_worker(
                op_id, new_op_seq, resource_assign, state_manager
            )
            resource_assign[pos] = (selected_machine_id, selected_worker_id)
    return resource_assign


def deduplicate_sequence(op_sequence, resource_assign):
    seen = set()
    clean_seq = []
    clean_assign = []
    for op, res in zip(op_sequence, resource_assign):
        if op not in seen:
            seen.add(op)
            clean_seq.append(op)
            clean_assign.append(res)
    if len(clean_seq) != len(op_sequence):
        from collections import Counter
        counter = Counter(op_sequence)
        duplicates = {op: count for op, count in counter.items() if count > 1}
        logger.warning(f"去重: {len(op_sequence)} → {len(clean_seq)}, 重复工序: {duplicates}")
    return clean_seq, clean_assign


def deduplicate_op_list(op_list):
    seen = set()
    result = []
    for op in op_list:
        if op not in seen:
            seen.add(op)
            result.append(op)
    return result


def _random_resource(machines, workers):
    machine_id = MachineId(int(np.random.choice(machines))) if machines else MachineId(-1)
    worker_id = WorkerId(int(np.random.choice(workers))) if workers else WorkerId(-1)
    return machine_id, worker_id

def get_actual_batch_capacity(machine_meta, operation_meta) -> int:
    """计算该零件在指定设备上的单批最大容量"""
    if machine_meta is None or machine_meta.standard_capacity <= 0:
        return operation_meta.op_quantity  # 不限容量
    return max(1, int(machine_meta.standard_capacity * operation_meta.size_factor))


def split_into_batches(total_qty: int, batch_capacity: int) -> List[int]:
    """均摊拆分，保证每批 ≤ batch_capacity"""
    if batch_capacity <= 0 or total_qty <= batch_capacity:
        return [total_qty]
    num_batches = (total_qty + batch_capacity - 1) // batch_capacity
    base = total_qty // num_batches
    remainder = total_qty % num_batches
    return [base + (1 if i < remainder else 0) for i in range(num_batches)]