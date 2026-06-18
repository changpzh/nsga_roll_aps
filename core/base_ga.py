# core/base_ga.py
import numpy as np
import heapq
import copy
from typing import List, Dict, Tuple,Any
from models import MachineId, WorkerId, OperationId, JobId, SchedulingTrackers, OperationSchedulingResult
from core.state_manager import ProductionStateManager
from utils.common_utils import select_best_worker, select_best_machine
from core.calendar import WorkCalendar
import config.settings as cfg
from config.settings import (
    WORKER_SWITCH_COST, WIP_WEIGHT_COEFFICIENT, POPULATION_SIZE,
    ROLLING_HISTORY_SEED_RATIO, ROLLING_HEURISTIC_SEED_RATIO, ROLLING_PERTURB_RATE
)
from utils.log_utils import get_logger

logger = get_logger(__name__)


def init_single_chromosome(reorder_job_seq: List[int], state_manager: ProductionStateManager, shuffle_free: bool = True) -> dict:
    """
    【函数整体功能】
    生成一条完整可行的排程染色体基础模板，区分冻结工序与可浮动工序；
    冻结工序完全继承上一轮排程的设备、工人、开工顺序，禁止改动；
    浮动工序可选择随机打乱订单顺序或固定订单顺序，同订单内部严格按工艺工序号升序排布；
    浮动工序随机分配同资源组内可用机床与操作工；
    输出结构标准化染色体字典，解码后可直接计算6项目标适应度，全程保证硬约束前置合规。

    【输入参数】
    :param reorder_job_seq: List[int] 本次参与重排的订单ID列表
    :param state_manager: ProductionStateManager 全局生产状态管理器实例
    :param shuffle_free: bool 浮动订单是否随机打乱顺序；True=随机打乱（随机种群用），False=保持输入订单顺序（启发式种群用）

    【输出返回】
    :return: dict 单条染色体字典，固定两个key
        "op_sequence": List[int] 全局工序ID执行顺序（冻结工序在前，浮动工序在后）
        "resource_assign": List[Tuple[int,int]] 和op_sequence一一对应，每项(机床ID,工人ID)资源分配
    """
    # ====================== 步骤1：筛选所有可优化浮动工序，同订单内按工艺编号升序 ======================
    target_ops = []
    # 获取全局所有允许优化的工序集合（过滤完工、运行中工序）
    all_opt_set = set(state_manager.get_optimizable_operation_ids())

    for job_id in reorder_job_seq:
        # 取出该订单绑定的全部工序ID
        job_op_list = [op for op, jid in state_manager.operation_id_to_job_id.items() if jid == job_id]
        # 过滤：只保留属于可优化池的工序
        valid_ops = [op for op in job_op_list if op in all_opt_set]
        # 按业务工艺编号数字从小到大排序，保证订单内部工艺先后顺序不变
        valid_ops.sort(key=lambda x: int(state_manager.op_meta_dict[x].business_op_no))
        target_ops.extend(valid_ops)

    # ====================== 步骤2：划分三类工序：冻结工序/旧浮动工序/全新工序 ======================
    # 冻结时间阈值：当前系统相对工时 + 计划冻结窗口时长
    frozen_boundary = state_manager.current_system_time + cfg.PLAN_FROZEN_HORIZON
    frozen_ops = []        # 冻结区间内工序：完全沿用上次排程，不可改顺序、资源
    unfrozen_old_ops = []  # 上次存在、本次在冻结窗口外的旧工序（可重排）
    new_ops = []           # 本次新增从未排过的工序

    for op_id in target_ops:
        if op_id in state_manager.last_schedule_result:
            last_start = state_manager.last_schedule_result[op_id]["start_time"]
            if last_start < frozen_boundary:
                # 开工时间落在冻结区间内，锁定不动
                frozen_ops.append(op_id)
            else:
                # 旧工序但在冻结窗口之后，允许重排
                unfrozen_old_ops.append(op_id)
        else:
            # 全新插单工序，无历史排程记录
            new_ops.append(op_id)

    # 冻结工序按上次实际开工时间升序排列，保证现场执行时序不变
    frozen_ops.sort(key=lambda x: state_manager.last_schedule_result[x]["start_time"])
    # 所有可自由调整的浮动工序集合
    free_ops = unfrozen_old_ops + new_ops

    # ====================== 步骤3：组装浮动工序序列，控制是否随机打乱订单 ======================
    # 构建 订单ID: 该订单下所有浮动工序列表
    job_op_map: Dict[int, List[int]] = {}
    for op_id in free_ops:
        jid = state_manager.operation_id_to_job_id[op_id]
        if jid not in job_op_map:
            job_op_map[jid] = []
        job_op_map[jid].append(op_id)

    job_list = list(job_op_map.keys())
    # shuffle_free=True时随机打乱订单先后顺序；启发式调用传False则保持原有订单排序
    if shuffle_free:
        np.random.shuffle(job_list)

    # 按订单顺序拼接浮动工序（订单内部工序早已按工艺号排好）
    free_sequence = []
    for jid in job_list:
        free_sequence.extend(job_op_map[jid])

    # 最终全局工序顺序：冻结工序固定在前，浮动工序在后
    op_sequence = frozen_ops + free_sequence

    # ====================== 步骤4：逐工序分配机床、工人资源 ======================
    resource_assign = []
    for op_id in op_sequence:
        if op_id in frozen_ops:
            # 冻结工序：完全复用上次排程的机床、工人，不做任何改动
            mid = state_manager.last_schedule_result[op_id]["machine_id"]
            wid = state_manager.last_schedule_result[op_id]["worker_id"]
            resource_assign.append((mid, wid))
        else:
            # 浮动工序：取工序归属资源组，随机挑选可用设备、操作工
            rg = state_manager.get_resource_group_by_op(op_id)
            # 筛选组内当前状态可用机床
            valid_macs = [m for m in rg.machine_id_list if state_manager.machine_available_dict.get(m, False)]
            # 筛选组内具备该工艺技能、在岗可用工人
            valid_workers = [w for w in rg.worker_id_list if state_manager.worker_available_dict.get(w, False)]
            # 随机选择一台机床、一名工人
            sel_mac = np.random.choice(valid_macs)
            sel_wid = np.random.choice(valid_workers)
            resource_assign.append((sel_mac, sel_wid))

    return {"op_sequence": op_sequence, "resource_assign": resource_assign}


def init_mixed_population(reorder_job_seq: List[int], state_manager: ProductionStateManager) -> List[dict]:
    """
    【函数整体功能】
    滚动APS-NSGA混合种群初始化函数，采用「历史扰动解 + 启发式优先解 + 随机可行解」三类个体混合填充种群；
    适配滚动插单、设备故障、长期稳态排产场景，自动自适应新订单占比调整种子配比，兼顾收敛速度、计划稳定性、种群多样性；
    所有生成染色体均经过硬约束矫正，种群内全部为可行生产排程方案，无非法不可行个体。

    【输入参数】
    :param reorder_job_seq: List[int] 本次允许重排优化的订单ID列表，限定算法仅处理这些订单内浮动工序
    :param state_manager: ProductionStateManager 全局生产状态管理器实例，存储历史帕累托解、工序元数据、订单映射、日历、冻结状态等

    【输出返回】
    :return: List[dict] 完整遗传种群列表，列表长度严格等于POPULATION_SIZE；
             列表内每一个dict为一条染色体，代表一套完整工序排程方案，key包含op_sequence、资源分配、各目标适应度等结构。
    """
    # -------------------------- 步骤1：基础配额拆分 --------------------------
    pop_size = POPULATION_SIZE
    # 按配置比例初始化三类个体数量
    history_num = int(pop_size * ROLLING_HISTORY_SEED_RATIO)
    heuristic_num = int(pop_size * ROLLING_HEURISTIC_SEED_RATIO)
    random_num = pop_size - history_num - heuristic_num
    population = []

    # -------------------------- 步骤2：计算新订单占比，自适应调配比 --------------------------
    old_job_set = set()
    # 遍历上一轮所有帕累托解，提取历史解中存在的旧订单ID
    for chrom in state_manager.last_pareto_solutions:
        for op in chrom["op_sequence"]:
            job_id = state_manager.operation_id_to_job_id[op]
            old_job_set.add(job_id)
    # 计算本次待排订单里全新订单的占比
    new_job_ratio = state_manager.get_new_job_ratio(old_job_set)
    # 工程经验阈值：新订单超30%，缩减历史解占比，放开随机搜索空间
    if new_job_ratio > 0.3:
        history_num = int(pop_size * 0.3)
        heuristic_num = int(pop_size * 0.25)
        random_num = pop_size - history_num - heuristic_num

    # -------------------------- 步骤3：生成历史扰动种子个体 --------------------------
    history_candidates = state_manager.last_pareto_solutions
    if len(history_candidates) > 0:
        # 存在上一轮帕累托优质解，执行裁剪+扰动逻辑
        for _ in range(history_num):
            # 随机抽取一条历史最优染色体，深拷贝防止篡改原始存储数据
            base_chrom = copy.deepcopy(np.random.choice(history_candidates))
            # 裁剪：剔除冻结、完工、正在加工的锁定工序，只保留本次可优化浮动工序
            trimmed_chrom = trim_historical_chromosome(base_chrom, state_manager)
            # 对浮动工序小幅局部扰动（交换顺序/调换同组设备工人），生成新种子
            new_history_seed = perturb_historical_chromosome(trimmed_chrom, state_manager)
            population.append(new_history_seed)
    else:
        # 冷启动场景：无历史解集，历史配额全部用随机可行染色体填充兜底
        for _ in range(history_num):
            rand_chrom = init_single_chromosome(reorder_job_seq, state_manager)
            population.append(rand_chrom)

    # -------------------------- 步骤4：生成启发式规则种子个体 --------------------------
    for _ in range(heuristic_num):
        # 按业务优先级规则定向构建高质量初始排程方案
        heur_chrom = generate_heuristic_chromosome(reorder_job_seq, state_manager)
        population.append(heur_chrom)

    # -------------------------- 步骤5：生成纯随机可行种子个体 --------------------------
    for _ in range(random_num):
        # 随机编码浮动工序，解码阶段自动矫正所有硬约束，保证输出可行解
        rand_chrom = init_single_chromosome(reorder_job_seq, state_manager)
        population.append(rand_chrom)

    # -------------------------- 步骤6：返回完整混合种群 --------------------------
    return population


def perturb_historical_chromosome(trimmed_chrom: dict, state_manager: ProductionStateManager) -> dict:
    chrom = copy.deepcopy(trimmed_chrom)
    opt_ops = set(state_manager.get_optimizable_operation_ids())
    seq = chrom["op_sequence"]
    assign = chrom["resource_assign"]
    total_len = len(seq)
    perturb_num = max(1, int(total_len * ROLLING_PERTURB_RATE))
    perturb_indexes = np.random.choice(range(total_len), perturb_num, replace=False)
    for idx in perturb_indexes:
        op_id = seq[idx]
        if op_id not in opt_ops or state_manager.is_op_manual_locked(op_id):
            continue
        jid = state_manager.operation_id_to_job_id[op_id]
        job_meta = state_manager.job_meta_dict[jid]
        if job_meta.priority in ["high", "urgent"]:
            if np.random.random() > 0.4:
                continue
        rg = state_manager.get_resource_group_by_op(op_id)
        valid_macs = [m for m in rg.machine_id_list if state_manager.machine_available_dict.get(m, False)]
        valid_workers = [w for w in rg.worker_id_list if state_manager.worker_available_dict.get(w, False)]
        new_mac = np.random.choice(valid_macs)
        new_worker = np.random.choice(valid_workers)
        assign[idx] = (new_mac, new_worker)
    chrom["resource_assign"] = assign
    return chrom


def generate_heuristic_chromosome(reorder_job_seq: List[int], state_manager: ProductionStateManager) -> dict:
    """
    【函数整体功能】
    生成启发式优质初始染色体，核心规则：订单优先级最高、合同交期最早的订单优先排产；
    先对订单做加权+交期双层排序，再调用基础染色体生成函数，禁止随机打乱工序顺序，产出贴合交付诉求的基准可行排程方案；
    作为混合种群里的高质量种子个体，拉高种群整体初始适应度，加速NSGA收敛。
    """
    job_sort_info = []
    # 遍历待排订单，组装排序三元组
    for j_id in reorder_job_seq:
        meta = state_manager.job_meta_dict[j_id]
        # 负权重实现优先级降序，交期为次要升序条件
        job_sort_info.append((-meta.base_weight, meta.due_contract_time, j_id))
    # 升序排序：先对比第一元素(-权重)，再对比第二元素(合同交期)
    job_sort_info.sort()
    # 提取排好顺序的订单ID
    sorted_job_ids = [item[2] for item in job_sort_info]
    # shuffle_free=False：不随机打乱工序/订单顺序，严格遵循启发排序结果
    return init_single_chromosome(sorted_job_ids, state_manager, shuffle_free=False)


def trim_historical_chromosome(old_chrom: dict, state_manager: ProductionStateManager) -> dict:
    new_chrom = copy.deepcopy(old_chrom)
    opt_ops = set(state_manager.get_optimizable_operation_ids())
    new_op_sequence = []
    new_resource_assign = []
    old_seq = old_chrom["op_sequence"]
    old_assign = old_chrom["resource_assign"]

    # 修正1：只遍历历史染色体原有工序，筛选保留已完工/运行工序，不乱新增工序
    for op_idx in old_seq:
        status = state_manager.operation_status_dict.get(op_idx, -1)
        if status in [1, 2]:
            pos = old_seq.index(op_idx)
            new_op_sequence.append(op_idx)
            new_resource_assign.append(old_assign[pos])

    # 修正2：遍历当前所有可优化工序，补齐到序列中
    for op_id in opt_ops:
        if state_manager.is_op_manual_locked(op_id):
            new_op_sequence.append(op_id)
            lock_cfg = state_manager.get_lock_info(op_id)
            mac = lock_cfg.fixed_machine_id if lock_cfg.lock_machine else -1
            worker = lock_cfg.fixed_worker_id if lock_cfg.lock_worker else -1
            new_resource_assign.append((mac, worker))
            continue
        if op_id in old_seq:
            pos = old_seq.index(op_id)
            new_op_sequence.append(op_id)
            new_resource_assign.append(old_assign[pos])
        else:
            rg = state_manager.get_resource_group_by_op(op_id)
            valid_macs = [m for m in rg.machine_id_list if state_manager.machine_available_dict.get(m, False)]
            valid_workers = [w for w in rg.worker_id_list if state_manager.worker_available_dict.get(w, False)]
            new_op_sequence.append(op_id)
            new_resource_assign.append((np.random.choice(valid_macs), np.random.choice(valid_workers)))

    new_chrom["op_sequence"] = new_op_sequence
    new_chrom["resource_assign"] = new_resource_assign
    return new_chrom

def decode_chromosome(chromosome:Dict[str,Any], state_manager: ProductionStateManager) -> Tuple[List[float], List[dict]]:
    """
    通用多目标排程染色体解码函数
    输入：染色体（操作序列+资源分配）、生产状态管理器、配置参数
    输出：多维适应度向量 + 详细排程结果列表
    """
    # --------------------------
    # 1. 输入合法性验证（提前失败原则）
    # --------------------------
    _validate_chromosome_input(chromosome, state_manager)

    operation_sequence = chromosome["op_sequence"]
    resource_assignment = chromosome["resource_assign"]

    # --------------------------
    # 2. 初始化全局状态
    # --------------------------
    trackers = _initialize_tracking_structures(state_manager)
    schedule_detail = []

    # --------------------------
    # 3. 逐操作解码核心流程
    # --------------------------
    for idx, operation_id in enumerate(operation_sequence):
        # 步骤1：获取并验证资源分配
        raw_machine_id, raw_worker_id = resource_assignment[idx]
        # 类型转换（确保符合类型安全要求）
        raw_machine_id = MachineId(raw_machine_id)
        raw_worker_id = WorkerId(raw_worker_id)

        # 步骤2：应用手动锁定规则
        machine_id, worker_id = _apply_manual_locks(
            operation_id, raw_machine_id, raw_worker_id, trackers, state_manager
        )

        # 步骤3：确保资源有效（处理-1和不可用资源）
        machine_id, worker_id = _ensure_valid_resource(
            operation_id, machine_id, worker_id, trackers, state_manager
        )

        # 步骤4：执行单操作调度计算
        scheduling_result = _schedule_single_operation(
            operation_id, machine_id, worker_id, trackers, state_manager, cfg
        )

        # 步骤5：更新全局状态跟踪器
        _update_trackers(scheduling_result, trackers)

        # 步骤6：生成排程记录
        schedule_detail.append(_build_schedule_record(scheduling_result, state_manager))

    # --------------------------
    # 4. 计算最终适应度向量
    # --------------------------
    fitness_vector = _compute_fitness_vector(trackers, state_manager, cfg)

    return fitness_vector, schedule_detail

def decode_chromosome2(chrom: dict, state_manager: ProductionStateManager) -> Tuple[List[float], List[dict]]:
    """
    通用染色体解码函数，所有多目标算法共用
    输入染色体，输出6维目标适应度向量和详细排程结果
    """
    seq = chrom["op_sequence"]
    assign = chrom["resource_assign"]
    machine_last_end: Dict[int, float] = {}
    worker_task_list: Dict[int, List[Tuple[float, float]]] = {}
    job_last_op_end: Dict[int, float] = {}
    machine_prev_tech_type: Dict[int, int] = {}
    machine_workload: Dict[int, List[float]] = {}
    worker_workload: Dict[int, List[float]] = {}
    total_changeover_time = 0.0
    worker_switch_count = 0
    segment_overdue_penalty = 0.0
    total_wip_wait_time = 0.0
    schedule_detail = []
    for rg in state_manager.resource_group_dict.values():
        for mid in rg.machine_id_list:
            machine_last_end[mid] = 0.0
            machine_prev_tech_type[mid] = -1
            machine_workload[mid] = []
        for wid in rg.worker_id_list:
            worker_task_list[wid] = []
            worker_workload[wid] = []
    for idx, op_id in enumerate(seq):
        mac_id, worker_id = assign[idx]
        if state_manager.is_op_manual_locked(op_id):
            lock_cfg = state_manager.manual_lock_dict[op_id]
            original_mac_id = mac_id
            original_worker_id = worker_id
            if lock_cfg.lock_machine:
                mac_id = lock_cfg.fixed_machine_id
            if lock_cfg.lock_worker:
                worker_id = lock_cfg.fixed_worker_id
            if lock_cfg.lock_worker and not lock_cfg.lock_machine:
                rg = state_manager.get_resource_group_by_op(op_id)
                valid_macs = [m for m in rg.machine_id_list if state_manager.machine_available_dict.get(m, False)]
                mac_id = np.random.choice(valid_macs) if valid_macs else original_mac_id
            if lock_cfg.lock_machine and not lock_cfg.lock_worker:
                rg = state_manager.get_resource_group_by_op(op_id)
                valid_workers = [w for w in rg.worker_id_list if state_manager.worker_available_dict.get(w, False)]
                worker_id = np.random.choice(valid_workers) if valid_workers else original_worker_id
        op_meta = state_manager.op_meta_dict[op_id]
        job_id = op_meta.belong_job_id
        base_proc_t = op_meta.process_time
        tech_type = op_meta.op_tech_type
        material_earliest = op_meta.material_ready_time
        job_op_index = op_meta.op_index_in_job
        speed_ratio = 1.0
        if worker_id != -1:
            skill_info = state_manager.worker_skill_dict.get(worker_id)
            if skill_info is not None and tech_type in skill_info.tech_speed_ratio:
                speed_ratio = skill_info.tech_speed_ratio[tech_type]
        real_process_time = base_proc_t * speed_ratio
        job_pre_end = job_last_op_end.get(job_id, 0.0)
        machine_free_time = machine_last_end.get(mac_id, 0.0)
        rg = state_manager.get_resource_group_by_op(op_id)
        max_parallel = rg.worker_max_parallel
        ideal_start = max(machine_free_time, job_pre_end, material_earliest)
        active_tasks = []
        if worker_id != -1:
            active_tasks = [(s,e) for s,e in worker_task_list[worker_id] if not (e < ideal_start)]
        worker_earliest = 0.0
        if len(active_tasks) >= max_parallel:
            worker_earliest = min([e for s,e in active_tasks])
        ideal_start = max(ideal_start, worker_earliest)
        start_time = state_manager.get_valid_start_time(ideal_start)
        if mac_id != -1:
            machine_prev_tech = machine_prev_tech_type[mac_id]
            if machine_prev_tech != -1 and machine_prev_tech != tech_type:
                tech_param = state_manager.machine_tech_dict[mac_id]
                inner_dict = tech_param.changeover_time_map.get(machine_prev_tech, {})
                cot = inner_dict.get(tech_type, 0.0)
                total_changeover_time += cot
                start_time = state_manager.calculate_actual_work_end_time(start_time, cot)
            machine_prev_tech_type[mac_id] = tech_type
        end_time = state_manager.calculate_actual_work_end_time(start_time, real_process_time)
        frozen_boundary = state_manager.current_system_time + cfg.PLAN_FROZEN_HORIZON
        is_frozen = (start_time < frozen_boundary)
        if job_id in job_last_op_end:
            prev_end = job_last_op_end[job_id]
            if prev_end < start_time:
                total_wip_wait_time += (start_time - prev_end)
        worker_switch_count += 1
        if mac_id != -1:
            machine_last_end[mac_id] = end_time
            machine_workload[mac_id].append(end_time - start_time)
        if worker_id != -1:
            worker_task_list[worker_id].append((start_time, end_time))
            worker_workload[worker_id].append(end_time - start_time)
        job_last_op_end[job_id] = end_time
        rg = state_manager.get_resource_group_by_op(op_id)
        if mac_id == -1:
            mac_id = rg.machine_id_list[0]
        if worker_id == -1:
            worker_id = rg.worker_id_list[0]
        schedule_detail.append({
            "op_id": op_id,
            "job_id": job_id,
            "job_op_index": job_op_index,
            "business_op_id": op_meta.business_op_id,
            "business_op_no": op_meta.business_op_no,
            "op_name": op_meta.op_name,
            "op_content": op_meta.op_content,
            "resource_group_id": op_meta.resource_group_id,
            "resource_group_name": op_meta.resource_group_name,
            "machine_id": mac_id,
            "worker_id": worker_id,
            "start_time": start_time,
            "end_time": end_time,
            "real_start_time": state_manager.relative_hour_to_iso(start_time),
            "real_end_time": state_manager.relative_hour_to_iso(end_time),
            "tech_type": tech_type,
            "real_process_time": real_process_time,
            "is_frozen": is_frozen,
            "is_manual_locked": state_manager.is_op_manual_locked(op_id)
        })
    makespan = max(machine_last_end.values()) if machine_last_end else 0.0
    machine_total_load = [sum(v) for v in machine_workload.values()]
    machine_overload_punish = 0.0
    for mid, total_load in zip(machine_workload.keys(), machine_total_load):
        machine_overload_punish += state_manager.get_machine_overload_penalty(mid, total_load)
    for job_id, finish_t in job_last_op_end.items():
        job_meta = state_manager.job_meta_dict.get(job_id)
        if job_meta:
            segment_overdue_penalty += state_manager.calc_segment_overdue_penalty(finish_t, job_meta)
    total_overdue_cost = segment_overdue_penalty
    machine_var = np.var(machine_total_load) if len(machine_total_load) > 1 else 0.0
    machine_unbalance = machine_var
    worker_total_load = [sum(v) for v in worker_workload.values()]
    worker_var = np.var(worker_total_load) if len(worker_total_load) > 1 else 0.0
    worker_switch_cost = worker_switch_count * WORKER_SWITCH_COST
    worker_unbalance = worker_var + worker_switch_cost
    weighted_wip_total = total_wip_wait_time * WIP_WEIGHT_COEFFICIENT
    fit_vector = [makespan, total_overdue_cost, total_changeover_time, machine_unbalance, worker_unbalance, weighted_wip_total]
    fit_vector[0] += machine_overload_punish
    return fit_vector, schedule_detail


def fast_non_dominated_sorting(pop_fits: List[List[float]]) -> Tuple[List[List[int]], List[int]]:
    """通用快速非支配排序，快速非支配排序流程：
        1. 初始统计得到原始 dom_count：记录一共多少个体支配自己；
        2. 先把 dom_count=0 的划为 rank=0；
        3. 遍历 rank0 所有个体，把它们支配的所有个体 dom_count -= 1；
        4. 新 dom_count=0 的这批划为 rank=1；
        5. 再遍历 rank1，继续扣减下层 dom_count，产生 rank=2……"""

    def is_dominates(candidate: List[float], other: List[float]) -> bool:
        """
        判断 candidate 是否支配 other（针对最小化问题）。
        支配条件：candidate 在所有目标上 <= other，且至少有一个目标严格 < other。
        """
        all_less_or_equal = all(
            candidate_val <= other_val
            for candidate_val, other_val in zip(candidate, other)
        )
        any_strictly_better = any(
            candidate_val < other_val
            for candidate_val, other_val in zip(candidate, other)
        )

        return all_less_or_equal and any_strictly_better

    def extract_fronts(dominated: List[List[int]], dom_count: List[int]) -> Tuple[List[List[int]], List[int]]:
        """
        利用支配关系，通过迭代剥离的方式构建非支配前沿。
        返回 fronts 和 rank。
        """
        n = len(dom_count)
        rank = [0] * n  # rank[i] = 个体 i 所属的帕累托前沿层级编号
        fronts = [[]]   # 各层前沿的索引列表，fronts[0] = 第一层前沿所有个体的索引

        # 第一层
        for p in range(n):
            if dom_count[p] == 0:   # 如果支配p的个体数为0
                rank[p] = 0
                fronts[0].append(p)

        # 后续层
        front_idx = 0
        while front_idx < len(fronts):  # 这里front是个动态值，fronts.append后，会变大一个值。
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

    n = len(pop_fits)
    # 关键数据结构1：dominated[p] = 所有被p支配的个体的索引列表,简称：p支配的个体列表。
    dominated: List[List[int]] = [[] for _ in range(n)]
    # 关键数据结构2：dom_count[p] = 支配p的个体的数量（即p被多少个个体支配）
    dom_count: List[int] = [0]*n

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if is_dominates(pop_fits[p], pop_fits[q]):  # 情况1：p支配q → 把q加入p的支配列表
                dominated[p].append(q)
            elif is_dominates(pop_fits[q], pop_fits[p]):    # 情况2：q支配p → p的被支配计数器+1
                dom_count[p] += 1

    fronts, rank = extract_fronts(dominated, dom_count)
    return fronts, rank





def pox_crossover(p1: dict, p2: dict, state_manager: ProductionStateManager) -> Tuple[dict, dict]:
    """通用POX工序交叉算子，NSGA-2/NSGA-3共用
    最核心两大底层本质
    强制保证子代一定是合法可行解：同一个订单（工件）内部工序的先后加工顺序永远不会被打乱，不会生成违背工艺约束的非法排程；
    按订单（工件）分组交叉，不是按单个工序乱交换，高效继承父代优良排序片段，兼顾全局搜索能力。
    1. 收集全部订单集合，随机拆分成两个非空分组
    2. 构造子代 1（C1）规则
        2.1从头到尾遍历父代 P1，把归属Group1订单的工序按原有先后顺序存入 C1；
        2.2再从头到尾遍历父代 P2，把归属Group2订单的工序按原有先后顺序追加到 C1 末尾；
    3. 构造子代 2（C2）对称互补规则
        3.1从头到尾遍历父代 P2，把归属Group1订单的工序按原有先后顺序存入 C2；
        3.2再从头到尾遍历父代 P1，把归属Group2订单的工序按原有先后顺序追加到 C2 末尾；
    """

    def get_frozen_operations(seq: List[int], state_manager, frozen_boundary: float) -> List[int]:
        """
        返回序列中所有开始时间早于冻结边界的操作（按原序列顺序）。
        """
        frozen = []
        last_schedule = state_manager.last_schedule_result
        for op_id in seq:
            if op_id in last_schedule and last_schedule[op_id]["start_time"] < frozen_boundary:
                frozen.append(op_id)
        return frozen

    def extract_free_operations(seq: List[int], frozen_ops: List[int]) -> List[int]:
        """
        假设冻结操作均位于序列开头，提取剩余的自由操作（保持原有相对顺序）。
        若冻结操作并非全部位于前缀，此方法将失效——但算法设计保证冻结操作在前。
        """
        return seq[len(frozen_ops):]

    def pox_crossover_free_sequences(
            free_ops1: List[int],
            free_ops2: List[int],
            state_manager
    ) -> Tuple[List[int], List[int]]:
        """
        对自由操作序列执行 POX 交叉（基于作业分组）。
        返回两个子代的自由操作序列。
        """
        op_to_job = state_manager.operation_id_to_job_id
        # 取两个父代自由操作中所有作业的并集
        job_set = set(op_to_job[op] for op in free_ops1 + free_ops2)
        job_list = list(job_set)

        if len(job_list) <= 1:
            # 仅一个作业时无需交叉，直接拷贝
            return free_ops1.copy(), free_ops2.copy()

        # 随机划分作业组
        split_idx = np.random.randint(1, len(job_list))
        group1 = set(job_list[:split_idx])
        group2 = set(job_list[split_idx:])

        child1_free = []
        child2_free = []

        # 核心是，我爸的属于第一组的放前头，对方爸属于第二组的放后头。保持相对顺序不变
        # 子代1：继承父代1中属于组1的操作 + 父代2中属于组2的操作
        for op in free_ops1:
            if op_to_job[op] in group1:
                child1_free.append(op)
        for op in free_ops2:
            if op_to_job[op] in group2:
                child1_free.append(op)

        # 子代2：继承父代2中属于组1的操作 + 父代1中属于组2的操作
        for op in free_ops2:
            if op_to_job[op] in group1:
                child2_free.append(op)
        for op in free_ops1:
            if op_to_job[op] in group2:
                child2_free.append(op)

        return child1_free, child2_free

    def random_select_resources(op_id: int, state_manager) -> Tuple[int, int]:
        """
        为给定操作随机选取一台可用机器和一名可用工人。
        """
        rg = state_manager.get_resource_group_by_op(op_id)
        avail_machines = [m for m in rg.machine_id_list if state_manager.machine_available_dict[m]]
        avail_workers = [w for w in rg.worker_id_list if state_manager.worker_available_dict[w]]
        machine = np.random.choice(avail_machines)
        worker = np.random.choice(avail_workers)
        return machine, worker

    def build_child_assignments(
            child_seq: List[int],
            parent_seq: List[int],
            parent_assign: List[Tuple[int, int]],
            frozen_ops: List[int],
            state_manager
    ) -> List[Tuple[int, int]]:
        """
        根据子代工序序列构建资源分配列表。
        冻结操作继承自上次调度结果，非冻结操作优先从父代继承，否则随机生成。
        """
        op_to_assign = dict(zip(parent_seq, parent_assign))
        last_schedule = state_manager.last_schedule_result
        child_assign = []

        for op_id in child_seq:
            if op_id in frozen_ops:
                # 冻结操作必须使用上次调度的机器与工人
                res = last_schedule[op_id]
                child_assign.append((res["machine_id"], res["worker_id"]))
            else:
                # 存在父代映射的直接取父代分配的资源，保证滚动排程业务连续性
                if op_id in op_to_assign:
                    child_assign.append(op_to_assign[op_id])
                else:
                    # 安全兜底：正常情况下不会进入此分支
                    child_assign.append(random_select_resources(op_id, state_manager))
        return child_assign

    seq1, assign1 = p1["op_sequence"], p1["resource_assign"]
    seq2, assign2 = p2["op_sequence"], p2["resource_assign"]

    frozen_boundary = state_manager.current_system_time + cfg.PLAN_FROZEN_HORIZON
    frozen_ops = get_frozen_operations(seq1, state_manager, frozen_boundary)

    # 若全部操作被冻结，则直接返回父代副本
    if len(frozen_ops) == len(seq1):
        return copy.deepcopy(p1), copy.deepcopy(p2)

    # 提取自由操作（冻结操作位于序列前端）
    free_ops1 = extract_free_operations(seq1, frozen_ops)
    free_ops2 = extract_free_operations(seq2, frozen_ops)

    # POX 交叉产生子代自由序列
    child1_free, child2_free = pox_crossover_free_sequences(
        free_ops1, free_ops2, state_manager
    )

    # 合并冻结部分与自由部分
    child1_seq = frozen_ops + child1_free
    child2_seq = frozen_ops + child2_free

    # 构建子代的资源分配
    child1_assign = build_child_assignments(
        child1_seq, seq1, assign1, frozen_ops, state_manager
    )
    child2_assign = build_child_assignments(
        child2_seq, seq2, assign2, frozen_ops, state_manager
    )

    return (
        {"op_sequence": child1_seq, "resource_assign": child1_assign},
        {"op_sequence": child2_seq, "resource_assign": child2_assign},
    )


def mutate_chromosome(chrom: dict, state_manager: ProductionStateManager) -> dict:
    """通用染色体变异算子，NSGA-2/NSGA-3共用"""
    def is_operation_frozen(op_id: int, frozen_boundary, state_manager: ProductionStateManager) -> bool:
        last_result =  state_manager.last_schedule_result.get(op_id)
        if last_result is None:
            return False
        last_start_time = last_result.get("start_time")
        if last_start_time is None:
            return False
        return last_start_time < frozen_boundary

    new_chrom = copy.deepcopy(chrom)
    seq = new_chrom["op_sequence"]
    resource_assign = new_chrom["resource_assign"]
    frozen_boundary = state_manager.current_system_time + cfg.PLAN_FROZEN_HORIZON
    for idx, op_id in enumerate(seq):
        if is_operation_frozen(op_id, frozen_boundary, state_manager):
            continue
        if np.random.random() > cfg.MUTATION_RATE:
            continue
        resource_group = state_manager.get_resource_group_by_op(op_id)
        available_machines = [machine for machine in resource_group.machine_id_list if state_manager.machine_available_dict[machine]]
        available_workers = [worker for worker in resource_group.worker_id_list if state_manager.worker_available_dict[worker]]
        if available_machines and available_workers:
            resource_assign[idx] = (np.random.choice(available_machines), np.random.choice(available_workers))
    new_chrom["resource_assign"] = resource_assign
    return new_chrom

def select_optimal_solution(pareto_set: List[dict], all_pop_fits: List[List[float]], pareto_index_list: List[int]):
    """8目标加权择优，权重顺序：
    [fit1, fit2, fit3, fit4, fit5, fit6, fit7, fit8]
    """
    # 可根据工厂考核优先级自行调整权重
    weight = [0.22, 0.16, 0.15, 0.14, 0.10, 0.09, 0.07, 0.07]
    score_list = []
    for idx in pareto_index_list:
        fit_vec = all_pop_fits[idx]
        total_score = sum(weight[i] * fit_vec[i] for i in range(len(weight)))
        score_list.append(total_score)
    best_in_pareto_idx = np.argmin(score_list)
    real_pop_idx = pareto_index_list[best_in_pareto_idx]
    return pareto_set[best_in_pareto_idx], all_pop_fits[real_pop_idx]

def evaluate_population_fitness(population: List[Dict[str, Any]],state_manager: ProductionStateManager) -> List[List[float]]:
    """批量解码种群，返回每个个体的适应度向量。"""
    fitness_list = []
    for chrom in population:
        fit_value, _ = decode_chromosome(chrom, state_manager)
        fitness_list.append(fit_value)
    return fitness_list


def _validate_chromosome_input(chromosome: Dict[str, Any], state_manager: Any) -> None:
    """验证染色体输入的合法性，提前发现错误避免后续调试困难"""
    if "op_sequence" not in chromosome or "resource_assign" not in chromosome:
        raise ValueError("染色体必须包含'op_sequence'和'resource_assign'两个核心字段")

    op_sequence = chromosome["op_sequence"]
    resource_assign = chromosome["resource_assign"]

    if len(op_sequence) != len(resource_assign):
        raise ValueError(
            f"操作序列长度({len(op_sequence)})与资源分配长度({len(resource_assign)})不匹配"
        )

    for op_id in op_sequence:
        if op_id not in state_manager.op_meta_dict:
            raise ValueError(f"未知的操作ID: {op_id}")

def _initialize_tracking_structures(state_manager: Any) -> SchedulingTrackers:
    """根据生产状态初始化所有跟踪数据结构"""
    trackers = SchedulingTrackers()

    for resource_group in state_manager.resource_group_dict.values():
        # 初始化机器
        for machine_id in resource_group.machine_id_list:
            typed_machine_id = MachineId(machine_id)
            trackers.machine_last_end_time[typed_machine_id] = 0.0
            trackers.machine_previous_technology_type[typed_machine_id] = -1
            # 单台机器总可用工时（排程总区间时长，由state_manager提供）
            trackers.machine_total_available_hour[typed_machine_id] = state_manager.get_schedule_total_horizon()
            trackers.machine_total_process_hour[typed_machine_id] = 0.0

        # 初始化工人
        for worker_id in resource_group.worker_id_list:
            typed_worker_id = WorkerId(worker_id)
            trackers.worker_task_intervals[typed_worker_id] = []
            trackers.worker_task_ends_heap[typed_worker_id] = []

    return trackers

def _apply_manual_locks(
        operation_id: str,
        original_machine_id: MachineId,
        original_worker_id: WorkerId,
        trackers: SchedulingTrackers,
        state_manager: Any
) -> Tuple[MachineId, WorkerId]:
    """
    应用手动锁定规则
    核心原则：优先保留算法进化出的资源分配，仅在锁定时替换
    """
    if not state_manager.is_op_manual_locked(operation_id):
        return original_machine_id, original_worker_id

    lock_config = state_manager.manual_lock_dict[operation_id]
    machine_id = original_machine_id
    worker_id = original_worker_id

    # 应用机器锁定
    if lock_config.lock_machine:
        machine_id = MachineId(lock_config.fixed_machine_id)
        logger.debug(f"操作[{operation_id}] 手动锁定机器: {original_machine_id} -> {machine_id}")

    # 应用工人锁定
    if lock_config.lock_worker:
        worker_id = WorkerId(lock_config.fixed_worker_id)
        logger.debug(f"操作[{operation_id}] 手动锁定工人: {original_worker_id} -> {worker_id}")

    # 只锁工人不锁机器：优先保留原可用的机器，不可用时选最优机器
    if lock_config.lock_worker and not lock_config.lock_machine:
        if not state_manager.machine_available_dict.get(original_machine_id, False):
            resource_group = state_manager.get_resource_group_by_op(operation_id)
            valid_machines = [
                MachineId(m) for m in resource_group.machine_id_list
                if state_manager.machine_available_dict[m]
            ]
            if not valid_machines:  # 优化，这里不能终止程序，需将其排程到虚拟设备上
                raise ValueError(f"操作[{operation_id}] 锁定工人后无可用机器")

            machine_id = select_best_machine(valid_machines, trackers)
            logger.debug(f"操作[{operation_id}] 原机器不可用，自动选择: {machine_id}")

    # 只锁机器不锁工人：优先保留原可用工人，不可用时选最优工人
    if lock_config.lock_machine and not lock_config.lock_worker:
        if not state_manager.worker_available_dict.get(original_worker_id, False):
            resource_group = state_manager.get_resource_group_by_op(operation_id)
            valid_workers = [
                WorkerId(w) for w in resource_group.worker_id_list
                if state_manager.worker_available_dict[w]
            ]
            if not valid_workers:   # 优化，这里不能终止程序，需将其排程到虚拟人员上
                raise ValueError(f"操作[{operation_id}] 锁定机器后无可用工人")

            # ✅ 已修复：使用统一的最优工人选择逻辑
            worker_id = select_best_worker(valid_workers, trackers)
            logger.debug(f"操作[{operation_id}] 原工人不可用，自动选择: {worker_id}")

    return machine_id, worker_id

def _ensure_valid_resource(
        operation_id: str,
        machine_id: MachineId,
        worker_id: WorkerId,
        trackers: SchedulingTrackers,
        state_manager: Any
) -> Tuple[MachineId, WorkerId]:
    """
    确保资源ID有效（处理-1和不可用资源）
    必须在调度前调用，避免跟踪器中出现-1的无效键
    """
    resource_group = state_manager.get_resource_group_by_op(operation_id)

    # 处理无效机器ID
    if machine_id == MachineId(-1) or not state_manager.machine_available_dict.get(machine_id, False):
        valid_machines = [
            MachineId(m) for m in resource_group.machine_id_list
            if state_manager.machine_available_dict[m]
        ]
        if not valid_machines:
            raise ValueError(f"操作[{operation_id}] 无可用机器")

        machine_id = select_best_machine(valid_machines, trackers)
        logger.debug(f"操作[{operation_id}] 机器无效，自动分配: {machine_id}")

    # 处理无效工人ID
    if worker_id == WorkerId(-1) or not state_manager.worker_available_dict.get(worker_id, False):
        valid_workers = [
            WorkerId(w) for w in resource_group.worker_id_list
            if state_manager.worker_available_dict[w]
        ]
        if not valid_workers:
            raise ValueError(f"操作[{operation_id}] 无可用工人")

        # 使用统一的最优工人选择逻辑
        worker_id = select_best_worker(valid_workers, trackers)
        logger.debug(f"操作[{operation_id}] 工人无效，自动分配: {worker_id}")

    return machine_id, worker_id

def _schedule_single_operation(
        operation_id: str,
        machine_id: MachineId,
        worker_id: WorkerId,
        trackers: SchedulingTrackers,
        state_manager: Any,
        config: Dict[str, Any]
) -> OperationSchedulingResult:
    """
    单个操作的调度计算核心
    计算开始/结束时间，处理所有约束条件
    """
    operation_metadata = state_manager.op_meta_dict[operation_id]
    job_id = operation_metadata.belong_job_id
    technology_type = operation_metadata.op_tech_type

    # --------------------------
    # 1. 计算实际加工时间（考虑工人技能系数）
    # --------------------------
    base_processing_time = operation_metadata.process_time
    speed_ratio = 1.0
    if worker_id != WorkerId(-1):
        skill_info = state_manager.worker_skill_dict.get(worker_id)
        if skill_info and technology_type in skill_info.tech_speed_ratio:
            speed_ratio = skill_info.tech_speed_ratio[technology_type]
    actual_processing_time = base_processing_time * speed_ratio

    # --------------------------
    # 2. 计算理想最早开始时间（基础约束）
    # --------------------------
    machine_available_time = trackers.machine_last_end_time[machine_id]
    job_available_time = trackers.job_last_operation_end_time.get(job_id, 0.0)
    material_ready_time = operation_metadata.material_ready_time
    ideal_start_time = max(machine_available_time, job_available_time, material_ready_time)

    # --------------------------
    # 3. 处理工人并行任务约束（O(log n)时间复杂度）
    # --------------------------
    if worker_id != WorkerId(-1):
        resource_group = state_manager.get_resource_group_by_op(operation_id)
        max_parallel = resource_group.worker_max_parallel

        while True:
            # 先清理堆中已结束的任务
            while (trackers.worker_task_ends_heap[worker_id]
                   and trackers.worker_task_ends_heap[worker_id][0] <= ideal_start_time):
                heapq.heappop(trackers.worker_task_ends_heap[worker_id])

            # 检查是否满足并行限制
            if len(trackers.worker_task_ends_heap[worker_id]) < max_parallel:
                break

            # 等待最早的任务结束
            earliest_end = trackers.worker_task_ends_heap[worker_id][0]
            ideal_start_time = earliest_end

    # --------------------------
    # 4. 处理机器换型时间（提前换型优化）
    # --------------------------
    previous_tech = trackers.machine_previous_technology_type[machine_id]
    if previous_tech != -1 and previous_tech != technology_type:
        changeover_map = state_manager.machine_tech_dict[machine_id].changeover_time_map
        changeover_time = changeover_map.get(previous_tech, {}).get(technology_type, 0.0)
        trackers.total_changeover_time += changeover_time

        # 换型从机器可用时间开始
        changeover_end = state_manager.calculate_actual_work_end_time(machine_available_time, changeover_time)
        ideal_start_time = max(ideal_start_time, changeover_end)

    # 更新机器工艺类型（无论是否换型都要更新）
    trackers.machine_previous_technology_type[machine_id] = technology_type

    # --------------------------
    # 5. 应用班次日历约束，获取合法开始/结束时间
    # --------------------------
    start_time = state_manager.get_valid_start_time(ideal_start_time)
    end_time = state_manager.calculate_actual_work_end_time(start_time, actual_processing_time)

    # --------------------------
    # 6. 判断是否处于计划冻结区间
    # --------------------------
    frozen_time_limit = state_manager.current_system_time + cfg.PLAN_FROZEN_HORIZON
    is_frozen = start_time < frozen_time_limit

    # --------------------------
    # 7. 构造调度结果对象
    # --------------------------
    return OperationSchedulingResult(
        operation_id=operation_id,
        job_id=job_id,
        operation_index_in_job=operation_metadata.op_index_in_job,
        machine_id=machine_id,
        worker_id=worker_id,
        start_time=start_time,
        end_time=end_time,
        actual_processing_time=actual_processing_time,
        technology_type=technology_type,
        is_frozen=is_frozen,
        is_manual_locked=state_manager.is_op_manual_locked(operation_id),
        operation_metadata=operation_metadata
    )

def _update_trackers(result: OperationSchedulingResult, trackers: SchedulingTrackers) -> None:
    """用调度结果更新所有全局状态"""
    job_id = result.job_id
    machine_id = result.machine_id
    worker_id = result.worker_id
    start_time = result.start_time
    end_time = result.end_time
    actual_processing_time = result.actual_processing_time

    # 1. 累计在制品等待时间
    if job_id in trackers.job_last_operation_end_time:
        previous_end = trackers.job_last_operation_end_time[job_id]
        if previous_end < start_time:
            trackers.total_wip_wait_time += (start_time - previous_end)

    # 2. 更新机器状态 + 累加实际加工工时
    trackers.machine_last_end_time[machine_id] = end_time
    trackers.machine_workloads[machine_id].append(actual_processing_time)
    trackers.machine_total_process_hour[machine_id] += actual_processing_time

    #
    # 3. 更新工人状态
    if worker_id != WorkerId(-1):
        # 统计工人切换次数（只有连续任务才不算切换）# 这里待把工人切换次数定义清楚了再使用。
        # if trackers.worker_task_intervals[worker_id]:
        #     last_task_end = trackers.worker_task_intervals[worker_id][-1][1]
        #     if last_task_end == start_time:
        #         trackers.worker_switch_count += 1

        # 更新任务区间和结束时间堆
        trackers.worker_task_intervals[worker_id].append((start_time, end_time))
        heapq.heappush(trackers.worker_task_ends_heap[worker_id], end_time)
        trackers.worker_workloads[worker_id].append(actual_processing_time)

    # 4. 更新工件状态
    trackers.job_last_operation_end_time[job_id] = end_time

    # 5. 总工序计数
    trackers.total_operation_count += 1

def _build_schedule_record(result: OperationSchedulingResult, state_manager: Any) -> Dict[str, Any]:
    """生成前端展示和持久化用的排程记录字典"""
    op_meta = result.operation_metadata

    return {
        "op_id": result.operation_id,
        "job_id": result.job_id,
        "job_op_index": result.operation_index_in_job,
        "business_op_id": op_meta.business_op_id,
        "business_op_no": op_meta.business_op_no,
        "op_name": op_meta.op_name,
        "op_content": op_meta.op_content,
        "resource_group_id": op_meta.resource_group_id,
        "resource_group_name": op_meta.resource_group_name,
        "machine_id": int(result.machine_id),  # 转换为普通int方便JSON序列化
        "worker_id": int(result.worker_id),  # 转换为普通int方便JSON序列化
        "start_time": result.start_time,
        "end_time": result.end_time,
        "real_start_time": state_manager.relative_hour_to_iso(result.start_time),
        "real_end_time": state_manager.relative_hour_to_iso(result.end_time),
        "tech_type": result.technology_type,
        "real_process_time": result.actual_processing_time,
        "is_frozen": result.is_frozen,
        "is_manual_locked": result.is_manual_locked,
    }

def _compute_fitness_vector(
        trackers: SchedulingTrackers,
        state_manager: Any,
        config: Dict[str, Any]
) -> List[float]:
    """
    8维多目标适应度（全部最小化，顺序严格对齐需求）
    fit[0] = fit1：逾期订单总数
    fit[1] = fit2：订单逾期总惩罚成本
    fit[2] = fit3：总工期+机器超载惩罚
    fit[3] = fit4：设备整体闲置率
    fit[4] = fit5：设备总换型时间
    fit[5] = fit6：机器负载不平衡度（工时方差）
    fit[6] = fit7：工人负载不平衡度（工时方差）
    fit[7] = fit8：在制品加权等待总成本
    """

    # ===================== fit0: 逾期订单总数、fit1: 订单分段逾期总惩罚 =====================
    overdue_count = 0
    total_overdue_penalty = 0.0
    for job_id, finish_time in trackers.job_last_operation_end_time.items():
        job_meta = state_manager.job_meta_dict.get(job_id)
        if not job_meta:
            continue
        penalty = state_manager.calc_segment_overdue_penalty(finish_time, job_meta)
        total_overdue_penalty += penalty
        if finish_time > job_meta.due_contract_time:
            overdue_count += 1

    # ===================== fit2: 总工期 + 机器超载惩罚 =====================
    makespan = max(trackers.machine_last_end_time.values()) if trackers.machine_last_end_time else 0.0
    machine_overload_penalty = 0.0
    for machine_id, workloads in trackers.machine_workloads.items():
        total_load = sum(workloads)
        machine_overload_penalty += state_manager.get_machine_overload_penalty(machine_id, total_load)
    makespan_and_penalty = makespan + machine_overload_penalty

    # ===================== fit3: 设备整体闲置率 = 总空闲时长 / 总可用工时 =====================
    total_all_available_time = 0.0
    total_all_idle_time = 0.0
    for mid, avail_h in trackers.machine_total_available_hour.items():
        proc_h = trackers.machine_total_process_hour[mid]
        idle_h = max(0.0, avail_h - proc_h)
        total_all_available_time += avail_h
        total_all_idle_time += idle_h
    if total_all_available_time > 1e-9:
        overall_equipment_idle_rate = total_all_idle_time / total_all_available_time
    else:
        overall_equipment_idle_rate = 0.0

    # ===================== fit4: 设备总换型时间 =====================
    machine_total_changeover_time = trackers.total_changeover_time

    # ===================== fit5: 机器负载不平衡度（加工工时方差） =====================
    machine_total_loads = [sum(loads) for loads in trackers.machine_workloads.values()]
    machine_unbalance = np.var(machine_total_loads) if len(machine_total_loads) > 1 else 0.0

    # ===================== fit6: 工人负载不平衡度（仅工时方差，去掉切换成本） =====================
    worker_total_loads = [sum(loads) for loads in trackers.worker_workloads.values()]
    worker_unbalance = np.var(worker_total_loads) if len(worker_total_loads) > 1 else 0.0

    # ===================== fit7: 在制品加权等待总成本 =====================
    wip_cost = trackers.total_wip_wait_time * cfg.WIP_WEIGHT_COEFFICIENT

    # 组装最终适应度向量
    return [
        overdue_count,
        total_overdue_penalty,
        makespan_and_penalty,
        overall_equipment_idle_rate,
        machine_total_changeover_time,
        machine_unbalance,
        worker_unbalance,
        wip_cost
    ]