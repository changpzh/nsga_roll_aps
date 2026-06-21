# core/base_ga.py
import numpy as np
import heapq
import copy
from typing import List, Dict, Tuple,Any
from models import MachineId, WorkerId, OperationId, JobId, SchedulingTrackers, OperationSchedulingResult
from core.state_manager import ProductionStateManager
from utils.common_utils import select_best_worker, select_best_machine
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
    all_optimizable_ops_set = set(state_manager.get_optimizable_operation_ids())

    for job_id in reorder_job_seq:
        # 取出该订单绑定的全部工序ID
        job_op_list = [op for op, jid in state_manager.operation_id_to_job_id.items() if jid == job_id]
        # 过滤：只保留属于可优化池的工序
        valid_ops = [op for op in job_op_list if op in all_optimizable_ops_set]
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

    job_id_list = list(job_op_map.keys())

    # shuffle_free=True时随机打乱订单先后顺序；启发式调用传False则保持原有订单排序
    if shuffle_free:
        np.random.shuffle(job_id_list)

    # 按订单顺序拼接浮动工序（订单内部工序早已按工艺号排好）
    free_sequence = []
    for jid in job_id_list:
        free_sequence.extend(job_op_map[jid])

    # 最终全局工序顺序：冻结工序固定在前，浮动工序在后
    op_sequence = frozen_ops + free_sequence

    # ====================== 步骤4：逐工序分配机床、工人资源, 同订单的连续工序复用前工序的设备，减少换型时间 ======================
    job_prev_machine_map: Dict[int, MachineId] = {}
    resource_assign = []
    for op_id in op_sequence:
        if op_id in frozen_ops:
            mid = state_manager.last_schedule_result[op_id]["machine_id"]
            wid = state_manager.last_schedule_result[op_id]["worker_id"]
            resource_assign.append((mid, wid))
            job_prev_machine_map[state_manager.op_meta_dict[op_id].belong_job_id] = mid
            continue
        # 判断能否复用前道同组机床
        reuse_mid = try_get_prev_same_group_machine(
            current_op_id=op_id,
            op_sequence=op_sequence,
            resource_assign=resource_assign,
            state_manager=state_manager
        )

        rg = state_manager.get_resource_group_by_op(op_id)
        available_machines = state_manager.get_available_machines(rg.machine_id_list)
        available_workers = state_manager.get_available_workers(rg.worker_id_list)
        if reuse_mid is not None:
            selected_machine_id = reuse_mid
        else:
            selected_machine_id = np.random.choice(available_machines)

        selected_worker_id = np.random.choice(available_workers)
        resource_assign.append((selected_machine_id, selected_worker_id))

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
            # 裁剪：保留冻结、完工、正在加工的锁定工序，保留本次可优化浮动工序的资源，新增工序随机分配可用资源
            trimmed_chrom = trim_historical_chromosome(base_chrom, state_manager)
            # 对浮动工序小幅局部扰动（交换顺序/调换同组设备工人），生成新种子
            new_history_seed = perturb_historical_chromosome(trimmed_chrom, state_manager)
            population.append(new_history_seed)
    else:
        # 冷启动场景：无历史解集，历史配额全部用随机可行染色体填充兜底
        for _ in range(history_num):
            random_chrom = init_single_chromosome(reorder_job_seq, state_manager)
            population.append(random_chrom)

    # -------------------------- 步骤4：启发式生成染色体种子个体 --------------------------
    for _ in range(heuristic_num):
        # 按业务优先级、交期规则定向构建高质量初始排程方案
        heuristic_chromosome = generate_heuristic_chromosome(reorder_job_seq, state_manager)
        population.append(heuristic_chromosome)

    # -------------------------- 步骤5：生成纯随机可行种子个体 --------------------------
    for _ in range(random_num):
        # 随机编码浮动工序，解码阶段自动矫正所有硬约束，保证输出可行解
        random_chrom = init_single_chromosome(reorder_job_seq, state_manager)
        population.append(random_chrom)

    # -------------------------- 步骤6：返回完整混合种群 --------------------------
    return population


def perturb_historical_chromosome(trimmed_chrom: dict, state_manager: ProductionStateManager) -> dict:
    """
    现在这个函数只会扰动同时满足以下所有条件的工序：
    1. 状态为 OP_STATUS_OPTIMIZABLE（未开工、未取消）
    2. 开工时间在冻结窗口之外
    3. 没有被手动锁定
    4. 属于普通优先级订单（高 / 紧急优先级只有 40% 概率被扰动）
    """
    chrom = copy.deepcopy(trimmed_chrom)
    opt_ops = set(state_manager.get_optimizable_operation_ids())
    op_seq = chrom["op_sequence"]
    assign = chrom["resource_assign"]
    total_op_len = len(op_seq)

    # 计算冻结边界
    freeze_boundary = state_manager.current_system_time + cfg.PLAN_FROZEN_HORIZON
    last_schedule = state_manager.last_schedule_result

    # 滚动扰动比例，通常配置为 0.05~0.2（5%~20%）,max(1, ...)：保证至少扰动1个工序，避免扰动失效
    perturb_num = max(1, int(total_op_len * ROLLING_PERTURB_RATE)) # 计算本次要扰动的工序数量

    # 随机从range（total_op_len）中选择 perturb_num 个索引位置，replace=False表示不放回，保证不重复的索引位置
    perturb_indexes = np.random.choice(range(total_op_len), perturb_num, replace=False)
    # 遍历选中位置，执行差异化扰动
    for idx in perturb_indexes:
        op_id = op_seq[idx]
        # 过滤1：非可优化工序（已完工/运行中/冻结区域）直接跳过，绝对不能扰动
        # 过滤2：手动锁定的工序直接跳过，尊重人工调度决策
        # 过滤3：冻结区域内的未开工工序
        if (op_id not in opt_ops
            or state_manager.is_op_manual_locked(op_id))\
            or (op_id in last_schedule and last_schedule[op_id]["start_time"] < freeze_boundary):
            continue
        # 获取该工序所属订单的优先级
        jid = state_manager.operation_id_to_job_id[op_id]
        job_meta = state_manager.job_meta_dict[jid]
        # 高优先级/紧急订单差异化扰动：60%概率跳过，只40%概率被扰动。目的：保证紧急订单的排程稳定性，尽量不改动它们的资源分配
        if job_meta.priority in ["high", "urgent"]:
            if np.random.random() > 0.4:
                continue
        # 获取该工序所属的资源组，筛选当前可用的机床和工人
        rg = state_manager.get_resource_group_by_op(op_id)
        available_machines = state_manager.get_available_machines(rg.machine_id_list)
        available_workers = state_manager.get_available_workers(rg.worker_id_list)

        # 随机选择新的机床和工人，替换原来的资源分配
        new_machine = np.random.choice(available_machines)
        new_worker = np.random.choice(available_workers)
        assign[idx] = (new_machine, new_worker)
    chrom["resource_assign"] = assign
    return chrom


def generate_heuristic_chromosome(reorder_job_seq: List[int], state_manager: ProductionStateManager) -> dict:
    """
    【函数整体功能】
    生成启发式优质初始染色体，核心规则：订单优先级最高、交期最早的订单优先排产；
    先对订单做加权+交期双层排序，再调用基础染色体生成函数，禁止随机打乱工序顺序，产出贴合交付诉求的基准可行排程方案；
    作为混合种群里的高质量种子个体，拉高种群整体初始适应度，加速NSGA收敛。
    """
    job_sort_info = []
    # 遍历待排订单，组装排序三元组
    for j_id in reorder_job_seq:
        meta = state_manager.job_meta_dict[j_id]
        # 负权重实现优先级降序，交期为次要升序条件
        job_sort_info.append((-meta.base_weight, meta.due_delivery_time, j_id))
    # 升序排序：先对比第一元素(-权重)，再对比第二元素(订单交期)
    job_sort_info.sort()
    # 提取排好顺序的订单ID
    sorted_job_ids = [item[2] for item in job_sort_info]
    # shuffle_free=False：不随机打乱工序/订单顺序，严格遵循启发排序结果
    return init_single_chromosome(sorted_job_ids, state_manager, shuffle_free=False)


def trim_historical_chromosome(old_chrom: dict, state_manager: ProductionStateManager) -> dict:
    """
     裁剪历史染色体：滚动排程前清理上一轮旧排程，生成符合当前状态的新初始染色体
     逻辑：保留已冻结区域 → 保留已完工/运行工序 → 将手动锁定的工序放到可优化工序最前面 → 继承历史可优化工序 → 补齐新增插单工序
     """
    new_chrom = copy.deepcopy(old_chrom)
    #  获取当前所有可优化工序的ID集合（状态 = OP_STATUS_OPTIMIZABLE）,  过滤掉了已完工、运行中、已取消的工序
    opt_ops = set(state_manager.get_optimizable_operation_ids())
    new_op_sequence = []
    new_resource_assign = []
    old_seq = old_chrom["op_sequence"]
    old_assign = old_chrom["resource_assign"]

    # 提前构建工序到位置的映射，O(1)查找，避免重复遍历
    op_to_pos = {op: idx for idx, op in enumerate(old_seq)}
    processed_ops = set()  # 记录已经处理过的工序，避免重复添加

    # ===================== 计算冻结边界 =====================
    # 冻结边界 = 当前系统时间 + 计划冻结窗口时长
    freeze_boundary = state_manager.current_system_time + cfg.PLAN_FROZEN_HORIZON
    # 从上一轮排程结果中获取所有工序的历史开工时间
    last_schedule = state_manager.last_schedule_result

    # ===================== 第一步：保留不可变工序（已完工>运行中>冻结区未开工） =====================
    for op in old_seq:
        status = state_manager.operation_status_dict.get(op, -1)
        if status in [cfg.OP_STATUS_FINISHED, cfg.OP_STATUS_RUNNING]:  # 已完工/正在运行，绝对不可变
            new_op_sequence.append(op)
            new_resource_assign.append(old_assign[op_to_pos[op]])
            processed_ops.add(op)
            continue
        if op in last_schedule:
            op_start_time = last_schedule[op]["start_time"]
            if op_start_time < freeze_boundary:
                new_op_sequence.append(op)
                new_resource_assign.append(old_assign[op_to_pos[op]])
                processed_ops.add(op)

    # ===================== 第二步：处理可优化工序 =====================
    # 先处理手动锁定的工序（优先级最高，放到可优化工序最前面）
    locked_ops = [op for op in opt_ops if state_manager.is_op_manual_locked(op)]
    for op in locked_ops:
        if op in processed_ops:
            continue
        lock_cfg = state_manager.get_lock_info(op)
        machine = lock_cfg.fixed_machine_id if lock_cfg.lock_machine else -1
        worker = lock_cfg.fixed_worker_id if lock_cfg.lock_worker else -1
        new_op_sequence.append(op)
        new_resource_assign.append((machine, worker))
        processed_ops.add(op)

    # 再处理历史中存在的可优化工序（继承历史顺序和资源）
    for op in old_seq:
        if op in processed_ops or op not in opt_ops:
            continue
        new_op_sequence.append(op)
        new_resource_assign.append(old_assign[op_to_pos[op]])
        processed_ops.add(op)

    # 最后处理本轮新增的插单工序（历史中不存在）
    for op in opt_ops:
        if op in processed_ops:
            continue
        rg = state_manager.get_resource_group_by_op(op)
        available_machines = state_manager.get_available_machines(rg.machine_id_list)
        available_workers = state_manager.get_available_workers(rg.worker_id_list)
        new_op_sequence.append(op)
        new_resource_assign.append((np.random.choice(available_machines), np.random.choice(available_workers)))
        processed_ops.add(op)

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


def fast_non_dominated_sorting(pop_fits: List[List[float]]) -> Tuple[List[List[int]], List[int]]:
    """通用快速非支配排序，快速非支配排序流程：
        1. 初始统计得到原始 dom_count：记录一共多少个体支配自己；
        2. 先把 dom_count=0 的划为 rank=0；
        3. 遍历 rank0 所有个体，把它们支配的所有个体 dom_count -= 1；
        4. 新 dom_count=0 的这批划为 rank=1；
        5. 再遍历 rank1，继续扣减下层 dom_count，产生 rank=2……"""

    n = len(pop_fits)
    # 关键数据结构1：dominated[p] = 所有被p支配的个体的索引列表,简称：p支配的个体列表。
    dominated: List[List[int]] = [[] for _ in range(n)]
    # 关键数据结构2：dom_count[p] = 支配p的个体的数量（即p被多少个个体支配）
    dom_count: List[int] = [0]*n

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _is_dominates(pop_fits[p], pop_fits[q]):  # 情况1：p支配q → 把q加入p的支配列表
                dominated[p].append(q)
            elif _is_dominates(pop_fits[q], pop_fits[p]):    # 情况2：q支配p → p的被支配计数器+1
                dom_count[p] += 1

    fronts, rank = _extract_fronts(dominated, dom_count)
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

    seq1, assign1 = p1["op_sequence"], p1["resource_assign"]
    seq2, assign2 = p2["op_sequence"], p2["resource_assign"]

    frozen_boundary = state_manager.current_system_time + cfg.PLAN_FROZEN_HORIZON
    frozen_ops = _get_frozen_operations(seq1, state_manager, frozen_boundary)

    # 若全部操作被冻结，则直接返回父代副本
    if len(frozen_ops) == len(seq1):
        return copy.deepcopy(p1), copy.deepcopy(p2)

    # 提取自由操作（冻结操作位于序列前端）
    free_ops1 = _extract_free_operations(seq1, frozen_ops)
    free_ops2 = _extract_free_operations(seq2, frozen_ops)

    # POX 交叉产生子代【自由工序序列】（正确：工序ID列表）
    child1_free, child2_free = _pox_crossover_free_sequences(
        free_ops1, free_ops2, state_manager
    )
    # 【同订单连续工艺同组复用机床】生成对应自由段资源分配
    c1_res = rebuild_resource_assign(child1_free, state_manager, frozen_boundary)
    c2_res = rebuild_resource_assign(child2_free, state_manager, frozen_boundary)

    # ========== 工序序列 = 冻结工序 + POX生成的新工序序列 ==========
    child1_seq = frozen_ops + child1_free
    child2_seq = frozen_ops + child2_free

    # 手动拼接资源，彻底删除冲突的 _build_child_assignments 调用
    # 1. 先填充冻结段资源
    child1_assign = []
    child2_assign = []
    for fid in frozen_ops:
        s = state_manager.last_schedule_result[fid]
        child1_assign.append((s["machine_id"], s["worker_id"]))
        child2_assign.append((s["machine_id"], s["worker_id"]))
    # 2. 拼接重建好的带复用约束的自由段资源
    child1_assign.extend(c1_res)
    child2_assign.extend(c2_res)

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
    op_seq = new_chrom["op_sequence"]
    resource_assign = new_chrom["resource_assign"]
    frozen_boundary = state_manager.current_system_time + cfg.PLAN_FROZEN_HORIZON
    for idx, raw_op_id in enumerate(op_seq):
        op_id = int(raw_op_id)
        if is_operation_frozen(op_id, frozen_boundary, state_manager):
            continue
        if np.random.random() > cfg.MUTATION_RATE:
            continue

        resource_group = state_manager.get_resource_group_by_op(op_id)
        available_machines = state_manager.get_available_machines(resource_group.machine_id_list)
        available_workers = state_manager.get_available_workers(resource_group.worker_id_list)

        reuse_mid = try_get_prev_same_group_machine(op_id, op_seq, resource_assign, state_manager)
        if reuse_mid is not None:
            selected_machine_id = reuse_mid
        else:
            selected_machine_id = np.random.choice(available_machines)

        selected_worker_id = np.random.choice(available_workers)

        resource_assign[idx] = (selected_machine_id, selected_worker_id)
    new_chrom["resource_assign"] = resource_assign
    return new_chrom

def select_optimal_solution_by_weight(pareto_set: List[dict], all_pop_fits: List[List[float]], pareto_index_list: List[int], weight: List[float]) -> \
tuple[dict, list[float]]:
    """8目标加权择优，权重顺序：
    [fit1, fit2, fit3, fit4, fit5, fit6, fit7]
    从帕累托最优解集中选出综合得分最优的排程方案，所有目标均为最小化目标，综合得分越低，方案越优
    """
    score_list = []
    for idx in pareto_index_list:
        fit_vec = all_pop_fits[idx]
        # 计算加权综合得分：每个目标得分 × 对应权重，然后求和
        total_score = sum(weight[i] * fit_vec[i] for i in range(len(weight)))
        score_list.append(total_score)
    # 因为所有目标都是越小越好，所以得分越低方案越优，np.argmin() 返回的是「最小值在数组中所在的索引位置」，不是最小值本身
    best_in_pareto_idx = np.argmin(score_list)
    # 转换为该解在整个原始种群中的索引
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
            trackers.machine_last_end_time_dict[typed_machine_id] = 0.0
            trackers.machine_previous_technology_type_dict[typed_machine_id] = -1
            # 单台机器总可用工时,初始化时均为0
            trackers.machine_total_available_hour[typed_machine_id] = 0.0
            trackers.machine_total_process_hour[typed_machine_id] = 0.0

        # 初始化工人
        for worker_id in resource_group.worker_id_list:
            typed_worker_id = WorkerId(worker_id)
            trackers.worker_task_intervals_dict[typed_worker_id] = []
            trackers.worker_task_ends_heap_dict[typed_worker_id] = []

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
        if not state_manager.is_machine_available(original_machine_id):
            resource_group = state_manager.get_resource_group_by_op(operation_id)
            available_machines = state_manager.get_available_machines(resource_group.machine_id_list)
            if not available_machines:  # 优化，这里不能终止程序，需将其排程到虚拟设备上
                raise ValueError(f"操作[{operation_id}] 锁定工人后无可用机器")

            machine_id = select_best_machine(available_machines, trackers)
            logger.debug(f"操作[{operation_id}] 原机器不可用，自动选择: {machine_id}")

    # 只锁机器不锁工人：优先保留原可用工人，不可用时选最优工人
    if lock_config.lock_machine and not lock_config.lock_worker:
        if not state_manager.is_worker_available(original_worker_id):
            resource_group = state_manager.get_resource_group_by_op(operation_id)
            available_workers = state_manager.get_available_workers(resource_group.worker_id_list)
            if not available_workers:   # 优化，这里不能终止程序，需将其排程到虚拟人员上
                raise ValueError(f"操作[{operation_id}] 锁定机器后无可用工人")

            # ✅ 已修复：使用统一的最优工人选择逻辑
            worker_id = select_best_worker(available_workers, trackers)
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
    if machine_id == MachineId(-1) or not state_manager.is_machine_available(machine_id):
        available_machines = state_manager.get_available_machines(resource_group.machine_id_list)
        if not available_machines:
            raise ValueError(f"操作[{operation_id}] 无可用机器")

        machine_id = select_best_machine(available_machines, trackers)
        logger.debug(f"操作[{operation_id}] 机器无效，自动分配: {machine_id}")

    # 处理无效工人ID
    if worker_id == WorkerId(-1) or not state_manager.is_worker_available(worker_id):
        available_workers = state_manager.get_available_workers(resource_group.worker_id_list)
        if not available_workers:
            raise ValueError(f"操作[{operation_id}] 无可用工人")

        # 使用统一的最优工人选择逻辑
        worker_id = select_best_worker(available_workers, trackers)
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
        worker_meta = state_manager.worker_meta_dict.get(worker_id)
        if worker_meta and technology_type in worker_meta.tech_speed_ratio:
            speed_ratio = worker_meta.tech_speed_ratio[technology_type]
    actual_processing_time = round(base_processing_time * speed_ratio,1)

    # --------------------------
    # 2. 计算理想最早开始时间（基础约束）
    # --------------------------
    machine_available_time = trackers.machine_last_end_time_dict[machine_id]
    job_available_time = trackers.job_last_operation_end_time_dict.get(job_id, 0.0)
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
            while (trackers.worker_task_ends_heap_dict[worker_id]
                   and trackers.worker_task_ends_heap_dict[worker_id][0] <= ideal_start_time):
                heapq.heappop(trackers.worker_task_ends_heap_dict[worker_id])

            # 检查是否满足并行限制
            if len(trackers.worker_task_ends_heap_dict[worker_id]) < max_parallel:
                break

            # 等待最早的任务结束
            earliest_end = trackers.worker_task_ends_heap_dict[worker_id][0]
            ideal_start_time = earliest_end

    # --------------------------
    # 4. 处理机器换型时间（提前换型优化）
    # --------------------------
    previous_tech = trackers.machine_previous_technology_type_dict[machine_id]
    if previous_tech != -1 and previous_tech != technology_type:
        changeover_map = state_manager.machine_meta_dict[machine_id].changeover_time_map
        changeover_time = changeover_map.get(previous_tech, {}).get(technology_type, 0.0)
        trackers.total_changeover_time += changeover_time

        # 换型从机器可用时间开始
        changeover_end = state_manager.calculate_actual_work_end_time(machine_available_time, changeover_time)
        ideal_start_time = max(ideal_start_time, changeover_end)

    # 更新机器工艺类型（无论是否换型都要更新）
    trackers.machine_previous_technology_type_dict[machine_id] = technology_type

    # --------------------------
    # 5. 应用班次日历约束，获取合法开始/结束时间
    # --------------------------
    actual_start_time = state_manager.get_valid_start_time(ideal_start_time)
    actual_end_time = state_manager.calculate_actual_work_end_time(actual_start_time, actual_processing_time)

    # --------------------------
    # 6. 判断是否处于计划冻结区间
    # --------------------------
    frozen_time_limit = state_manager.current_system_time + cfg.PLAN_FROZEN_HORIZON
    is_frozen = actual_start_time < frozen_time_limit

    # --------------------------
    # 7. 构造调度结果对象
    # --------------------------
    return OperationSchedulingResult(
        operation_id=OperationId(operation_id),
        job_id=job_id,
        operation_index_in_job=operation_metadata.op_index_in_job,
        machine_id=machine_id,
        worker_id=worker_id,
        start_time=actual_start_time,
        end_time=actual_end_time,
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
    if job_id in trackers.job_last_operation_end_time_dict:
        previous_end = trackers.job_last_operation_end_time_dict[job_id]
        if previous_end < start_time:
            trackers.total_wip_wait_time += (start_time - previous_end)

    # 2. 更新机器状态 + 累加实际加工工时
    trackers.machine_last_end_time_dict[machine_id] = end_time
    trackers.machine_workloads_dict[machine_id].append(actual_processing_time)
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
        trackers.worker_task_intervals_dict[worker_id].append((start_time, end_time))
        heapq.heappush(trackers.worker_task_ends_heap_dict[worker_id], end_time)
        trackers.worker_workloads_dict[worker_id].append(actual_processing_time)

    # 4. 更新工件状态
    trackers.job_last_operation_end_time_dict[job_id] = end_time

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
    fit[2] = fit3：总工期
    fit[3] = fit4：设备整体闲置率
    fit[5] = fit6：机器负载不平衡度（工时方差）
    fit[6] = fit7：工人负载不平衡度（工时方差）
    fit[7] = fit8：在制品加权等待总成本
    """

    # ===================== fit0: 逾期订单总数、fit1: 订单逾期总惩罚 =====================
    overdue_count = 0
    total_overdue_penalty = 0.0

    for job_id, finish_time in trackers.job_last_operation_end_time_dict.items():
        job_meta = state_manager.job_meta_dict.get(job_id)
        if not job_meta:
            continue
        # print(f"=============jobid={job_id}, finishtime={finish_time}, job_delivery_time={job_meta.due_delivery_time}, delivery_date={job_meta.due_delivery_date}")
        if finish_time > job_meta.due_delivery_time:
            penalty = state_manager.calc_delivery_overdue_penalty(finish_time, job_meta)
            total_overdue_penalty += penalty
            overdue_count += 1

    # ===================== fit2: 总工期 =====================
    makespan = max(trackers.job_last_operation_end_time_dict.values()) if trackers.job_last_operation_end_time_dict else 0.0
    # machine_overload_penalty = 0.0
    # for machine_id, workloads in trackers.machine_workloads_dict.items():
    #     total_load = sum(workloads)
    #     machine_overload_penalty += state_manager.get_machine_overload_penalty(machine_id, total_load)
    # makespan_and_penalty = makespan + machine_overload_penalty

    # ===================== fit3: 设备整体闲置率 = 总空闲时长 / 总可用工时 =====================
    total_all_available_time = 0.0
    total_all_idle_time = 0.0
    for mid in trackers.machine_last_end_time_dict.keys():
        proc_h = trackers.machine_total_process_hour[mid]
        machine_available_h = state_manager.get_schedule_total_work_hours_horizon(makespan, state_manager.machine_meta_dict[mid].planned_daily_hour)
        idle_h = max(0.0, machine_available_h - proc_h)
        total_all_available_time += machine_available_h
        total_all_idle_time += idle_h
    if total_all_available_time > 1e-9:
        overall_equipment_idle_rate = round(total_all_idle_time / total_all_available_time, 3)
    else:
        overall_equipment_idle_rate = 0.0

    # ===================== fit4: 设备总换型时间【当前不实现，论证清楚后可实现】 =====================
    machine_total_changeover_time = trackers.total_changeover_time

    # ===================== fit5: 机器负载不平衡度（加工工时方差） =====================
    machine_total_loads = [sum(loads) for loads in trackers.machine_workloads_dict.values()]
    machine_unbalance = round(np.var(machine_total_loads), 2) if len(machine_total_loads) > 1 else 0.0

    # ===================== fit6: 工人负载不平衡度（仅工时方差，去掉切换成本） =====================
    worker_total_loads = [sum(loads) for loads in trackers.worker_workloads_dict.values()]
    worker_unbalance = round(np.var(worker_total_loads), 2) if len(worker_total_loads) > 1 else 0.0

    # ===================== fit7: 在制品加权等待总成本 =====================
    wip_cost = trackers.total_wip_wait_time * cfg.WIP_WEIGHT_COEFFICIENT

    # 组装最终适应度向量
    return [
        overdue_count,
        total_overdue_penalty,
        makespan,
        overall_equipment_idle_rate,
        machine_unbalance,
        worker_unbalance,
        wip_cost
    ]

def _get_frozen_operations(seq: List[int], state_manager, frozen_boundary: float) -> List[int]:
    """
    返回序列中所有开始时间早于冻结边界的操作（按原序列顺序）。
    """
    frozen = []
    last_schedule = state_manager.last_schedule_result
    for op_id in seq:
        if op_id in last_schedule and last_schedule[op_id]["start_time"] < frozen_boundary:
            frozen.append(op_id)
    return frozen

def _extract_free_operations(seq: List[int], frozen_ops: List[int]) -> List[int]:
    """
    假设冻结操作均位于序列开头，提取剩余的自由操作（保持原有相对顺序）。
    若冻结操作并非全部位于前缀，此方法将失效——但算法设计保证冻结操作在前。
    """
    return seq[len(frozen_ops):]

def _pox_crossover_free_sequences(
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

    # 核心是，我属于第一组的放前头，对方爸属于第二组的放后头。保持相对顺序不变
    # 子代1：继承父代1中属于组1的操作 + 父代2中属于组2的操作
    for op in free_ops1:
        if op_to_job[op] in group1:
            child1_free.append(int(op))
    for op in free_ops2:
        if op_to_job[op] in group2:
            child1_free.append(int(op))

    # 子代2：继承父代2中属于组1的操作 + 父代1中属于组2的操作
    for op in free_ops2:
        if op_to_job[op] in group1:
            child2_free.append(int(op))
    for op in free_ops1:
        if op_to_job[op] in group2:
            child2_free.append(int(op))

    return child1_free, child2_free

def _random_select_resources(op_id: int, state_manager) -> Tuple[int, int]:
    """
    为给定操作随机选取一台可用机器和一名可用工人。
    """
    rg = state_manager.get_resource_group_by_op(op_id)
    # 筛选组内当前状态可用机床
    available_machines = state_manager.get_available_machines(rg.machine_id_list)
    # 筛选组内具备该工艺技能、在岗可用工人
    available_workers = state_manager.get_available_workers(rg.worker_id_list)
    machine = np.random.choice(available_machines)
    worker = np.random.choice(available_workers)
    return machine, worker


def _is_dominates(candidate: List[float], other: List[float]) -> bool:
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

def _extract_fronts(dominated: List[List[int]], dom_count: List[int]) -> Tuple[List[List[int]], List[int]]:
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


def try_get_prev_same_group_machine(
    current_op_id: int,
    op_sequence: list[int],
    resource_assign: list[tuple[MachineId, WorkerId]],
    state_manager
) -> MachineId | None:
    """
    复用判定规则（自适应任意工艺编号步长）
    1. 筛选同一个订单全部工序
    2. 把该订单所有工序按 business_op_no 从小到大整体排序
    3. 排序后紧挨着自己的上一道工序 = 工艺直接前驱
    4. 两道工序资源组ID必须完全一致，才允许复用机床
    5. 不管两道工序在染色体排程序列隔多远，只看工艺逻辑先后
    """
    op_meta = state_manager.op_meta_dict[current_op_id]
    job_id = op_meta.belong_job_id
    current_rg_id = op_meta.resource_group_id
    current_biz_no = int(op_meta.business_op_no)

    # 取出该订单全部工序
    all_job_ops = [
        oid for oid, meta in state_manager.op_meta_dict.items()
        if meta.belong_job_id == job_id
    ]

    # 构建 {工艺编号: 工序ID}
    op_no_map = {}
    for oid in all_job_ops:
        no_val = int(state_manager.op_meta_dict[oid].business_op_no)
        op_no_map[no_val] = oid

    # 把该订单所有工艺编号升序排列
    sorted_no_list = sorted(op_no_map.keys())
    target_prev_no = None

    # 遍历找当前编号在排序列表里的位置，取前一个
    for idx, no in enumerate(sorted_no_list):
        if no == current_biz_no:
            if idx > 0:
                target_prev_no = sorted_no_list[idx - 1]
            break

    # 本订单没有工艺前驱
    if target_prev_no is None:
        return None
    prev_op_id = op_no_map[target_prev_no]

    # 校验前驱和当前工序资源组一致
    prev_op_meta = state_manager.op_meta_dict[prev_op_id]
    if prev_op_meta.resource_group_id != current_rg_id:
        return None

    # 查找前驱在当前染色体里的下标，读取已分配机床
    try:
        prev_idx = op_sequence.index(prev_op_id)
        prev_machine, _ = resource_assign[prev_idx]
    except ValueError:
        # 前驱工序不在当前子代浮动序列中（冻结/剔除）
        return None
    except TypeError:
        # 前驱还未完成资源赋值
        return None

    # 校验机床属于当前资源组、设备可用
    rg = state_manager.get_resource_group_by_op(current_op_id)
    if prev_machine not in rg.machine_id_list:
        logger.warning(f"前驱{prev_op_id}的机床{prev_machine}不在当前资源组{rg.machine_id_list}中，无法复用")
        return None
    if not state_manager.is_machine_available(prev_machine):
        logger.warning(f"前驱{prev_op_id}的机床{prev_machine}当前不可用，无法复用")
        return None

    return prev_machine


def rebuild_resource_assign(
    new_op_seq: list[int],
    state_manager: ProductionStateManager,
    frozen_boundary: float
) -> list[tuple]:
    """
    按订单工艺顺序分配资源，保证同订单工艺相邻、同资源组工序复用同一机床
    """
    total_len = len(new_op_seq)
    resource_assign = [None] * total_len
    op_index_map = {op: idx for idx, op in enumerate(new_op_seq)}

    def is_op_frozen(op_id: int) -> bool:
        res = state_manager.last_schedule_result.get(op_id)
        if res is None or res.get("start_time") is None:
            return False
        return res["start_time"] < frozen_boundary

    # 1、按订单分组
    job_op_dict: Dict[int, List[int]] = {}
    for op_id in new_op_seq:
        jid = state_manager.op_meta_dict[op_id].belong_job_id
        if jid not in job_op_dict:
            job_op_dict[jid] = []
        job_op_dict[jid].append(op_id)

    # 2、逐个订单，按工艺号从小到大分配（前驱必然先赋值）
    for job_id, op_list in job_op_dict.items():
        # 本订单工序按业务工艺编号升序排序
        op_list_sorted = sorted(op_list, key=lambda x: int(state_manager.op_meta_dict[x].business_op_no))
        for op_id in op_list_sorted:
            pos = op_index_map[op_id]
            if is_op_frozen(op_id):
                mid = state_manager.last_schedule_result[op_id]["machine_id"]
                wid = state_manager.last_schedule_result[op_id]["worker_id"]
                resource_assign[pos] = (mid, wid)
                continue

            # 查询工艺前驱机床
            reuse_mid = try_get_prev_same_group_machine(op_id, new_op_seq, resource_assign, state_manager)
            rg = state_manager.get_resource_group_by_op(op_id)
            avail_machines = state_manager.get_available_machines(rg.machine_id_list)
            avail_workers = state_manager.get_available_workers(rg.worker_id_list)

            if reuse_mid is not None:
                select_mid = reuse_mid
            else:
                select_mid = np.random.choice(avail_machines)
            select_wid = np.random.choice(avail_workers)
            resource_assign[pos] = (select_mid, select_wid)

    return resource_assign