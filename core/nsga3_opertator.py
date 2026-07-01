"""
模块：nsga3_operator.py
功能：NSGA-III 高维多目标排程算法主算子
适配场景：7目标及以上高维排程优化
核心改进：用参考点关联机制替代拥挤距离，解决NSGA-II高维下选择压力坍塌、多样性失效问题
复用能力：100%复用base_ga中的编码、解码、交叉、变异、约束处理、收敛监控、TOPSIS择优
"""

import numpy as np
import copy
from typing import List, Dict, Tuple, Any
from itertools import combinations_with_replacement

from core.state_manager import ProductionStateManager
from config.settings import (
    POPULATION_SIZE, MAX_GENERATION, MAX_FRONT_NUM, CROSSOVER_RATE
)
from utils.log_utils import get_logger

# 复用base_ga全部公共基础能力
from core.base_ga import (
    init_mixed_population, decode_chromosome, fast_non_dominated_sorting,
    pox_crossover, mutate_chromosome, evaluate_population_fitness,
    ConvergenceMonitor, select_optimal_solution_by_weight
)
from core.multi_criteria_decision import TopsisAllMinEvaluator

logger = get_logger(__name__)

# ======================================================================
# ===================== 模块级常量定义 =============
# ======================================================================

# 数值计算精度阈值
_NUMERICAL_EPSILON: float = 1e-9
# 锦标赛选择样本数量
_TOURNAMENT_SAMPLE_SIZE: int = 4
# 默认参考点划分次数（7目标下3次划分对应84个参考点，性价比最优）
_DEFAULT_REFERENCE_DIVISIONS: int = 3
# 成就标量化函数权重系数（用于极值点求解）
_ASF_LARGE_WEIGHT: float = 1e6
# 归一化异常值截断比例（剔除最差的5%个体，避免截距被劣解带偏）
_NORMALIZE_OUTLIER_RATIO: float = 0.05

# ======================================================================
# ===================== NSGA-III 核心工具函数 ===========================
# ======================================================================

def generate_das_dennis_points(objective_count: int, divisions: int) -> np.ndarray:
    """
    Das-Dennis 均匀参考点生成
    在M维单位单纯形上生成均匀分布的参考方向，是NSGA-III维持多样性的核心
    :param objective_count: 优化目标维度数量
    :param divisions: 每个维度划分次数，数值越大参考点越多，多样性越好但计算量上升
    :return: 参考点矩阵 shape=(参考点数量, objective_count)，每行元素和为1
    """
    if divisions <= 0:
        uniform_point = np.ones((1, objective_count), dtype=np.float64) / objective_count
        return uniform_point

    dimension_indices = list(range(objective_count))
    point_collection = []

    for dimension_combo in combinations_with_replacement(dimension_indices, divisions):
        dimension_counter = np.zeros(objective_count, dtype=np.float64)
        for dim_idx in dimension_combo:
            dimension_counter[dim_idx] += 1.0
        normalized_point = dimension_counter / divisions
        point_collection.append(normalized_point)

    return np.array(point_collection, dtype=np.float64)


def generate_boundary_reference_points(objective_count: int) -> np.ndarray:
    """
    生成极端边界参考点（每个维度单独取1，其余为0）
    用于保护极限最优解，如纯工期最短、纯逾期最小等生产高价值方案
    :param objective_count: 优化目标维度数量
    :return: 边界参考点矩阵 shape=(objective_count, objective_count)
    """
    return np.eye(objective_count, dtype=np.float64)


def normalize_objectives(fitness_matrix: np.ndarray, trim_outlier: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NSGA-III 自适应目标归一化（优化鲁棒性）
    消除多目标量纲差异，映射到单位超平面，适配参考点关联计算
    流程：异常值截断 → 计算理想点 → 目标平移 → ASF法求极值点拟合超平面 → 截距归一化
    :param fitness_matrix: 原始适应度矩阵 shape=(个体数, 目标数)，全最小化目标
    :param trim_outlier: 是否截断极端劣解，提高归一化稳定性
    :return: (归一化后矩阵, 理想点向量, 截距向量)
    """
    objective_count = fitness_matrix.shape[1]
    epsilon = _NUMERICAL_EPSILON

    # 异常值截断：按每个目标剔除最差的5%个体，避免截距被极端劣解带偏
    calc_matrix = fitness_matrix.copy()
    if trim_outlier and fitness_matrix.shape[0] > 20:
        trim_count = int(fitness_matrix.shape[0] * _NORMALIZE_OUTLIER_RATIO)
        for dim in range(objective_count):
            sorted_vals = np.sort(calc_matrix[:, dim])
            threshold = sorted_vals[-trim_count]
            calc_matrix[:, dim] = np.minimum(calc_matrix[:, dim], threshold)

    # 1. 计算理想点（每个目标的最小值）
    ideal_point = np.min(calc_matrix, axis=0)

    # 2. 目标值平移，使理想点落在原点
    translated_fitness = calc_matrix - ideal_point

    # 3. 成就标量化函数(ASF)求解每个目标轴的极值点
    intercepts = np.full(objective_count, epsilon, dtype=np.float64)
    for target_dim in range(objective_count):
        weight_vector = np.full(objective_count, _ASF_LARGE_WEIGHT, dtype=np.float64)
        weight_vector[target_dim] = 1.0
        asf_values = np.max(translated_fitness / weight_vector, axis=1)
        extreme_point_index = np.argmin(asf_values)
        extreme_value = translated_fitness[extreme_point_index, target_dim]
        if abs(extreme_value) > epsilon:
            intercepts[target_dim] = extreme_value

    # 4. 截距归一化到单位超平面（对原始全量数据做归一化，仅用计算得到的理想点和截距）
    full_translated = fitness_matrix - ideal_point
    normalized_fitness = full_translated / intercepts
    return normalized_fitness, ideal_point, intercepts


def associate_to_references(
    normalized_fitness: np.ndarray,
    reference_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    种群个体-参考点关联
    计算每个个体到各参考方向的垂直距离，取最近的作为关联参考点
    :param normalized_fitness: 归一化后的适应度矩阵 shape=(个体数, 目标数)
    :param reference_points: 参考点矩阵 shape=(参考点数, 目标数)
    :return: (个体关联的参考点索引数组, 个体到对应参考线的垂直距离数组)
    """
    population_size = normalized_fitness.shape[0]
    reference_count = reference_points.shape[0]
    associate_index_array = np.zeros(population_size, dtype=int)
    perpendicular_distance_array = np.zeros(population_size, dtype=np.float64)

    for individual_idx in range(population_size):
        objective_vector = normalized_fitness[individual_idx]
        min_distance = np.inf
        best_reference_index = 0

        for ref_idx in range(reference_count):
            ref_vector = reference_points[ref_idx]
            ref_norm_squared = np.dot(ref_vector, ref_vector)
            if ref_norm_squared < _NUMERICAL_EPSILON:
                continue

            # 向量投影计算垂直距离
            projection_length = np.dot(objective_vector, ref_vector) / ref_norm_squared
            # 投影长度非负校验，避免极端异常值导致方向反转
            projection_length = max(projection_length, 0.0)
            projection_vector = projection_length * ref_vector
            current_distance = np.linalg.norm(objective_vector - projection_vector)

            if current_distance < min_distance:
                min_distance = current_distance
                best_reference_index = ref_idx

        associate_index_array[individual_idx] = best_reference_index
        perpendicular_distance_array[individual_idx] = min_distance

    return associate_index_array, perpendicular_distance_array


def nsga3_environmental_selection(
    combined_fitness: List[List[float]],
    target_population_size: int,
    reference_points: np.ndarray
) -> List[int]:
    """
    NSGA-III 环境选择核心逻辑（修复：全量种群归一化）
    流程：非支配排序粗分层 → 逐层加入下一代 → 最后一层按参考点小生境计数筛选
    :param combined_fitness: 父子合并后的适应度列表（2N个个体）
    :param target_population_size: 目标种群大小（N）
    :param reference_points: 参考点矩阵
    :return: 选中的个体下标列表，长度=target_population_size
    """
    fitness_matrix = np.array(combined_fitness, dtype=np.float64)
    total_individual_count = fitness_matrix.shape[0]

    # 1. 快速非支配排序，得到各层前沿
    all_fronts, _ = fast_non_dominated_sorting(combined_fitness)

    # 2. 逐层加入下一代种群，直到加入当前层后超过目标大小
    selected_individuals = []
    last_front_level = -1

    for front_level, front_individuals in enumerate(all_fronts):
        if len(selected_individuals) + len(front_individuals) <= target_population_size:
            selected_individuals.extend(front_individuals)
        else:
            last_front_level = front_level
            break
    else:
        logger.warning("所有非支配解总数小于种群规模，已全部保留")
        return selected_individuals

    last_front_individuals = all_fronts[last_front_level]
    need_select_count = target_population_size - len(selected_individuals)

    # ========== 修复点：基于全量合并种群做统一归一化 ==========
    norm_all, _, _ = normalize_objectives(fitness_matrix, trim_outlier=True)
    # 提取已选个体与最后一层个体的归一化结果
    norm_selected = norm_all[selected_individuals] if selected_individuals else None
    norm_last_front = norm_all[last_front_individuals]

    # 3. 关联最后一层前沿的参考点
    last_front_associate, last_front_distances = associate_to_references(norm_last_front, reference_points)

    # 4. 统计已选个体在各参考点的小生境计数
    niche_count_array = np.zeros(len(reference_points), dtype=int)
    if norm_selected is not None and len(norm_selected) > 0:
        selected_associate, _ = associate_to_references(norm_selected, reference_points)
        for ref_id in selected_associate:
            niche_count_array[ref_id] += 1

    # 5. 构建参考点候选池：每个参考点下的个体按距离从小到大排序
    reference_candidate_pool: Dict[int, List[Tuple[int, float]]] = {}
    for front_inner_idx, ref_id in enumerate(last_front_associate):
        original_pop_idx = last_front_individuals[front_inner_idx]
        distance = last_front_distances[front_inner_idx]
        if ref_id not in reference_candidate_pool:
            reference_candidate_pool[ref_id] = []
        reference_candidate_pool[ref_id].append((original_pop_idx, distance))

    for ref_id in reference_candidate_pool:
        reference_candidate_pool[ref_id].sort(key=lambda item: item[1])

    # 6. 迭代选择：优先给小生境计数最小的参考方向补充最优个体
    selected_from_last_front = []
    while len(selected_from_last_front) < need_select_count and reference_candidate_pool:
        # 找当前小生境计数最小的参考点
        min_niche_count = np.inf
        target_reference_id = -1
        for ref_id in reference_candidate_pool:
            if niche_count_array[ref_id] < min_niche_count:
                min_niche_count = niche_count_array[ref_id]
                target_reference_id = ref_id

        if target_reference_id == -1 or not reference_candidate_pool[target_reference_id]:
            del reference_candidate_pool[target_reference_id]
            continue

        # 取出该参考点下距离最近的个体
        chosen_individual_idx, _ = reference_candidate_pool[target_reference_id].pop(0)
        selected_from_last_front.append(chosen_individual_idx)
        niche_count_array[target_reference_id] += 1

        if not reference_candidate_pool[target_reference_id]:
            del reference_candidate_pool[target_reference_id]

    # 极端情况兜底：参考点筛选不足时随机补充
    if len(selected_from_last_front) < need_select_count:
        remaining_candidates = set(last_front_individuals) - set(selected_from_last_front)
        fill_count = need_select_count - len(selected_from_last_front)
        selected_from_last_front.extend(list(remaining_candidates)[:fill_count])
        logger.warning(f"NSGA-III 最后一层参考点筛选不足，随机补充{fill_count}个个体")

    selected_individuals.extend(selected_from_last_front[:need_select_count])
    return selected_individuals


def tournament_selection_nsga3(
    population: List[dict],
    fitness_list: List[List[float]],
    state_manager: ProductionStateManager
) -> dict:
    """
    NSGA-III 锦标赛选择
    基于非支配排序优先选层级高的个体，同一层级随机选择（选择压力主要由环境选择提供）
    :param population: 种群列表
    :param fitness_list: 对应适应度列表
    :param state_manager: 生产状态管理器（兼容接口）
    :return: 选中的个体
    """
    sample_size = _TOURNAMENT_SAMPLE_SIZE
    total_individual_count = len(population)
    all_indices = list(range(total_individual_count))
    sample_indices = np.random.choice(all_indices, size=sample_size, replace=False)
    sample_fitness = [fitness_list[idx] for idx in sample_indices]

    sample_fronts, _ = fast_non_dominated_sorting(sample_fitness)
    first_front_indices = sample_fronts[0]

    chosen_sample_idx = np.random.choice(first_front_indices)
    return population[sample_indices[chosen_sample_idx]]

# ======================================================================
# ===================== 种群迭代与主入口 ================================
# ======================================================================

def create_next_generation_nsga3(
    parent_population: List[Dict[str, Any]],
    parent_fitness: List[List[float]],
    reference_points: np.ndarray,
    target_pop_size: int,
    state_manager: ProductionStateManager
) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    """
    生成NSGA-III下一代种群（标准精英策略：父子合并后环境选择）
    :param parent_population: 父代种群
    :param parent_fitness: 父代对应适应度列表
    :param reference_points: 参考点矩阵
    :param target_pop_size: 目标种群大小
    :param state_manager: 生产状态管理器
    :return: (下一代种群, 对应适应度列表)
    """
    # 1. 锦标赛选择 + 交叉变异生成等量子代
    offspring_population = []
    while len(offspring_population) < target_pop_size:
        parent_a = tournament_selection_nsga3(parent_population, parent_fitness, state_manager)
        parent_b = tournament_selection_nsga3(parent_population, parent_fitness, state_manager)

        if np.random.random() < CROSSOVER_RATE:
            child_a, child_b = pox_crossover(parent_a, parent_b, state_manager)
        else:
            child_a = copy.deepcopy(parent_a)
            child_b = copy.deepcopy(parent_b)

        child_a = mutate_chromosome(child_a, state_manager)
        child_b = mutate_chromosome(child_b, state_manager)

        offspring_population.append(child_a)
        if len(offspring_population) < target_pop_size:
            offspring_population.append(child_b)

    # 2. 评估子代适应度
    offspring_fitness = evaluate_population_fitness(offspring_population, state_manager)

    # 3. 父子种群与适应度合并
    combined_population = parent_population + offspring_population
    combined_fitness = parent_fitness + offspring_fitness

    # 4. NSGA-III环境选择选出下一代
    selected_indices = nsga3_environmental_selection(combined_fitness, target_pop_size, reference_points)

    next_generation_pop = [combined_population[idx] for idx in selected_indices]
    next_generation_fit = [combined_fitness[idx] for idx in selected_indices]

    return next_generation_pop, next_generation_fit

def _adjust_population_size(current_pop_size: int, reference_point_count: int) -> Tuple[int, bool]:
    """
    根据参考点数量自动校准种群规模
    规则：参考点数量 > 当前种群的80%时，自动扩容
    要求：新种群规模 > 参考点数量，且为4的整数倍（适配锦标赛4样本配比）
    :param current_pop_size: 当前配置的种群规模
    :param reference_point_count: 参考点总数量
    :return: (校准后的种群规模, 是否发生了调整)
    """
    # 触发扩容的阈值比例
    size_warn_threshold_ratio: float = 0.8
    # 种群规模对齐基数（锦标赛选择样本数为4，保证整数配比）
    size_alignment_base: int = 4

    # 无需调整的情况：参考点数量在阈值以内
    if reference_point_count <= current_pop_size * size_warn_threshold_ratio:
        return current_pop_size, False

    # 计算最小合法规模：必须严格大于参考点数量
    min_required_size = int(np.ceil(reference_point_count / size_warn_threshold_ratio))

    # 向上取最近的4的倍数
    remainder = min_required_size % size_alignment_base
    if remainder == 0:
        adjusted_size = min_required_size
    else:
        adjusted_size = min_required_size + (size_alignment_base - remainder)

    return adjusted_size, True

def nsga3_rolling_schedule(
    state_manager: ProductionStateManager,
    reorder_job_sequence: List[int],
    reference_divisions: int = _DEFAULT_REFERENCE_DIVISIONS,
    add_boundary_points: bool = True
) -> Tuple[List[dict], List[List[float]], List[int]]:
    """
    NSGA-III 滚动排程主入口
    输入输出与nsga2_rolling_schedule完全对齐，可无缝替换
    :param state_manager: 生产状态管理器
    :param reorder_job_sequence: 订单重排序列表
    :param reference_divisions: 参考点划分次数，7目标建议取3（对应84个均匀参考点）
    :param add_boundary_points: 是否补充极端边界参考点，保护极限方案
    :return: Tuple[帕累托种群, 对应适应度列表, 帕累托解在最终种群中的索引]
    """
    # -------------------------- 1. 配置读取与初始化 --------------------------
    population_size = POPULATION_SIZE
    max_generation_count = MAX_GENERATION
    max_pareto_keep_count = MAX_FRONT_NUM

    # 动态获取目标维度（从配置或初代种群推断，避免硬编码）
    # 先生成初代种群，获取第一个适应度向量的维度
    temp_population = init_mixed_population(reorder_job_sequence, state_manager)
    temp_fit, _ = decode_chromosome(temp_population[0], state_manager)
    objective_dimension = len(temp_fit)
    del temp_population, temp_fit

    # 生成参考点：均匀基础点 + 边界点
    uniform_ref_points = generate_das_dennis_points(objective_dimension, reference_divisions)
    if add_boundary_points:
        boundary_ref_points = generate_boundary_reference_points(objective_dimension)
        reference_points = np.concatenate([uniform_ref_points, boundary_ref_points], axis=0)
    else:
        reference_points = uniform_ref_points

    # 参数校验与自动校准
    adjusted_pop_size, is_adjusted = _adjust_population_size(population_size, len(reference_points))
    if is_adjusted:
        logger.warning(
            f"参考点数量({len(reference_points)})超过种群规模({population_size})的80%，"
            f"已自动校准种群规模为 {adjusted_pop_size}（大于参考点且为4的倍数）"
        )
        population_size = adjusted_pop_size

    logger.info(
        f"NSGA-III 初始化：{objective_dimension}个目标，"
        f"{len(reference_points)}个参考点（均匀{len(uniform_ref_points)}+边界{objective_dimension if add_boundary_points else 0}），"
        f"划分次数={reference_divisions}"
    )

    # 复用TOPSIS与收敛监控，与NSGA-II逻辑完全一致
    topsis_evaluator = TopsisAllMinEvaluator(decimal_reserve=6)
    convergence_weight = state_manager.topsis_weight
    convergence_monitor = ConvergenceMonitor(window_size=10, rel_tol=0.005)
    convergence_trigger_generation = max_generation_count

    # -------------------------- 2. 初始化混合种群（复用滚动排程种子） --------------------------
    current_population = init_mixed_population(reorder_job_sequence, state_manager)
    current_fitness = evaluate_population_fitness(current_population, state_manager)

    # -------------------------- 3. NSGA-III 主迭代循环 --------------------------
    for generation in range(max_generation_count):
        # 3.1 非支配排序，提取当代帕累托前沿
        generation_fronts, _ = fast_non_dominated_sorting(current_fitness)

        # 3.2 收敛监控（TOPSIS择优，与NSGA-II逻辑完全一致）
        if generation_fronts and len(generation_fronts[0]) > 0:
            current_pareto_indices = generation_fronts[0]
            current_pareto_set = [current_population[idx] for idx in current_pareto_indices]
            _, best_fitness_vector, _, _ = select_optimal_solution_by_weight(
                pareto_set=current_pareto_set,
                all_pop_fits=current_fitness,
                pareto_index_list=current_pareto_indices,
                weight=convergence_weight,
                topsis_evaluator=topsis_evaluator
            )
            convergence_monitor.add_generation_best(best_fitness_vector)

            if convergence_monitor.is_converged():
                convergence_trigger_generation = generation + 1
                logger.info(
                    f"【收敛触发】连续{convergence_monitor.window_size}代最优解波动小于{convergence_monitor.rel_tol * 100:.2f}%")
                logger.info(f"迭代提前终止：当前代数{generation + 1}，预设最大迭代{max_generation_count}")
                break

        # 3.3 生成下一代种群
        current_population, current_fitness = create_next_generation_nsga3(
            parent_population=current_population,
            parent_fitness=current_fitness,
            reference_points=reference_points,
            target_pop_size=population_size,
            state_manager=state_manager
        )

        logger.info(f"NSGA-III 第{generation + 1}/{max_generation_count}代迭代完成")

    # -------------------------- 4. 最终评估与帕累托筛选 --------------------------
    final_fitness = evaluate_population_fitness(current_population, state_manager)
    final_fronts, _ = fast_non_dominated_sorting(final_fitness)

    if final_fronts:
        first_front_indices = final_fronts[0]
        # 超过最大保留数时，按参考点筛选均匀解（替代原拥挤度筛选）
        if len(first_front_indices) > max_pareto_keep_count:
            first_front_fit = np.array([final_fitness[idx] for idx in first_front_indices])
            norm_first_front, _, _ = normalize_objectives(first_front_fit, trim_outlier=False)
            first_front_assoc, first_front_dists = associate_to_references(norm_first_front, reference_points)

            # 每个参考点保留距离最近的1个最优解
            reference_best_map: Dict[int, Tuple[int, float]] = {}
            for inner_idx, ref_id in enumerate(first_front_assoc):
                original_idx = first_front_indices[inner_idx]
                dist = first_front_dists[inner_idx]
                if ref_id not in reference_best_map or dist < reference_best_map[ref_id][1]:
                    reference_best_map[ref_id] = (original_idx, dist)

            keep_indices = [item[0] for item in reference_best_map.values()]
            # 仍超过限制则按距离从小到大截取
            if len(keep_indices) > max_pareto_keep_count:
                # 修复：直接用距离值排序，避免index线性查找
                keep_with_dist = [(idx, dist) for idx, dist in reference_best_map.values()]
                keep_with_dist.sort(key=lambda x: x[1])
                keep_indices = [idx for idx, _ in keep_with_dist[:max_pareto_keep_count]]
        else:
            keep_indices = first_front_indices
    else:
        keep_indices = []

    pareto_population = [current_population[idx] for idx in keep_indices]

    # -------------------------- 5. 保存历史解（兼容滚动排程） --------------------------
    state_manager.last_pareto_solutions = copy.deepcopy(pareto_population)

    logger.info(f"NSGA-III 优化完成：共迭代{convergence_trigger_generation}代，最终帕累托解{len(keep_indices)}个")
    return pareto_population, final_fitness, keep_indices