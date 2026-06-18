# core/nsga2_operator.py
import numpy as np
import copy
from typing import List, Dict, Tuple, Any
from core.state_manager import ProductionStateManager
import config.settings as cfg
from config.settings import (
    POPULATION_SIZE, MAX_GENERATION, ELITE_RATE, MAX_FRONT_NUM, CROSSOVER_RATE
)
from utils.log_utils import get_logger
# 导入base_ga里的所有公共函数
from core.base_ga import (
    init_mixed_population, decode_chromosome, fast_non_dominated_sorting,
    pox_crossover, mutate_chromosome,evaluate_population_fitness
)

logger = get_logger(__name__)


def calculate_crowding_distance(fits: List[List[float]], front_idx_list: List[int]) -> List[float]:
    """NSGA-II独有：拥挤距离计算"""
    f_size = len(front_idx_list)
    if f_size <=2:
        return [np.inf]*f_size
    dist = [0.0]*f_size
    obj_num = len(fits[0])
    for obj in range(obj_num):
        sorted_idx = sorted(front_idx_list, key=lambda x: fits[x][obj])
        dist[0] = np.inf
        dist[-1] = np.inf
        f_min = fits[sorted_idx[0]][obj]
        f_max = fits[sorted_idx[-1]][obj]
        if f_max == f_min:
            continue
        for i in range(1, f_size-1):
            pre = fits[sorted_idx[i-1]][obj]
            nxt = fits[sorted_idx[i+1]][obj]
            dist[i] += (nxt - pre)/(f_max - f_min)
    return dist


def tournament_selection(population: List[dict], fits: List[List[float]], state_manager: ProductionStateManager) -> dict:
    """NSGA-II独有：基于拥挤度的锦标赛选择"""
    sample_size =4
    all_index = list(range(len(population)))
    sample_idx = np.random.choice(all_index, size=sample_size, replace=False)
    sample_pop = [population[i] for i in sample_idx]
    sample_fit = [fits[i] for i in sample_idx]
    fronts, _ = fast_non_dominated_sorting(sample_fit)
    first_f = fronts[0]
    cd = calculate_crowding_distance(sample_fit, first_f)
    sorted_ind = sorted(zip(first_f, cd), key=lambda x:-x[1])
    return sample_pop[sorted_ind[0][0]]


def nsga2_rolling_schedule(state_manager: ProductionStateManager, reorder_job_seq: List[int]) -> Tuple[List[dict], List[List[float]], List[int]]:
    """
    NSGA-II滚动排程主入口:执行多目标进化优化，返回最终种群、适应度矩阵及历史前沿解
    Returns:
        Tuple[最终种群, 对应适应度列表, 历史帕累托前沿索引]
    整体算法流程：
    1. 初始化第一代种群
    2. 重复迭代进化种群，每代按以下方式迭代进化
        2.1 保留最优秀的父代染色体到下一代
        2.2 下一代不足部分种群染色体，通过精英选择父代，交叉、变异形成子代
    3. 最后获取最后一代种群的帕累托解集
    4. 保留历史解
    """
    # 新增调试代码
    print("所有工序状态:", state_manager.operation_status_dict)
    print("OP_STATUS_OPTIMIZABLE的值:", cfg.OP_STATUS_OPTIMIZABLE)

    # -------------------------- 1. 读取全局算法超参 --------------------------
    population_size = POPULATION_SIZE        # 种群总规模
    max_generation_num = MAX_GENERATION          # 最大迭代代数
    elite_count = int(population_size * ELITE_RATE)  # 每代精英保留数量上限
    max_front_solution_num = MAX_FRONT_NUM         # 最终第一层前沿最多保留多少个均匀解

    # -------------------------- 2. 初始化混合种群（调用base_ga公共函数） --------------------------
    population = init_mixed_population(reorder_job_seq, state_manager)

    # -------------------------- 3. NSGA-II主迭代循环 --------------------------
    for generation in range(max_generation_num):
        # 3.1 评估当前种群
        fitness_list = evaluate_population_fitness(population, state_manager)
        # 3.2 快速非支配排序与前沿提取
        frontiers, _ = fast_non_dominated_sorting(fitness_list)
        # 3.3 构建下一代种群（精英保留 + 交叉变异生成）
        next_population = create_next_generation(
            population=population,
            fitness_list=fitness_list,
            frontiers=frontiers,
            elite_count=elite_count,
            population_size=population_size,
            state_manager=state_manager
        )

        # 更新种群
        population = next_population
        logger.info(f"NSGA-II 第{generation+1}/{max_generation_num}代迭代完成")

    # -------------------------- 4. 迭代结束，计算最终适应度 --------------------------
    final_fitness = evaluate_population_fitness(population, state_manager)

    # -------------------------- 5. 筛选第一层帕累托前沿 --------------------------
    final_frontiers, _ = fast_non_dominated_sorting(final_fitness)
    if final_frontiers:
        first_front_ids = final_frontiers[0]
        if len(first_front_ids) > max_front_solution_num:
            crowd_distance_list = calculate_crowding_distance(final_fitness, first_front_ids)
            sorted_ids = sorted(zip(first_front_ids, crowd_distance_list), key=lambda x: -x[1])
            keep_ids = [idx for idx, _ in sorted_ids[:max_front_solution_num]]
        else:
            keep_ids = first_front_ids
    else:
        keep_ids = []

    pareto_population = [population[i] for i in keep_ids]
    # -------------------------- 6. 保存历史解 --------------------------
    state_manager.last_pareto_solutions = copy.deepcopy(pareto_population)

    return pareto_population, final_fitness, keep_ids

def create_next_generation(
    population: List[Dict[str, Any]],
    fitness_list: List[List[float]],
    frontiers: List[List[int]],
    elite_count: int,
    population_size: int,
    state_manager: ProductionStateManager
) -> List[Dict[str, Any]]:
    """
    根据精英保留策略和交叉变异生成新一代种群。
    步骤：
    1. 按前沿顺序将个体加入新种群，直到达到 elite_count（精英保留）。
    2. 通过锦标赛选择父代，进行交叉和变异，填充剩余个体。
    """
    next_pop = []

    # ---------- 1.精英保留 ----------
    for front_ids in frontiers:
        # 如果加入整个前沿后仍不超过 elite_count，则全部保留
        if len(next_pop) + len(front_ids) <= elite_count:
            next_pop.extend([population[i] for i in front_ids])
        else:
            # 否则按拥挤度降序选择精英个体
            crowd_dist = calculate_crowding_distance(fitness_list, front_ids)
            sorted_front = sorted(
                zip(front_ids, crowd_dist),
                key=lambda x: -x[1]          # 拥挤度降序
            )
            need_num = elite_count - len(next_pop)
            selected_ids = [idx for idx, _ in sorted_front[:need_num]]
            next_pop.extend([population[id] for id in selected_ids])
            break   # 精英已填满

    # ----------2.交叉/变异填充剩余个体 ----------
    while len(next_pop) < population_size:
        # 锦标赛选择两个父代
        parent1 = tournament_selection(population, fitness_list, state_manager)
        parent2 = tournament_selection(population, fitness_list, state_manager)

        # 交叉（依据概率）
        if np.random.random() < CROSSOVER_RATE:
            child1, child2 = pox_crossover(parent1, parent2, state_manager)
        else:
            child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        # 变异
        child1 = mutate_chromosome(child1, state_manager)
        child2 = mutate_chromosome(child2, state_manager)

        # 加入子代，保持种群大小
        next_pop.append(child1)
        if len(next_pop) < population_size:
            next_pop.append(child2)

    return next_pop