import numpy as np
import copy
from typing import List, Dict, Tuple
from core.state_manager import ProductionStateManager
import config.settings as cfg
from config.settings import WORKER_SWITCH_COST, WIP_WEIGHT_COEFFICIENT, POPULATION_SIZE, MAX_GENERATION, ELITE_RATE, MAX_FRONT_NUM, MUTATION_RATE, CROSSOVER_RATE, ROLLING_HISTORY_SEED_RATIO, ROLLING_HEURISTIC_SEED_RATIO, ROLLING_PERTURB_RATE
from nsga2_operator import init_mixed_population, decode_chromosome, fast_non_dominated_sorting,tournament_selection,mutate_chromosome

def nsga3_rolling_schedule(state_manager: ProductionStateManager, reorder_job_seq: List[int], divisions: int = 3) -> Tuple[List[dict], List[List[float]], List[int]]:
    pop_size = POPULATION_SIZE
    max_gen = MAX_GENERATION
    elite_num = int(pop_size * ELITE_RATE)
    max_front = MAX_FRONT_NUM
    obj_dim = 6  # 你的六个优化目标
    # 生成全局均匀参考点，全程复用
    ref_points = generate_reference_points(obj_dim, divisions)
    population = init_mixed_population(reorder_job_seq, state_manager)

    for gen in range(max_gen):
        fits = []
        for chrom in population:
            fv, _ = decode_chromosome(chrom, state_manager)
            fits.append(fv)
        fits = np.array(fits)
        fronts, _ = fast_non_dominated_sorting(fits)
        new_pop = []
        # ========== NSGA-III 核心：分层选取 + 参考点小生境替代拥挤距离 ==========
        for front in fronts:
            if len(new_pop) + len(front) <= elite_num:
                new_pop.extend([population[i] for i in front])
            else:
                # 剩余需要选取的个体数量
                rest_num = elite_num - len(new_pop)
                front_ind = np.array(front)
                front_fits = fits[front_ind]
                # 1. 目标归一化
                norm_fits, _ = normalize_objectives(front_fits)
                # 2. 每个个体关联最近的参考点
                dist_all = []
                nearest_ref_idx = []
                for p in norm_fits:
                    dists = perpendicular_distance(np.tile(p, (len(ref_points),1)), ref_points)
                    min_idx = np.argmin(dists)
                    nearest_ref_idx.append(min_idx)
                    dist_all.append(dists[min_idx])
                # 3. 统计已选个体的参考点小生境数量
                selected_ref_count = np.zeros(len(ref_points))
                # 遍历已经选入new_pop的个体，统计归属
                all_selected_fits = np.array([fits[population.index(c)] for c in new_pop])
                if len(all_selected_fits) > 0:
                    norm_selected, _ = normalize_objectives(all_selected_fits)
                    for p in norm_selected:
                        dists = perpendicular_distance(np.tile(p, (len(ref_points),1)), ref_points)
                        selected_ref_count[np.argmin(dists)] += 1
                # 4. 优先选取小生境数量最少的参考点内的个体
                score = []
                for rid, d in zip(nearest_ref_idx, dist_all):
                    score.append((selected_ref_count[rid], d))
                # 优先：已选数量少 > 垂直距离近
                sorted_local = sorted(enumerate(score), key=lambda x: (x[1][0], x[1][1]))
                pick_idx = [front_ind[i[0]] for i in sorted_local[:rest_num]]
                new_pop.extend([population[i] for i in pick_idx])
                break
        # 锦标赛、交叉、变异 和 NSGA-II 完全一致，可以复用
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fits, state_manager)
            p2 = tournament_selection(population, fits, state_manager)
            if np.random.random() < CROSSOVER_RATE:
                c1, c2 = pox_crossover(p1, p2, state_manager)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
            c1 = mutate_chromosome(c1, state_manager)
            c2 = mutate_chromosome(c2, state_manager)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        population = new_pop

    # 迭代结束后前沿筛选
    final_fits = []
    for chrom in population:
        fv, _ = decode_chromosome(chrom, state_manager)
        final_fits.append(fv)
    final_fits = np.array(final_fits)
    fronts, _ = fast_non_dominated_sorting(final_fits)
    first_front = fronts[0]
    # NSGA-III最终筛选依旧使用参考点分布择优，不再使用拥挤距离
    front_fits = final_fits[first_front]
    norm_front, _ = normalize_objectives(front_fits)
    ref_count = np.zeros(len(ref_points))
    final_score = []
    for p in norm_front:
        dists = perpendicular_distance(np.tile(p, (len(ref_points),1)), ref_points)
        min_r = np.argmin(dists)
        final_score.append((ref_count[min_r], dists[min_r]))
        ref_count[min_r] += 1
    sorted_final = sorted(zip(first_front, final_score), key=lambda x:(x[1][0], x[1][1]))
    keep_idx = [item[0] for item in sorted_final[:max_front]]
    pareto_set = [population[i] for i in keep_idx]
    state_manager.last_pareto_solutions = copy.deepcopy(pareto_set)
    return pareto_set, final_fits.tolist(), keep_idx

def generate_reference_points(num_obj: int, divisions: int) -> np.ndarray:
    """
    生成均匀分布参考点，Das-Dennis分层采样
    :param num_obj: 目标维度，你的项目为6
    :param divisions: 每个维度分割份数
    :return: 参考点矩阵 shape=(point_num, num_obj)
    """
    def recursive_gen(level, start, current, res):
        if level == num_obj - 1:
            current.append(divisions - sum(current))
            res.append(current.copy())
            current.pop()
            return
        for i in range(start, divisions - sum(current) + 1):
            current.append(i)
            recursive_gen(level + 1, 0, current, res)
            current.pop()
    ref_points = []
    recursive_gen(0, 0, [], ref_points)
    ref_points = np.array(ref_points) / divisions
    return ref_points

def normalize_objectives(fits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """目标值归一化，计算理想点"""
    n, m = fits.shape
    ideal_point = np.min(fits, axis=0)
    norm_fits = fits - ideal_point
    return norm_fits, ideal_point

def perpendicular_distance(points: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    """计算个体到参考点连线的垂直距离"""
    dot = np.sum(points * ref_point, axis=1)
    ref_norm = np.sum(ref_point ** 2)
    proj = (dot / ref_norm)[:, np.newaxis] * ref_point
    dist = np.sqrt(np.sum((points - proj) ** 2, axis=1))
    return dist