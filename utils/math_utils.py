import numpy as np
import random
from typing import List


def set_random_seed(seed: int = 42):
    """固定所有随机数种子，保证算法可复现"""
    random.seed(seed)
    np.random.seed(seed)


def pareto_front(fitness: np.ndarray) -> List[int]:
    """计算帕累托前沿（最小化问题）"""
    n = fitness.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0].tolist()


def normalize_fitness(fitness: np.ndarray) -> np.ndarray:
    """归一化适应度到[0,1]区间"""
    min_vals = np.min(fitness, axis=0)
    max_vals = np.max(fitness, axis=0)
    return (fitness - min_vals) / (max_vals - min_vals + 1e-8)