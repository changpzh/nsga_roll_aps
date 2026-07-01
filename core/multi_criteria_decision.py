"""
模块：multi_criteria_decision.py
功能：多准则决策评价工具集
当前内置算法：TOPSIS（全部指标为数值越小越优）
修改说明：
1. 权重改为每次evaluate调用时传入，支持逐次自定义权重评价
2. evaluate返回：(最优样本下标, 全体样本排名列表)
"""
import numpy as np
import pandas as pd


class TopsisAllMinEvaluator:
    """
    TOPSIS 评价器：所有评价指标均为成本型（指标值越小，方案越优）
    完整计算链路：向量归一化 → 加权矩阵构建 → 求解正负理想解 → 欧式距离计算 → 贴近度计算 → 优劣排序
    权重规则：不再初始化固定权重，每次调用evaluate可单独传入本轮权重
    evaluate返回格式：(第一名样本下标, 原样本对应排名列表)
    """
    def __init__(self, decimal_reserve: int = 6):
        """
        初始化TOPSIS基础配置，权重移至evaluate方法动态传入
        :param decimal_reserve: 结果保留小数位数，默认6位，便于论文验算
        """
        # 外部配置参数
        self.decimal_places = decimal_reserve

        # ========== 运算中间缓存变量（运算后可直接访问） ==========
        self.original_data_matrix: np.ndarray | None = None       # 原始输入矩阵
        self.normalized_matrix: np.ndarray | None = None          # 向量归一化矩阵 Z
        self.weighted_matrix: np.ndarray | None = None            # 加权归一化矩阵 V
        self.positive_ideal_solution: np.ndarray | None = None   # 正理想解 V+（最优虚拟方案）
        self.negative_ideal_solution: np.ndarray | None = None    # 负理想解 V-（最差虚拟方案）
        self.distance_to_positive: np.ndarray | None = None      # D+：每个样本到正理想解距离
        self.distance_to_negative: np.ndarray | None = None       # D-：每个样本到负理想解距离
        self.closeness_coefficient: np.ndarray | None = None    # Ci 综合贴近度
        self.evaluation_result_df: pd.DataFrame | None = None     # 结构化评价结果表（保留，可按需查看）
        self.current_weight_array: np.ndarray | None = None       # 新增：存储当前本轮使用的权重
        self.rank_list: list | None = None                        # 全体样本排名列表
        self.best_sample_idx: int | None = None                   # 最优样本下标
        self.sorted_sample_idx:list | None = None                 # 全体升序排序后的样本下标列表

    import numpy as np

    def _range_corrected_weight(self, raw_weight: list, data_matrix: np.ndarray | list) -> list:
        """
        极差耦合修正TOPSIS权重
        公式: wj* = (wj / Rj) / Σ(wk / Rk)
        作用：抵消不同指标值域跨度不一致带来的打分敏感度差异
        :param raw_weight: 人工预设主观权重列表，长度等于指标数量
        :param data_matrix: 原始评价矩阵 shape=(样本数, 指标数)
        :return: corrected_weight: 修正后归一权重数组，总和=1
        """
        # 转为numpy数组
        mat = np.array(data_matrix, dtype=np.float64)
        w_raw = np.array(raw_weight, dtype=np.float64)

        # 逐列计算指标极差 Rj = max - min
        col_max = np.max(mat, axis=0)
        col_min = np.min(mat, axis=0)
        col_range = col_max - col_min

        # 防止某列全部数值相同，极差为0导致除零错误
        eps = 1e-9
        col_range[col_range < eps] = eps

        # 计算中间权重 wj / Rj
        temp_w = w_raw / col_range

        # 归一得到最终修正权重
        corrected_w = temp_w / np.sum(temp_w)

        return corrected_w.tolist()


    def _min_max_normalize(self, matrix: np.ndarray) -> np.ndarray:
        """
        多目标全部为【越小越优】，按列独立Min-Max极差归一
        归一公式：
        z_ij = (x_jmax - x_ij) / (x_jmax - x_jmin)
        x↓⇒z↑， 所以返回的归一矩阵里：数值越大 = 方案该项越优秀
        边界保护：若该列最大值=最小值，全部置 0，避免除以0报错
        :param matrix: 输入二维数组 shape=(种群数量, 目标维度)
        :return: 归一后同shape矩阵，每列 ∈ [0, 1]
        """
        n_rows, n_cols = matrix.shape
        norm_matrix = np.zeros_like(matrix, dtype=np.float64)

        for col_idx in range(n_cols):
            col_data = matrix[:, col_idx]
            col_min = np.min(col_data)
            col_max = np.max(col_data)

            # 该列所有值完全一致，分母防除零
            if abs(col_max - col_min) < 1e-12:
                norm_matrix[:, col_idx] = np.zeros(n_rows)
            else:
                norm_matrix[:, col_idx] = (col_max - col_data) / (col_max - col_min)
        return norm_matrix

    def _vector_normalize(self, raw_mat: np.ndarray) -> np.ndarray:
        """
        【内部私有方法】TOPSIS专用向量归一化，消除不同指标量纲影响
        公式：z_ij = x_ij / sqrt( 该指标列所有元素平方和 )
        x↓⇒z↓， 所以返回的归一矩阵里：数值越小 = 方案该项越优秀
        :param raw_mat: 原始二维评价矩阵 n行(样本) m列(指标)
        :return: 归一化完成矩阵
        """
        norm_result = raw_mat / np.sqrt((raw_mat ** 2).sum(axis=0, keepdims=True))
        return norm_result

    def _build_weighted_matrix(self, norm_mat: np.ndarray, input_weights: list | None) -> np.ndarray:
        """
        【内部私有方法】对归一化矩阵施加本轮传入指标权重，生成加权矩阵V
        公式：v_ij = w_j * z_ij
        :param norm_mat: 向量归一化之后的矩阵Z
        :param input_weights: 本轮评价传入权重，None则自动均分权重
        :return: 加权矩阵V
        """
        if norm_mat.ndim != 2:
            raise ValueError(f"输入矩阵必须为2维，当前为{norm_mat.ndim}维")
        _, indicator_num = norm_mat.shape

        # 权重归一化，保证权重总和=1
        if input_weights is None:
            weight_array = np.ones(indicator_num) / indicator_num
        else:
            weight_array = np.array(input_weights, dtype=float)
            if len(weight_array) != indicator_num:
                raise ValueError(
                    f"权重数量({len(input_weights)})与指标数量({indicator_num})不匹配，请检查输入"
                )
            if np.any(weight_array < 0):
                raise ValueError("指标权重不能为负数，请重新设置")
            total = np.sum(weight_array)
            if total == 0:
                raise ValueError("权重总和不能为0，请重新设置")

            weight_array = weight_array / total

        # 缓存本轮权重，便于外部查看
        self.current_weight_array = weight_array
        weighted_result = norm_mat * weight_array
        return weighted_result

    def _solve_min_ideal_solution(self, weighted_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        【内部私有方法】全成本型指标专属：求解正负理想解
        规则（全部指标越小越优）：
            正理想解V+ = 对应指标列最小值（理论最优值）
            负理想解V- = 对应指标列最大值（理论最差值）
        :param weighted_mat: 加权归一化矩阵V
        :return: (正理想解数组, 负理想解数组)
        """
        pos_ideal = np.min(weighted_mat, axis=0)
        neg_ideal = np.max(weighted_mat, axis=0)
        return pos_ideal, neg_ideal

    def _solve_max_ideal_solution(self, weighted_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        【内部私有方法】全成本型指标专属：求解正负理想解
        前置说明：已经经过极小值适配Min-Max归一
        归一后 z 越大代表原始指标越优
            正理想解V+ = 对应指标列最大值（归一后最优）
            负理想解V- = 对应指标列最小值（归一后最差）
        :param weighted_mat: 加权归一化矩阵V
        :return: (正理想解数组, 负理想解数组)
        """
        pos_ideal = np.max(weighted_mat, axis=0)
        neg_ideal = np.min(weighted_mat, axis=0)
        return pos_ideal, neg_ideal

    def _calc_euclidean_distance(self, matrix: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
        """
        【内部私有方法】批量计算每个样本到目标理想解的欧式距离
        公式：D = sqrt( Σ (v_ij - ideal_j)² )
        :param matrix: 加权矩阵V
        :param target_vector: 目标理想解（V+ 或 V-）
        :return: 一维数组：每个样本对应的距离
        """
        difference_square = (matrix - target_vector) ** 2
        sum_square_diff = np.sum(difference_square, axis=1)
        distance_array = np.sqrt(sum_square_diff)
        return distance_array

    def _calc_closeness(self, d_pos: np.ndarray, d_neg: np.ndarray) -> np.ndarray:
        """
        【内部私有方法】计算综合贴近度Ci
        公式：Ci = D- / (D+ + D-)
        含义：Ci取值0~1，数值越大代表方案综合表现越优秀
        """
        closeness = d_neg / (d_pos + d_neg)
        return closeness

    def evaluate(self, input_data: list | np.ndarray, weight: list = None) -> tuple[int, list[int]]:
        """
        【对外公开入口】执行完整TOPSIS评价流程
        :param input_data: 二维列表/二维数组，待评价方案原始数据
        :param weight: 本轮评价指标权重列表；传None则自动均分权重
        :return: tuple(最优样本下标, 名次从优到劣排序的样本下标列表, 原始顺序对应名次列表)
                 1. best_idx：排名第一的样本下标（从0开始）
                 2. sorted_sample_idx：样本下标按综合名次从第1名→最后一名排序
        """
        # 1. 原始数据类型转换存储
        self.original_data_matrix = np.array(input_data, dtype=float)
        sample_total, indicator_total = self.original_data_matrix.shape

        # 2. 分步运算，每一步结果存入实例属性，可随时调取查看
        # self.normalized_matrix = self._vector_normalize(self.original_data_matrix)
        self.normalized_matrix = self._min_max_normalize(self.original_data_matrix)

        corrected_weight = self._range_corrected_weight(weight, self.original_data_matrix)
        # 传入本轮权重进行加权计算
        self.weighted_matrix = self._build_weighted_matrix(self.normalized_matrix, input_weights=corrected_weight)
        self.positive_ideal_solution, self.negative_ideal_solution = self._solve_max_ideal_solution(self.weighted_matrix)
        self.distance_to_positive = self._calc_euclidean_distance(self.weighted_matrix, self.positive_ideal_solution)
        self.distance_to_negative = self._calc_euclidean_distance(self.weighted_matrix, self.negative_ideal_solution)
        self.closeness_coefficient = self._calc_closeness(self.distance_to_positive, self.distance_to_negative)

        # 3. 生成排名：贴近度降序排序，匹配样本正确名次
        desc_sorted_sample_index = np.argsort(-self.closeness_coefficient)

        # 算出原数组中各个位置的排名
        rank_array = np.empty_like(self.closeness_coefficient, dtype=int)
        for rank_position, sample_index in enumerate(desc_sorted_sample_index):
            rank_array[sample_index] = rank_position + 1

        # 赋值实例缓存
        # self.rank_list = rank_array.tolist()
        self.best_sample_idx = int(desc_sorted_sample_index[0])
        self.sorted_sample_idx = desc_sorted_sample_index.tolist()

        # 返回二元组：第一名下标、名次排序下标列表
        return self.best_sample_idx, self.sorted_sample_idx
