import unittest
from core.base_ga import fast_non_dominated_sorting


class TestFastNonDominatedSorting(unittest.TestCase):
    """
    快速非支配排序算法测试集
    所有测试用例均基于【最小化多目标优化问题】设计
    支配定义：个体A支配B ⇨ A所有目标 ≤ B所有目标，且至少有一个目标严格小于B
    """

    def test_single_individual(self):
        """边界测试：单个个体，必然属于第一层前沿"""
        pop_fits = [[1.0, 2.0]]
        fronts, rank = fast_non_dominated_sorting(pop_fits)

        self.assertEqual(len(fronts), 1)
        self.assertEqual(sorted(fronts[0]), [0])
        self.assertEqual(rank[0], 0)

    def test_two_individuals_non_dominated(self):
        """两个个体互不支配，同属第一层前沿"""
        # 个体0：[1, 3]，个体1：[2, 2]
        # 0的目标1更好，1的目标2更好，互不支配
        pop_fits = [[1.0, 3.0], [2.0, 2.0]]
        fronts, rank = fast_non_dominated_sorting(pop_fits)

        self.assertEqual(len(fronts), 1)
        self.assertEqual(sorted(fronts[0]), [0, 1])
        self.assertEqual(rank, [0, 0])

    def test_two_individuals_dominated(self):
        """两个个体存在支配关系，分属两层前沿"""
        # 个体0：[2, 3]，个体1：[1, 2]
        # 个体1完全支配个体0
        pop_fits = [[2.0, 3.0], [1.0, 2.0]]
        fronts, rank = fast_non_dominated_sorting(pop_fits)

        self.assertEqual(len(fronts), 2)
        self.assertEqual(sorted(fronts[0]), [1])
        self.assertEqual(sorted(fronts[1]), [0])
        self.assertEqual(rank, [1, 0])

    def test_three_individuals_all_non_dominated(self):
        """经典三目标互不支配场景，全部属于第一层前沿"""
        # 个体0：[1, 5]
        # 个体1：[2, 3]
        # 个体2：[3, 1]
        # 三个个体形成帕累托前沿，无任何支配关系
        pop_fits = [[1.0, 5.0], [2.0, 3.0], [3.0, 1.0]]
        fronts, rank = fast_non_dominated_sorting(pop_fits)

        self.assertEqual(len(fronts), 1)
        self.assertEqual(sorted(fronts[0]), [0, 1, 2])
        self.assertEqual(rank, [0, 0, 0])

    def test_two_fronts_basic(self):
        """两层前沿基础场景"""
        # 个体0：[1, 4] → 被个体1支配
        # 个体1：[1, 3] → 第一层
        # 个体2：[2, 2] → 第一层
        # 个体3：[3, 1] → 第一层
        pop_fits = [[1.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]
        fronts, rank = fast_non_dominated_sorting(pop_fits)

        self.assertEqual(len(fronts), 2)
        self.assertEqual(sorted(fronts[0]), [1, 2, 3])
        self.assertEqual(sorted(fronts[1]), [0])
        self.assertEqual(rank, [1, 0, 0, 0])

    def test_three_fronts_chain(self):
        """三层全支配链场景"""
        # 个体0：[1, 1] → 第一层
        # 个体1：[2, 2] → 被0支配，第二层
        # 个体2：[3, 3] → 被0和1支配，第三层
        pop_fits = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        fronts, rank = fast_non_dominated_sorting(pop_fits)

        self.assertEqual(len(fronts), 3)
        self.assertEqual(sorted(fronts[0]), [0])
        self.assertEqual(sorted(fronts[1]), [1])
        self.assertEqual(sorted(fronts[2]), [2])
        self.assertEqual(rank, [0, 1, 2])

    def test_identical_fitness_individuals(self):
        """两个个体适应度完全相同，互不支配，同属第一层"""
        # 完全相同的个体不互相支配
        pop_fits = [[2.0, 3.0], [2.0, 3.0], [1.0, 4.0]]
        fronts, rank = fast_non_dominated_sorting(pop_fits)

        self.assertEqual(len(fronts), 2)
        self.assertEqual(sorted(fronts[0]), [0, 1])
        self.assertEqual(sorted(fronts[1]), [2])
        self.assertEqual(rank, [0, 0, 1])

    def test_three_objective_problem(self):
        """三目标问题兼容性测试"""
        # 验证算法支持任意数量的目标函数
        # 个体0：[1, 2, 3]
        # 个体1：[2, 1, 3]
        # 个体2：[2, 2, 2]
        # 三个个体互不支配，同属第一层
        pop_fits = [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [2.0, 2.0, 2.0]]
        fronts, rank = fast_non_dominated_sorting(pop_fits)

        self.assertEqual(len(fronts), 1)
        self.assertEqual(sorted(fronts[0]), [0, 1, 2])
        self.assertEqual(rank, [0, 0, 0])

    def test_complex_mixed_dominance(self):
        """复杂混合支配场景，四层前沿"""
        # 预期前沿：
        # 第一层：0(1,6), 1(2,5), 2(3,2), 3(4,1)
        # 第二层：4(5,3) 被2,3支配
        # 第三层：5(6,4) 被2,3,4支配
        pop_fits = [
            [1.0, 6.0],
            [2.0, 5.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [5.0, 3.0],
            [6.0, 4.0]
        ]
        fronts, rank = fast_non_dominated_sorting(pop_fits)

        self.assertEqual(len(fronts), 3)
        self.assertEqual(sorted(fronts[0]), [0, 1, 2, 3])
        self.assertEqual(sorted(fronts[1]), [4])
        self.assertEqual(sorted(fronts[2]), [5])
        self.assertEqual(rank, [0, 0, 0, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()