import sys
import os
from datetime import datetime

# 项目根目录加入搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from typing import List, Dict, Tuple

# 全局配置常量（和真实项目一致）
from config.settings import PLAN_FROZEN_HORIZON, OP_STATUS_OPTIMIZABLE
# 业务函数 & 真实状态管理类
from core.base_ga import pox_crossover
from core.state_manager import ProductionStateManager

# -------------------------- 仅用于测试的简易资源组模拟类（和项目 ResourceGroup 结构对齐） --------------------------
class MockResourceGroup:
    def __init__(self, machine_id_list: List[int], worker_id_list: List[int]):
        self.machine_id_list = machine_id_list
        self.worker_id_list = worker_id_list

# -------------------------- 校验工具函数 --------------------------
def check_frozen_unchanged(parent_seq: List[int], child_seq: List[int], frozen_ops: List[int]) -> bool:
    """校验：子代前缀冻结工序和父代完全一致"""
    return child_seq[:len(frozen_ops)] == frozen_ops

def check_job_internal_order(seq: List[int], op_job_map: Dict[int, int]) -> bool:
    """校验POX核心约束：同一订单内部工序相对顺序不变"""
    job_op_record = {}
    for op in seq:
        jid = op_job_map[op]
        if jid not in job_op_record:
            job_op_record[jid] = []
        job_op_record[jid].append(op)

    job_origin_ops = {}
    for op, jid in op_job_map.items():
        if jid not in job_origin_ops:
            job_origin_ops[jid] = []
        job_origin_ops[jid].append(op)

    for j, ops in job_op_record.items():
        if ops != job_origin_ops[j]:
            return False
    return True

# -------------------------- 测试用例1：带冻结工序正常POX交叉 --------------------------
def test_pox_crossover_normal_with_frozen():
    print("===== 测试用例1：带冻结工序正常POX交叉 =====")
    # 1. 构造工序-订单映射
    op_job = {
        101: 1, 102: 1,
        201: 2, 202: 2,
        301: 3, 302: 3,
        401: 4, 402: 4,
        501: 5, 502: 5
    }
    # 资源组对象
    rg1 = MockResourceGroup(machine_id_list=[1, 2], worker_id_list=[11, 12])

    # 实例化真实 ProductionStateManager
    sm = ProductionStateManager()
    # 手动填充必要映射（模拟加载数据库数据）
    sm.operation_id_to_job_id = op_job
    for op_id in op_job:
        sm.resource_group_dict[op_id] = rg1
        sm.operation_status_dict.append(OP_STATUS_OPTIMIZABLE)

    # 历史调度结果（对应 last_schedule_result）
    last_schedule = {
        301: {"start_time": 5.0, "machine_id": 1, "worker_id": 11},
        302: {"start_time": 6.0, "machine_id": 2, "worker_id": 12},
        101: {"start_time": 10.0, "machine_id": 1, "worker_id": 11},
        102: {"start_time": 12.0, "machine_id": 2, "worker_id": 12},
        201: {"start_time": 31.0, "machine_id": 1, "worker_id": 11},
        202: {"start_time": 35.0, "machine_id": 2, "worker_id": 12},
    }
    sm.last_schedule_result = last_schedule

    # 系统时间设置
    sm.current_system_time = 20.0

    # 设备、工人可用标记
    sm.machine_available_dict = [True, True]
    sm.worker_available_dict = [True, True]

    # 父代染色体
    p1 = {
        "op_sequence": [301, 302, 101, 102, 201, 202, 401,402, 501,502],
        "resource_assign": [(1, 11), (2, 12), (1, 11), (2, 12), (1, 11), (2, 12),(1, 11), (2, 12),(1, 11), (2, 12)]
    }
    p2 = {
        "op_sequence": [301, 302, 101, 102, 401, 402, 201, 202, 501,502],
        "resource_assign": [(1, 11), (2, 12), (1, 11), (2, 12), (1, 11), (2, 12),(1, 11), (2, 12),(1, 11), (2, 12)]
    }

    np.random.seed(42)
    c1, c2 = pox_crossover(p1, p2, sm)

    frozen_boundary = sm.current_system_time + PLAN_FROZEN_HORIZON
    frozen_ops = [
        op for op in p1["op_sequence"]
        if op in sm.last_schedule_result and sm.last_schedule_result[op]["start_time"] < frozen_boundary
    ]

    # 断言校验
    assert check_frozen_unchanged(p1["op_sequence"], c1["op_sequence"], frozen_ops)
    assert check_frozen_unchanged(p2["op_sequence"], c2["op_sequence"], frozen_ops)
    assert set(c1["op_sequence"]) == set(p1["op_sequence"])
    assert set(c2["op_sequence"]) == set(p1["op_sequence"])
    assert check_job_internal_order(c1["op_sequence"], op_job)
    assert check_job_internal_order(c2["op_sequence"], op_job)

    for idx, op in enumerate(frozen_ops):
        res = sm.last_schedule_result[op]
        assert c1["resource_assign"][idx] == (res["machine_id"], res["worker_id"])
        assert c2["resource_assign"][idx] == (res["machine_id"], res["worker_id"])

    print(f"父代1序列: {p1['op_sequence']}")
    print(f"父代2序列: {p2['op_sequence']}")
    print(f"子代1序列: {c1['op_sequence']}")
    print(f"子代2序列: {c2['op_sequence']}")
    print("✅ 用例1通过：冻结不变、POX交叉合法、资源分配合规\n")

# -------------------------- 测试用例2：所有工序冻结，跳过交叉返回原副本 --------------------------
def test_pox_all_frozen_skip_cross():
    print("===== 测试用例2：所有工序冻结，跳过交叉返回原副本 =====")
    op_job = {101: 1, 102: 1, 201: 2}
    rg = MockResourceGroup([1], [11])

    sm = ProductionStateManager()
    sm.operation_id_to_job_id = op_job
    for op in op_job:
        sm.resource_group_dict[op] = rg
        sm.operation_status_dict.append(OP_STATUS_OPTIMIZABLE)

    last_schedule = {
        101: {"start_time": 5, "machine_id": 1, "worker_id": 11},
        102: {"start_time": 8, "machine_id": 1, "worker_id": 11},
        201: {"start_time": 11, "machine_id": 1, "worker_id": 11},
    }
    sm.last_schedule_result = last_schedule
    sm.current_system_time = 15
    sm.machine_available_dict = [True]
    sm.worker_available_dict = [True]

    p1 = {"op_sequence": [101, 102, 201], "resource_assign": [(1, 11), (1, 11), (1, 11)]}
    p2 = {"op_sequence": [201, 101, 102], "resource_assign": [(1, 11), (1, 11), (1, 11)]}
    c1, c2 = pox_crossover(p1, p2, sm)

    assert c1 == p1
    assert c2 == p2
    print("✅ 用例2通过：全冻结直接返回父代，未执行交叉\n")

# -------------------------- 测试用例3：浮动工序只有一个订单，不执行交叉 --------------------------
def test_pox_free_single_job():
    print("===== 测试用例3：浮动工序只有一个订单，不执行交叉 =====")
    op_job = {101: 1, 102: 1, 103: 1}
    rg = MockResourceGroup([1], [11])

    sm = ProductionStateManager()
    sm.operation_id_to_job_id = op_job
    for op in op_job:
        sm.resource_group_dict[op] = rg
        sm.operation_status_dict.append(OP_STATUS_OPTIMIZABLE)

    sm.last_schedule_result = {
        101: {"start_time": 10, "machine_id": 1, "worker_id": 11}
    }
    sm.current_system_time = 15
    sm.machine_available_dict = [True]
    sm.worker_available_dict = [True]

    p1 = {"op_sequence": [101, 102, 103], "resource_assign": [(1, 11), (1, 11), (1, 11)]}
    p2 = {"op_sequence": [101, 103, 102], "resource_assign": [(1, 11), (1, 11), (1, 11)]}
    c1, c2 = pox_crossover(p1, p2, sm)

    assert c1["op_sequence"] == [101, 102, 103]
    assert c2["op_sequence"] == [101, 103, 102]
    print("✅ 用例3通过：单订单浮动序列跳过交叉，原样继承\n")

if __name__ == "__main__":
    test_pox_crossover_normal_with_frozen()
    # test_pox_all_frozen_skip_cross()
    # test_pox_free_single_job()