import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import List, Dict, Tuple, Any

# 导入被测函数
from core.base_ga import decode_chromosome

# ========== 模拟类型定义（与原项目对齐，避免导入报错） ==========
class MachineId(int):
    pass

class WorkerId(int):
    pass

class OperationId(int):
    pass

class JobId(int):
    pass

class OperationMetadata:
    def __init__(self, op_id, job_id, op_index, proc_time, tech_type, mat_ready=0.0):
        self.op_id = op_id
        self.belong_job_id = job_id
        self.op_index_in_job = op_index
        self.process_time = proc_time
        self.op_tech_type = tech_type
        self.material_ready_time = mat_ready
        self.business_op_id = f"BOP{op_id}"
        self.business_op_no = str(op_index)
        self.op_name = f"工序{op_id}"
        self.op_content = ""
        self.resource_group_id = 1
        self.resource_group_name = "RG1"

class WorkerSkillInfo:
    def __init__(self, speed_ratio_map: dict):
        self.tech_speed_ratio = speed_ratio_map

class MachineTechParam:
    def __init__(self, changeover_map: dict):
        self.changeover_time_map = changeover_map

class ResourceGroup:
    def __init__(self, mid_list, wid_list, max_parallel=1):
        self.machine_id_list = mid_list
        self.worker_id_list = wid_list
        self.worker_max_parallel = max_parallel

class ManualLockConfig:
    def __init__(self, lock_mac=False, lock_wk=False, fixed_mac=-1, fixed_wk=-1):
        self.lock_machine = lock_mac
        self.lock_worker = lock_wk
        self.fixed_machine_id = fixed_mac
        self.fixed_worker_id = fixed_wk

class SchedulingTrackers:
    def __init__(self):
        self.machine_last_end_time: Dict[MachineId, float] = {}
        self.machine_previous_technology_type: Dict[MachineId, int] = {}
        self.worker_task_intervals: Dict[WorkerId, List[Tuple[float, float]]] = {}
        self.worker_task_ends_heap: Dict[WorkerId, List[float]] = []
        self.job_last_operation_end_time: Dict[JobId, float] = {}
        self.machine_workloads: Dict[MachineId, List[float]] = {}
        self.worker_workloads: Dict[WorkerId, List[float]] = {}
        self.total_changeover_time = 0.0
        self.total_wip_wait_time = 0.0
        self.worker_switch_count = 0
        self.total_operation_count = 0

class OperationSchedulingResult:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# ========== 全局配置mock ==========
mock_cfg = MagicMock()
mock_cfg.PLAN_FROZEN_HORIZON = 10.0
mock_cfg.WORKER_SWITCH_COST = 2
mock_cfg.WIP_WEIGHT_COEFFICIENT = 1.0

# ========== 构造模拟 ProductionStateManager 工厂函数 ==========
def build_mock_state_manager(
    current_time: float = 0.0,
    op_meta_dict: Dict[int, OperationMetadata] = None,
    op2job: Dict[int, int] = None,
    resource_groups: Dict[int, ResourceGroup] = None,
    machine_available: Dict[int, bool] = None,
    worker_available: Dict[int, bool] = None,
    worker_skill: Dict[int, WorkerSkillInfo] = None,
    machine_tech: Dict[int, MachineTechParam] = None,
    manual_locks: Dict[int, ManualLockConfig] = None,
    last_schedule: Dict[int, dict] = None
):
    sm = MagicMock()
    sm.current_system_time = current_time
    sm.op_meta_dict = op_meta_dict or {}
    sm.operation_id_to_job_id = op2job or {}
    sm.resource_group_dict = resource_groups or {}
    sm.machine_available_dict = machine_available or {}
    sm.worker_available_dict = worker_available or {}
    sm.worker_skill_dict = worker_skill or {}
    sm.machine_tech_dict = machine_tech or {}
    sm.manual_lock_dict = manual_locks or {}
    sm.last_schedule_result = last_schedule or {}

    def get_rg_by_op(op_id):
        for rg in sm.resource_group_dict.values():
            meta = sm.op_meta_dict.get(op_id)
            if meta and meta.resource_group_id == getattr(rg, "id", 1):
                return rg
        return list(sm.resource_group_dict.values())[0] if sm.resource_group_dict else None
    sm.get_resource_group_by_op = get_rg_by_op

    sm.get_valid_start_time = lambda t: t
    sm.calculate_actual_work_end_time = lambda s, d: s + d
    sm.relative_hour_to_iso = lambda t: f"ISO_{t:.2f}"
    sm.is_op_manual_locked = lambda op: op in sm.manual_lock_dict
    sm.get_lock_info = lambda op: sm.manual_lock_dict.get(op, ManualLockConfig())
    sm.get_machine_overload_penalty = lambda mid, load: 0.0
    sm.calc_segment_overdue_penalty = lambda ft, jmeta: 0.0
    sm.get_optimizable_operation_ids = lambda: list(sm.op_meta_dict.keys())
    return sm


class TestDecodeChromosome(unittest.TestCase):
    def setUp(self):
        """每个测试用例执行前自动打补丁替换全局cfg"""
        self.cfg_patch = patch("core.base_ga.cfg", mock_cfg)
        self.cfg_patch.start()

    def tearDown(self):
        """每个用例结束后撤销补丁"""
        self.cfg_patch.stop()

    def test_decode_single_op_normal(self):
        """用例1：单工序基础正常解码"""
        op1_meta = OperationMetadata(op_id=1, job_id=101, op_index=0, proc_time=5.0, tech_type=1)
        op2job = {1: 101}
        rg = ResourceGroup(mid_list=[10, 11], wid_list=[20, 21], max_parallel=1)
        rg.id = 1
        state_mgr = build_mock_state_manager(
            current_time=0.0,
            op_meta_dict={1: op1_meta},
            op2job=op2job,
            resource_groups={1: rg},
            machine_available={10: True, 11: True},
            worker_available={20: True, 21: True}
        )
        chrom = {
            "op_sequence": [1],
            "resource_assign": [(10, 20)]
        }
        fit_vec, schedule_detail = decode_chromosome(chrom, state_mgr)

        self.assertEqual(len(schedule_detail), 1)
        rec = schedule_detail[0]
        self.assertEqual(rec["op_id"], 1)
        self.assertEqual(rec["machine_id"], 10)
        self.assertEqual(rec["worker_id"], 20)
        self.assertEqual(rec["start_time"], 0.0)
        self.assertEqual(rec["end_time"], 5.0)
        self.assertFalse(rec["is_frozen"])

        self.assertAlmostEqual(fit_vec[0], 5.0, places=5)
        self.assertAlmostEqual(fit_vec[1], 0.0, places=5)
        self.assertAlmostEqual(fit_vec[2], 0.0, places=5)

    def test_decode_two_ops_same_job_precedence(self):
        """用例2：同订单两道工序，校验工艺先后约束"""
        op1 = OperationMetadata(op_id=2, job_id=201, op_index=0, proc_time=3, tech_type=1)
        op2 = OperationMetadata(op_id=3, job_id=201, op_index=1, proc_time=4, tech_type=1)
        op2job = {2: 201, 3: 201}
        rg = ResourceGroup([5], [8], 1)
        rg.id = 1
        sm = build_mock_state_manager(
            op_meta_dict={2: op1, 3: op2},
            op2job=op2job,
            resource_groups={1: rg},
            machine_available={5: True},
            worker_available={8: True}
        )
        chrom = {
            "op_sequence": [2, 3],
            "resource_assign": [(5, 8), (5, 8)]
        }
        fit, detail = decode_chromosome(chrom, sm)
        rec1, rec2 = detail
        self.assertEqual(rec1["end_time"], 3.0)
        self.assertEqual(rec2["start_time"], 3.0)
        self.assertEqual(rec2["end_time"], 7.0)
        self.assertAlmostEqual(fit[0], 7.0, places=5)
        self.assertGreater(fit[5], 0)

    def test_decode_three_ops_same_job_precedence(self):
        """用例2：同订单两道工序，校验工艺先后约束"""
        op1 = OperationMetadata(op_id=2, job_id=201, op_index=0, proc_time=3, tech_type=1)
        op2 = OperationMetadata(op_id=3, job_id=201, op_index=1, proc_time=10, tech_type=1)
        op3 = OperationMetadata(op_id=4, job_id=201, op_index=2, proc_time=9, tech_type=1)
        op2job = {2: 201, 3: 201, 4: 201}
        rg = ResourceGroup([5], [8], 2)
        rg.id = 1
        sm = build_mock_state_manager(
            op_meta_dict={2: op1, 3: op2, 4: op3},
            op2job=op2job,
            resource_groups={1: rg},
            machine_available={5: True},
            worker_available={8: True}
        )
        chrom = {
            "op_sequence": [2, 3, 4],
            "resource_assign": [(5, 8), (5, 8),(5, 8)]
        }
        fit, detail = decode_chromosome(chrom, sm)
        rec1, rec2 = detail
        self.assertEqual(rec1["end_time"], 3.0)
        self.assertEqual(rec2["start_time"], 3.0)
        self.assertEqual(rec2["end_time"], 7.0)
        self.assertAlmostEqual(fit[0], 7.0, places=5)
        self.assertGreater(fit[5], 0)
    def test_decode_manual_lock_resource(self):
        """用例3：工序手动锁定机床，覆盖染色体原始分配"""
        op_meta = OperationMetadata(4, 301, 0, 6, 2)
        lock_cfg = ManualLockConfig(lock_mac=True, lock_wk=False, fixed_mac=15)
        sm = build_mock_state_manager(
            op_meta_dict={4: op_meta},
            op2job={4: 301},
            resource_groups={1: ResourceGroup([15, 16], [30], 1)},
            machine_available={15: True, 16: True},
            worker_available={30: True},
            manual_locks={4: lock_cfg}
        )
        chrom = {"op_sequence": [4], "resource_assign": [(16, 30)]}
        _, detail = decode_chromosome(chrom, sm)
        self.assertEqual(detail[0]["machine_id"], 15)

    def test_decode_frozen_operation_flag(self):
        """用例4：工序落在冻结区间，is_frozen标记为True"""
        op_meta = OperationMetadata(5, 401, 0, 2, 1)
        last_sch = {5: {"start_time": 8.0, "machine_id": 20, "worker_id": 40}}
        sm = build_mock_state_manager(
            current_time=0.0,
            op_meta_dict={5: op_meta},
            op2job={5: 401},
            resource_groups={1: ResourceGroup([20], [40], 1)},
            machine_available={20: True},
            worker_available={40: True},
            last_schedule=last_sch
        )
        chrom = {"op_sequence": [5], "resource_assign": [(20, 40)]}
        _, detail = decode_chromosome(chrom)
        self.assertTrue(detail[0]["is_frozen"])

    def test_decode_chromosome_length_mismatch_raise(self):
        """用例5：op_sequence与resource_assign长度不一致，抛ValueError"""
        sm = build_mock_state_manager()
        chrom = {
            "op_sequence": [1, 2],
            "resource_assign": [(10, 20)]
        }
        with self.assertRaises(ValueError) as cm:
            decode_chromosome(chrom, sm)
        self.assertIn("操作序列长度", str(cm.exception))

    def test_decode_unknown_op_id_raise(self):
        """用例6：包含不存在工序ID，抛ValueError"""
        sm = build_mock_state_manager(op_meta_dict={1: OperationMetadata(1, 101, 0, 2, 1)})
        chrom = {"op_sequence": [999], "resource_assign": [(10, 20)]}
        with self.assertRaises(ValueError) as cm:
            decode_chromosome(chrom, sm)
        self.assertIn("未知的操作ID", str(cm.exception))


if __name__ == '__main__':
    unittest.main()