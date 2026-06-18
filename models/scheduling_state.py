"""
排程过程核心状态数据结构
所有解码算法共用的通用状态和结果定义
示例业务背景：机械加工车间，工艺类型1=车削、2=铣削、3=磨削
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

from .base_types import MachineId, WorkerId, JobId, OperationId


@dataclass
class SchedulingTrackers:
    """
    解码过程全局状态跟踪器
    所有累计状态集中管理，避免函数间传递零散变量
    设计原则：只存状态，不包含任何业务逻辑
    """
    # ============================== 机器状态 ==============================
    # 每台机器最后一个任务的结束时间 | 核心：判断机器可用时间 + 计算总工期 | 例子：{MachineId(1): 8.5, MachineId(2): 12.0}
    machine_last_end_time: Dict[MachineId, float] = field(default_factory=dict)
    # 每台机器上一个任务的工艺类型 | 核心：判断是否需要机器换型 + 计算换型时间 | 例子：{MachineId(1): 1, MachineId(2): 2}
    machine_previous_technology_type: Dict[MachineId, int] = field(default_factory=dict)
    # 每台机器所有任务的实际加工时间(每个任务时长)列表 | 核心：计算机器总负载 + 负载不平衡度 | 例子：{MachineId(1): [2.0, 3.5], MachineId(2): [4.0]}
    machine_workloads: Dict[MachineId, List[float]] = field(default_factory=lambda: defaultdict(list))

    # ============================== 工人状态 ==============================
    # 每个工人所有任务的时间区间列表[(start, end)] | 用途：统计切换次数 + 甘特图可视化 | 例子：{WorkerId(101): [(0.0, 2.0), (2.0, 5.5)]}
    worker_task_intervals: Dict[WorkerId, List[Tuple[float, float]]] = field(default_factory=lambda: defaultdict(list))
    # 每个工人任务结束时间的最小堆 | 核心优化：O(log n)判断工人并行约束 | 例子：{WorkerId(101): [2.0, 5.5]} (堆顶是最早结束时间)
    worker_task_ends_heap: Dict[WorkerId, List[float]] = field(default_factory=lambda: defaultdict(list))
    # 每个工人所有任务的实际加工时间(每个任务时长)列表 | 核心：计算工人总负载 + 负载不平衡度 | 例子：{WorkerId(101): [2.0, 3.5], WorkerId(102): [4.0]}
    worker_workloads: Dict[WorkerId, List[float]] = field(default_factory=lambda: defaultdict(list))

    # ============================== 工件状态 ==============================
    # 每个工件最后一个工序的结束时间 | 核心：判断工件前序约束 + 计算订单逾期惩罚 | 例子：{JobId(1001): 8.5, JobId(1002): 12.0}
    job_last_operation_end_time: Dict[JobId, float] = field(default_factory=dict)

    # ============================== 全局累计指标 ==============================
    # 所有机器的总换型时间 | 直接作为多目标优化的第3个目标 | 例子：2.5 (总共花了2.5小时换型)
    total_changeover_time: float = 0.0
    # 所有在制品的总等待时间 | 用于计算第6个目标：在制品加权等待成本 | 例子：15.0 (所有工件总共等了15小时)
    total_wip_wait_time: float = 0.0
    # 已调度的总操作数 | 用途：统计指标 + 扩展其他成本计算 | 例子：120 (本次共调度了120道工序)
    total_operation_count: int = 0
    # 所有工人任务切换总次数（仅统计连续任务之间的切换）| 核心：计算工人切换成本 | 例子：35 (所有工人总共切换了35次任务)
    worker_switch_count: int = 0

    # 新增闲置率统计
    machine_total_available_hour: Dict[MachineId, float] = field(default_factory=dict)
    machine_total_process_hour: Dict[MachineId, float] = field(default_factory=dict)
    # 新增逾期统计
    overdue_job_count = 0
    total_overdue_penalty = 0.0


@dataclass
class OperationSchedulingResult:
    """
    单个操作的调度结果
    所有解码算法的统一输出格式
    设计原则：字段完整，向后兼容，可直接序列化
    """
    # 算法内部操作唯一标识 | 关联state_manager中的原始元数据 | 例子：OperationId("OP_1001_01") (1001号工件的第1道工序)
    operation_id: OperationId = OperationId("")
    # 所属工件ID | 关联工件信息 + 保证工艺路线顺序 | 例子：JobId(1001) (生产订单号为1001的工件)
    job_id: JobId = JobId(-1)
    # 操作在工件中的工序序号 | 核心：确保工艺路线的先后顺序约束 | 例子：0 (第1道工序，从0开始计数)
    operation_index_in_job: int = -1
    # 分配的机器ID | -1表示未分配（正常流程中不会出现）| 例子：MachineId(1) (分配给1号车床)
    machine_id: MachineId = MachineId(-1)
    # 分配的工人ID | -1表示不需要工人 | 例子：WorkerId(101) (分配给101号车工)
    worker_id: WorkerId = WorkerId(-1)
    # 操作实际开始时间（相对小时，从系统当前时间开始计算）| 例子：0.0 (当前时间立即开始)
    start_time: float = 0.0
    # 操作实际结束时间（相对小时）| 例子：2.0 (2小时后完成)
    end_time: float = 0.0
    # 实际加工时间 = 基础加工时间 × 工人技能速度比 | 例子：2.0 (基础时间2.5小时，工人技能系数0.8)
    actual_processing_time: float = 0.0
    # 操作的工艺类型ID | 用于机器换型时间计算 | 例子：1 (车削工艺)
    technology_type: int = -1
    # 是否处于计划冻结区间 | 冻结区间内的操作不能被重调度修改 | 例子：True (在未来24小时冻结区间内)
    is_frozen: bool = False
    # 是否被用户手动锁定 | 算法不能修改手动锁定操作的资源分配 | 例子：False (未被锁定，算法可优化)
    is_manual_locked: bool = False
    # 原始业务元数据引用 | 核心桥梁：连接算法层和业务层 | 例子：包含{"business_op_no": "10", "op_name": "粗车外圆"}的对象
    operation_metadata: Optional[Any] = None