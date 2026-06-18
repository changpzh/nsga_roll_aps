"""
排程系统数据模型层
==================
本层是整个系统的通用语言，所有模块都依赖这里定义的数据结构。
设计原则：
1. 绝对纯净：只包含数据类、类型别名和枚举，无任何业务逻辑
2. 无外部依赖：只能导入Python标准库，不能导入core、utils等上层模块
3. 向后兼容：新增字段必须提供默认值，不能破坏现有代码

对外导出的所有数据结构都可以直接通过 `from models import X` 导入
"""

# 先导入基础类型（被其他数据结构依赖）
from .base_types import (
    MachineId,
    WorkerId,
    JobId,
    OperationId
)

# 再导入核心状态和结果数据结构
from .scheduling_state import (
    SchedulingTrackers,
    OperationSchedulingResult
)

# 明确导出的公共API（只有这里列出的才是对外公开的接口）
__all__ = [
    # 基础类型别名
    "MachineId",
    "WorkerId",
    "JobId",
    "OperationId",

    # 核心数据结构
    "SchedulingTrackers",
    "OperationSchedulingResult",
]