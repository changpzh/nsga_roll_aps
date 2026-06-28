# models/base_types.py
"""
排程系统基础类型定义
所有模块统一使用这里的类型别名，从根本上避免ID混淆
"""
from typing import NewType

# 资源ID类型别名（IDE和静态检查工具会自动拦截类型不匹配错误）
MachineId = NewType("MachineId", int)
WorkerId = NewType("WorkerId", int)
JobId = NewType("JobId", str)
OperationId = NewType("OperationId", int)