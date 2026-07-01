# config/settings.py

# 屏蔽调试matplotlib兼容报错
import os
os.environ["PYDEVD_DISABLE_MATPLOTLIB_SUPPORT"] = "1"

import matplotlib.pyplot as plt
from datetime import date
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


# ===================== 绘图全局中文配置 =====================
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 工序状态映射 =====================

OP_STATUS_MAP = {
    0: "⏳待排",
    1: "🔄运行中",
    2: "✅已完工",
    -1: "❓未知",
}

# ===================== 订单优先级逾期权重 =====================
JOB_PRIORITY_WEIGHT = {
    "low": 1.0,
    "normal": 2.0,
    "high": 4.0,
    "urgent": 7.0
}

# ===================== 工序状态枚举 =====================
OP_STATUS_FINISHED = 2
OP_STATUS_RUNNING = 1
OP_STATUS_OPTIMIZABLE = 0
OP_STATUS_MATERIAL_DELAY = 3

# ===================== 订单优先级枚举 =====================
JOB_PRIORITY_LOW = 1
JOB_PRIORITY_NORMAL = 2
JOB_PRIORITY_HIGH = 3
JOB_PRIORITY_URGENT = 4


# ===================== 业务成本系数 =====================
WORKER_SWITCH_COST = 2.0
WIP_WEIGHT_COEFFICIENT = 1

# ===================== NSGA 静态超参数 =====================
# 运行前请确认 POPULATION_SIZE ≥ 参考点数量（divisions=3 时 7 目标为 84），并取 4 的倍数，以保证小生境保留过程的稳定性。
POPULATION_SIZE = 88
MAX_GENERATION = 100
ELITE_RATE = 0.1
MAX_FRONT_NUM = 30
MUTATION_RATE = 0.10
CROSSOVER_RATE = 0.1

# ===================== 滚动调度配比参数 =====================
ROLLING_HISTORY_SEED_RATIO = 0.5
ROLLING_HEURISTIC_SEED_RATIO = 0.2
ROLLING_PERTURB_RATE = 0.15

# ===================== 惩罚系数 =====================
OVERLOAD_PENALTY_COEFFICIENT = 0.8
WARN_OVERDUE_COEFFICIENT = 0.12
CONTRACT_OVERDUE_COEFFICIENT = 0.25
DELIVERY_OVERDUE_COEFFICIENT = 0.025
PLAN_FROZEN_HORIZON_HOURS = 8.0 # 表示两天


# ===================== 日志全局配置（新增） =====================
LOG_CONFIG = {
    # 全局总日志级别：DEBUG/INFO/WARNING/ERROR/CRITICAL
    "global_level": "INFO",
    # 控制台打印级别
    "console_level": "INFO",
    # 文件写入级别（比控制台细，存全量调试信息）
    "file_level": "DEBUG",
    # 日志文件存储路径
    "log_file_path": PROJECT_ROOT / "logs" / "nsga2_roll_aps.log",
    # 单日志文件最大容量 10MB
    "max_file_size": 10 * 1024 * 1024,
    # 滚动备份日志数量
    "backup_count": 5,
    "encoding": "utf-8",
    # 日志格式模板
    "formats": {
        "console": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        "file": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d [PID:%(process)d] - %(message)s"
    },
    "date_format": "%Y-%m-%d %H:%M:%S"
}

# ===================== 工作日历班次数据源配置 =====================
# 班次配置加载方式：'json_file' 或 'database'
SHIFT_DATA_SOURCE = 'json_file'  # 可选 'json_file' | 'database'

# JSON 文件路径（当 SHIFT_DATA_SOURCE = 'json_file' 时生效）
SHIFT_CONFIG_FILE = PROJECT_ROOT / "config" / "shift_config.json"

# 数据库表名（当 SHIFT_DATA_SOURCE = 'database' 时生效）
SHIFT_CONFIG_TABLE = "shift_config"

# 基准日期（项目投产首日，启动时固定，滚动排程中不可更改）
BASE_DATE = date(2026, 6, 22)  # 根据实际投产日期修改

# 每日班次切换时间点
DAY_SHIFT_THRESHOLD = 8.0