import numpy as np
from core.state_manager import ProductionStateManager
from data.test_dataset import build_test_production_data
from trigger.rolling_trigger import RollingScheduleTrigger
from core.data_structs import ManualLockAssign
from core.nsga2_operator import nsga2_rolling_schedule
import core.base_ga as base_ga
from visual.plot_gantt import plot_pareto_front, plot_machine_gantt, plot_worker_gantt, plot_operation_gantt
from utils.log_utils import get_logger
import config as cfg

# 全局日志初始化（仅在main.py执行一次）
logger = get_logger(__name__)


if __name__ == "__main__":
    # ============================================================
    # 【初始化滚动排程】加载test_data1.json数据
    # ============================================================
    np.random.seed(40)
    sm = ProductionStateManager()
    trigger = RollingScheduleTrigger(sm)



