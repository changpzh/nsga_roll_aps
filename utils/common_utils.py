from typing import List
from models import MachineId, WorkerId, SchedulingTrackers


def select_best_machine(available_machines: List[MachineId], trackers: SchedulingTrackers) -> MachineId:
    """
    选择最优机器：当前最早可用的机器（工业界标准策略）
    统一所有机器自动分配逻辑，避免重复代码和不一致
    """
    return min(available_machines, key=lambda m: trackers.machine_last_end_time_dict[m])


def select_best_worker(available_workers: List[WorkerId], trackers: SchedulingTrackers) -> WorkerId:
    """
    选择最优工人：当前总负载最少的工人（工业界最常用的负载均衡策略）
    优势：任务时长差异大时负载均衡效果最好，计算简单高效
    """
    return min(available_workers, key=lambda w: sum(trackers.worker_workloads_dict.get(w, [])))