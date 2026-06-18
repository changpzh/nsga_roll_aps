from typing import List, Tuple
from core.state_manager import ProductionStateManager
from core.nsga2_operator import nsga2_rolling_schedule
from utils.log_utils import get_logger

logger = get_logger(__name__)

class RollingScheduleTrigger:
    def __init__(self, state_manager: ProductionStateManager):
        self.sm = state_manager

    def refresh_production_status(self):
        self.sm.refresh_production_status()

    def get_current_reorder_jobs(self) -> List[int]:
        opt_ops = self.sm.get_optimizable_operation_ids()
        job_set = set()
        for op in opt_ops:
            job_set.add(self.sm.operation_id_to_job_id[op])
        return list(job_set)

    def trigger_by_event(self, event_type: str, insert_job_info: dict = None) -> Tuple[List[dict] | None, List[List[float]] | None, List[int] | None]:
        support_events = ["new_order", "machine_fault", "material_delay", "batch_finished"]
        if event_type not in support_events:
            return None, None, None
        self.refresh_production_status()

        if event_type == "new_order" and insert_job_info is not None:
            op_ids = self.sm.insert_new_order(
                job_id=insert_job_info["job_id"],
                priority=insert_job_info["priority"],
                warn_due=insert_job_info["warn_due"],
                due_contract_time=insert_job_info["due_contract_time"],
                base_weight=insert_job_info["base_weight"],
                op_info_list=insert_job_info["op_info_list"]
            )
            self.sm.refresh_optimizable_operation_pool()

        reorder_jobs = self.get_current_reorder_jobs()
        if len(reorder_jobs) == 0:
            return None, None, None
        pareto_results, result_fits, pareto_idx = nsga2_rolling_schedule(self.sm, reorder_jobs)
        return pareto_results, result_fits, pareto_idx

    def trigger_by_timer(self) -> Tuple[List[dict], List[List[float]], List[int]]:
        self.refresh_production_status()
        reorder_jobs = self.get_current_reorder_jobs()
        return nsga2_rolling_schedule(self.sm, reorder_jobs)