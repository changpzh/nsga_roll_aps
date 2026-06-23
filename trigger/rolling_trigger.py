from typing import List, Dict, Any
from core.state_manager import ProductionStateManager
from core.nsga2_operator import nsga2_rolling_schedule
from utils.log_utils import get_logger

logger = get_logger(__name__)


class RollingScheduleTrigger:
    """滚动排程触发器，根据外部事件触发重排"""

    support_events = [
        "new_order", "machine_fault", "material_delay",
        "batch_finished", "full_reschedule", "daily_roll"
    ]

    def __init__(self, state_manager: ProductionStateManager):
        self.sm = state_manager
        self._data_provider = None

    def set_data_provider(self, provider):
        """注入最新数据获取函数"""
        self._data_provider = provider

    def trigger_by_event(self, event_type: str, event_data: Dict[str, Any]):
        """根据事件类型触发滚动排程"""
        logger.info(f"触发排程 | 事件:{event_type} | 系统时间:{self.sm.current_system_time:.1f}h")

        # 每次排程前刷新最新数据
        self._refresh_latest_data()

        if event_type == "new_order":
            return self._handle_new_order(event_data)
        elif event_type == "machine_fault":
            return self._handle_machine_fault(event_data)
        elif event_type == "material_delay":
            return self._handle_material_delay(event_data)
        elif event_type == "batch_finished":
            return self._handle_batch_finished(event_data)
        elif event_type == "full_reschedule":
            return self._handle_full_reschedule(event_data)
        elif event_type == "daily_roll":
            return self._handle_daily_roll(event_data)
        else:
            raise ValueError(f"不支持的事件类型: {event_type}")

    def _refresh_latest_data(self):
        """从外部数据源拉取最新数据并刷新到 state_manager"""
        if self._data_provider is None:
            return

        try:
            latest_data = self._data_provider()
            if latest_data is None:
                return

            from data.test_dataset import build_production_data_from_dict
            build_production_data_from_dict(self.sm, latest_data)
            logger.debug(f"最新数据刷新完成 | 订单:{len(self.sm.job_meta_dict)} | 工序:{len(self.sm.op_meta_dict)}")
        except Exception as e:
            logger.warning(f"刷新最新数据失败: {e}")

    def _handle_full_reschedule(self, event_data: dict):
        """全量重排"""
        system_time = event_data.get("system_time", self.sm.current_system_time)
        self.sm.last_schedule_result = {}
        self.sm.last_pareto_solutions = []
        self.sm.set_system_time(system_time)

        logger.info(f"全量重排 | 系统时间重置:{system_time:.1f}h | 历史缓存已清空")

        all_job_ids = list(self.sm.job_meta_dict.keys())
        return nsga2_rolling_schedule(self.sm, all_job_ids)

    def _handle_daily_roll(self, event_data: dict):
        """每日定时滚动排程"""
        advance_hours = event_data.get("advance_hours", 0)
        if advance_hours > 0:
            old_time = self.sm.current_system_time
            self.sm.advance_system_time(advance_hours)
            logger.info(f"每日滚动 | 时间推进:{advance_hours}h | {old_time:.1f}h → {self.sm.current_system_time:.1f}h")

        all_job_ids = list(self.sm.job_meta_dict.keys())
        return nsga2_rolling_schedule(self.sm, all_job_ids)

    def _handle_new_order(self, event_data: dict):
        """处理新订单插单"""
        job_id = event_data["job_id"]
        priority = event_data["priority"]
        warn_due = event_data.get("warn_due", event_data.get("due_warn_time", 0))
        due_contract_time = event_data.get("due_contract_time", 0)
        base_weight = event_data.get("base_weight", 1.0)
        op_info_list = event_data["op_info_list"]

        new_op_ids = self.sm.insert_new_order(
            job_id=job_id,
            priority=priority,
            warn_due=warn_due,
            due_contract_time=due_contract_time,
            base_weight=base_weight,
            op_info_list=op_info_list
        )

        logger.info(f"新订单插单 | job_id:{job_id} | 优先级:{priority} | 工序数:{len(new_op_ids)}")

        all_job_ids = list(self.sm.job_meta_dict.keys())
        return nsga2_rolling_schedule(self.sm, all_job_ids)

    def _handle_machine_fault(self, event_data: dict):
        """处理设备故障"""
        machine_id = event_data["machine_id"]
        fault_hours = event_data.get("fault_hours", 0)

        if machine_id in self.sm.machine_meta_dict:
            self.sm.machine_meta_dict[machine_id].available = False
            logger.warning(f"设备故障 | machine_id:{machine_id} | 预计修复:{fault_hours}h")

        all_job_ids = list(self.sm.job_meta_dict.keys())
        return nsga2_rolling_schedule(self.sm, all_job_ids)

    def _handle_material_delay(self, event_data: dict):
        """处理物料延迟"""
        op_id = event_data.get("op_id")
        delay_hours = event_data.get("delay_hours", 0)

        if op_id and op_id in self.sm.op_meta_dict:
            old_ready = self.sm.op_meta_dict[op_id].material_ready_time
            self.sm.op_meta_dict[op_id].material_ready_time += delay_hours
            logger.warning(
                f"物料延迟 | op_id:{op_id} | "
                f"就绪时间:{old_ready:.1f}h → {self.sm.op_meta_dict[op_id].material_ready_time:.1f}h"
            )

        all_job_ids = list(self.sm.job_meta_dict.keys())
        return nsga2_rolling_schedule(self.sm, all_job_ids)

    def _handle_batch_finished(self, event_data: dict):
        """处理批次完工"""
        advance_hours = event_data.get("advance_hours", 0)
        if advance_hours > 0:
            self.sm.advance_system_time(advance_hours)

        finished_ops = event_data.get("finished_ops", [])
        for op_id in finished_ops:
            if op_id in self.sm.operation_status_dict:
                self.sm.operation_status_dict[op_id] = 2

        if finished_ops:
            logger.info(f"批次完工 | 推进:{advance_hours}h | 完工工序:{len(finished_ops)}个")

        all_job_ids = list(self.sm.job_meta_dict.keys())
        return nsga2_rolling_schedule(self.sm, all_job_ids)