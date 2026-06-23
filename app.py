import numpy as np
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from core.state_manager import ProductionStateManager
from data.test_dataset import build_test_production_data, build_production_data_from_dict
from trigger.rolling_trigger import RollingScheduleTrigger
import core.base_ga as base_ga
import config as cfg
from utils.log_utils import get_logger
import json
import os

app = Flask(__name__)
logger = get_logger(__name__)

# ==================== 全局初始化 ====================
np.random.seed(40)
sm = ProductionStateManager()

all_job_op_map = build_test_production_data(sm, "test_data1.json")
sm.set_system_time(0.0)

weight = [0.30, 0.10, 0.20, 0.20, 0.10, 0.05, 0.05]

trigger = RollingScheduleTrigger(sm)


def fetch_latest_data():
    """从外部数据源获取最新数据"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "data", "latest_data.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


trigger.set_data_provider(fetch_latest_data)


# ==================== 定时调度器 ====================
scheduler = BackgroundScheduler()

daily_config = {
    "enabled": True,
    "hour": 6,
    "minute": 0,
    "advance_hours": 8,
}


def daily_schedule_job():
    """定时任务：每日滚动排程"""
    logger.info(f"每日滚动排程触发，当前系统时间: {sm.current_system_time:.1f}h")

    try:
        pareto_set, final_fits, pareto_idx = trigger.trigger_by_event(
            "daily_roll",
            {"advance_hours": daily_config["advance_hours"]}
        )

        if pareto_set and len(pareto_set) > 0:
            best_chrom, best_fit = base_ga.select_optimal_solution_by_weight(
                pareto_set, final_fits, pareto_idx, weight
            )
            _, schedule_detail = base_ga.decode_chromosome(best_chrom, sm)
            sm.cache_schedule_result(schedule_detail)

            frozen_count = sum(1 for s in schedule_detail if s["is_frozen"])
            logger.info(
                f"每日排程完成 | 总工序:{len(schedule_detail)} | "
                f"冻结:{frozen_count} | 重排:{len(schedule_detail) - frozen_count} | "
                f"逾期:{best_fit[0]:.0f}单 | 最大完工:{best_fit[2]:.1f}h"
            )
        else:
            logger.warning("每日排程无有效帕累托解")

    except Exception as e:
        logger.error(f"每日排程失败: {e}", exc_info=True)


def start_daily_scheduler():
    """启动每日定时排程"""
    if daily_config["enabled"]:
        scheduler.add_job(
            daily_schedule_job,
            CronTrigger(hour=daily_config["hour"], minute=daily_config["minute"]),
            id="daily_schedule",
            replace_existing=True
        )
        logger.info(f"每日定时排程已启动 | 时间:{daily_config['hour']:02d}:{daily_config['minute']:02d} | 推进:{daily_config['advance_hours']}h")
    else:
        logger.info("每日定时排程已禁用")


scheduler.start()
start_daily_scheduler()


# ==================== 首次排程 ====================
logger.info("执行首次全量排程...")
pareto_set, final_fits, pareto_idx_list = trigger.trigger_by_event(
    "full_reschedule", {"system_time": 0.0}
)

best_chrom, best_fit = base_ga.select_optimal_solution_by_weight(
    pareto_set, final_fits, pareto_idx_list, weight
)
_, schedule_detail = base_ga.decode_chromosome(best_chrom, sm)
sm.cache_schedule_result(schedule_detail)

frozen_count = sum(1 for s in schedule_detail if s["is_frozen"])
logger.info(
    f"首次全量排程完成 | 总工序:{len(schedule_detail)} | "
    f"冻结:{frozen_count} | 重排:{len(schedule_detail) - frozen_count} | "
    f"逾期:{best_fit[0]:.0f}单 | 最大完工:{best_fit[2]:.1f}h"
)


# ==================== API 接口 ====================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "current_system_time": sm.current_system_time,
        "frozen_boundary": sm.current_system_time + cfg.PLAN_FROZEN_HORIZON,
        "cached_operations": len(sm.last_schedule_result),
        "daily_scheduler": {
            "enabled": daily_config["enabled"],
            "hour": daily_config["hour"],
            "minute": daily_config["minute"],
            "advance_hours": daily_config["advance_hours"]
        }
    })


@app.route("/api/trigger", methods=["POST"])
def trigger_event():
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体不能为空"}), 400

    event_type = data.get("event_type")
    if event_type not in trigger.support_events:
        return jsonify({
            "error": f"不支持的事件类型: {event_type}",
            "supported_events": trigger.support_events
        }), 400

    event_data = data.get("event_data", {})
    logger.info(f"收到排程触发请求 | 事件类型:{event_type} | 系统时间:{sm.current_system_time:.1f}h")

    try:
        pareto_set, final_fits, pareto_idx = trigger.trigger_by_event(event_type, event_data)

        if pareto_set is None or len(pareto_set) == 0:
            logger.warning(f"事件{event_type}排程无有效解")
            return jsonify({"error": "排程无有效解"}), 500

        best_chrom, best_fit = base_ga.select_optimal_solution_by_weight(
            pareto_set, final_fits, pareto_idx, weight
        )
        _, schedule_detail = base_ga.decode_chromosome(best_chrom, sm)
        sm.cache_schedule_result(schedule_detail)

        frozen_count = sum(1 for s in schedule_detail if s["is_frozen"])
        logger.info(
            f"排程完成 | 事件:{event_type} | 总工序:{len(schedule_detail)} | "
            f"冻结:{frozen_count} | 重排:{len(schedule_detail) - frozen_count} | "
            f"逾期:{best_fit[0]:.0f}单 | 最大完工:{best_fit[2]:.1f}h"
        )

        result = {
            "event_type": event_type,
            "current_system_time": sm.current_system_time,
            "fitness": {
                "overdue_count": best_fit[0],
                "total_overdue_penalty": best_fit[1],
                "makespan": best_fit[2],
                "equipment_idle_rate": best_fit[3],
                "machine_unbalance": best_fit[4],
                "worker_unbalance": best_fit[5],
                "wip_cost": best_fit[6]
            },
            "schedule_summary": {
                "total_operations": len(schedule_detail),
                "frozen_operations": frozen_count,
                "replanned_operations": len(schedule_detail) - frozen_count
            },
            "schedule_detail": schedule_detail[:20]
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"排程失败 | 事件:{event_type} | 错误:{e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/schedule", methods=["GET"])
def get_schedule():
    page = request.args.get("page", 1, type=int)
    page_size = request.args.get("page_size", 50, type=int)
    start = (page - 1) * page_size
    end = start + page_size

    return jsonify({
        "current_system_time": sm.current_system_time,
        "frozen_boundary": sm.current_system_time + cfg.PLAN_FROZEN_HORIZON,
        "total_count": len(sm.last_schedule_result),
        "page": page,
        "page_size": page_size,
        "schedule": list(sm.last_schedule_result.values())[start:end]
    })


@app.route("/api/daily_config", methods=["GET", "POST"])
def daily_config_api():
    if request.method == "GET":
        return jsonify(daily_config)

    data = request.get_json()
    if not data:
        return jsonify({"error": "数据不能为空"}), 400

    old_config = daily_config.copy()

    if "enabled" in data:
        daily_config["enabled"] = data["enabled"]
    if "hour" in data:
        daily_config["hour"] = data["hour"]
    if "minute" in data:
        daily_config["minute"] = data["minute"]
    if "advance_hours" in data:
        daily_config["advance_hours"] = data["advance_hours"]

    scheduler.remove_job("daily_schedule")
    start_daily_scheduler()

    logger.info(f"每日定时配置已更新 | 旧:{old_config} | 新:{daily_config}")

    return jsonify({
        "status": "ok",
        "daily_config": daily_config
    })


@app.route("/api/trigger_daily_now", methods=["POST"])
def trigger_daily_now():
    logger.info("手动触发每日排程")
    daily_schedule_job()
    return jsonify({
        "status": "ok",
        "current_system_time": sm.current_system_time
    })


@app.route("/api/locks", methods=["GET"])
def get_locks():
    return jsonify({"locks": sm.export_all_manual_lock()})


@app.route("/api/advance_time", methods=["POST"])
def advance_time():
    data = request.get_json()
    hours = data.get("hours", 0)
    if hours <= 0:
        return jsonify({"error": "hours 必须大于0"}), 400

    old_time = sm.current_system_time
    sm.advance_system_time(hours)
    logger.info(f"系统时间推进 | {old_time:.1f}h → {sm.current_system_time:.1f}h | 推进:{hours}h")

    return jsonify({
        "current_system_time": sm.current_system_time,
        "frozen_boundary": sm.current_system_time + cfg.PLAN_FROZEN_HORIZON
    })


@app.route("/api/sync_data", methods=["POST"])
def sync_data():
    latest_data = request.get_json()
    if not latest_data:
        return jsonify({"error": "数据不能为空"}), 400

    build_production_data_from_dict(sm, latest_data)
    logger.info(f"数据同步完成 | 订单:{len(sm.job_meta_dict)} | 工序:{len(sm.op_meta_dict)}")

    return jsonify({
        "status": "ok",
        "total_orders": len(sm.job_meta_dict),
        "total_operations": len(sm.op_meta_dict)
    })


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("NSGA-II 滚动排程服务启动")
    logger.info(f"API 地址: http://0.0.0.0:5000")
    logger.info(f"支持事件: {trigger.support_events}")
    logger.info(f"每日定时排程: {daily_config['hour']:02d}:{daily_config['minute']:02d}")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)