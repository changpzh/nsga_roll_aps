import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from typing import Dict, List
import warnings
from datetime import datetime
from core.state_manager import ProductionStateManager


def plot_pareto_front(pareto_fits: List[List[float]]):
    makespan_list = [fit[2] for fit in pareto_fits]
    overdue_list = [fit[1] for fit in pareto_fits]
    plt.figure(figsize=(10, 6))
    plt.scatter(makespan_list, overdue_list, c="#2E86AB", s=45, alpha=0.7, label="帕累托最优解")
    plt.xlabel("最大完工时间（小时）")
    plt.ylabel("订单加权逾期惩罚")
    plt.title("NSGA-II 帕累托最优前沿｜7目标生产版")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_machine_gantt(schedule_detail: List[dict], state_manager: ProductionStateManager):
    df = pd.DataFrame(schedule_detail)

    current_time = state_manager.current_system_time  # datetime

    # 过滤无设备工序和已完工工序
    df = df[(df["machine_id"] != -1) & (df["start_time"] >= current_time)]

    if df.empty:
        print("没有可显示的机床排程数据")
        return

    df = df.sort_values(["machine_id", "start_time"], ascending=[True, True])
    unique_machines = sorted(df["machine_id"].unique())
    colors = plt.cm.tab20.colors
    job_color_map = {}
    unique_jobs = df["job_id"].unique()
    for idx, jid in enumerate(unique_jobs):
        job_color_map[jid] = colors[idx % len(colors)]

    canvas_height = max(8, len(unique_machines) * 1.2)
    total_duration = (df["end_time"].max() - df["start_time"].min()).total_seconds() / 3600  # timedelta → 小时
    canvas_width = max(20, total_duration / 8)
    fig, ax = plt.subplots(figsize=(canvas_width, canvas_height))
    y_tick_labels = []

    for y_idx, mid in enumerate(unique_machines):
        machine_data = df[df["machine_id"] == mid]
        y_tick_labels.append(f"机床 {mid}")

        for _, row in machine_data.iterrows():
            job_id = row["job_id"]
            start = row["start_time"]      # datetime
            end = row["end_time"]          # datetime
            duration = end - start         # timedelta
            is_frozen = row["is_frozen"]
            is_manual_locked = row["is_manual_locked"]
            rect_color = job_color_map[job_id]

            if is_manual_locked:
                hatch_style = "\\\\"
            elif is_frozen:
                hatch_style = "///"
            else:
                hatch_style = ""

            ax.barh(y=y_idx, width=duration, left=start,
                    color=rect_color, edgecolor="black", hatch=hatch_style, height=0.7)
            text_x = start + duration / 2

            if 'business_op_no' in row and 'op_name' in row:
                label_text = f"J{row['job_id']}-{row['business_op_no']}\n{row['op_name']}"
            else:
                label_text = f"J{row['job_id']}-OP{row['op_id']}"
            ax.text(x=text_x, y=y_idx, s=label_text, ha="center", va="center", fontsize=8, color="black",
                    fontweight="bold")

            start_label = start.strftime("%m-%d %H:%M")
            end_label = end.strftime("%m-%d %H:%M")
            ax.text(start, y_idx + 0.3, start_label, ha="right", va="bottom", fontsize=7)
            ax.text(end, y_idx - 0.3, end_label, ha="left", va="top", fontsize=7)

    ax.set_yticks(range(len(unique_machines)))
    ax.set_yticklabels(y_tick_labels, fontsize=10)

    # x轴从当前系统时间开始
    ax.set_xlim(left=current_time)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    if total_duration <= 72:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    elif total_duration <= 168:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45, ha="right")
    ax.set_xlabel("真实时间轴", fontsize=12)
    ax.set_title(
        f"机床排产甘特图（基准日期：{state_manager.work_calendar.base_date.strftime('%Y-%m-%d')}）\n\\\\=人工锁定 ///=计划冻结",
        fontsize=14)

    ax.set_ylabel("机床编号", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.18)

    warnings.filterwarnings("ignore", category=UserWarning)
    plt.show()


def plot_worker_gantt(schedule_detail: List[dict], state_manager: ProductionStateManager):
    df = pd.DataFrame(schedule_detail)

    current_time = state_manager.current_system_time  # datetime

    df = df[(df["worker_id"] != -1) & (df["start_time"] >= current_time)]

    if df.empty:
        print("没有可显示的工人排程数据")
        return

    df = df.sort_values(["worker_id", "start_time"], ascending=[True, True])
    unique_workers = sorted(df["worker_id"].unique())
    colors = plt.cm.tab20.colors
    job_color_map = {}
    unique_jobs = df["job_id"].unique()
    for idx, jid in enumerate(unique_jobs):
        job_color_map[jid] = colors[idx % len(colors)]

    canvas_height = max(8, len(unique_workers) * 1.2)
    total_duration = (df["end_time"].max() - df["start_time"].min()).total_seconds() / 3600
    canvas_width = max(20, total_duration / 8)
    fig, ax = plt.subplots(figsize=(canvas_width, canvas_height))
    y_tick_labels = []

    for y_idx, wid in enumerate(unique_workers):
        worker_data = df[df["worker_id"] == wid]
        y_tick_labels.append(f"工人 {wid}")

        for _, row in worker_data.iterrows():
            job_id = row["job_id"]
            start = row["start_time"]      # datetime
            end = row["end_time"]          # datetime
            duration = end - start         # timedelta
            is_frozen = row["is_frozen"]
            is_manual_locked = row["is_manual_locked"]
            rect_color = job_color_map[job_id]

            if is_manual_locked:
                hatch_style = "\\\\"
            elif is_frozen:
                hatch_style = "///"
            else:
                hatch_style = ""

            ax.barh(y=y_idx, width=duration, left=start,
                    color=rect_color, edgecolor="black", hatch=hatch_style, height=0.7)
            text_x = start + duration / 2

            if 'business_op_no' in row:
                label_text = f"J{row['job_id']}-{row['business_op_no']}\nM{row['machine_id']}"
            else:
                label_text = f"J{row['job_id']}\nM{row['machine_id']}"
            ax.text(x=text_x, y=y_idx, s=label_text, ha="center", va="center", fontsize=8, color="black",
                    fontweight="bold")

            start_label = start.strftime("%m-%d %H:%M")
            end_label = end.strftime("%m-%d %H:%M")
            ax.text(start, y_idx + 0.3, start_label, ha="right", va="bottom", fontsize=7)
            ax.text(end, y_idx - 0.3, end_label, ha="left", va="top", fontsize=7)

    ax.set_yticks(range(len(unique_workers)))
    ax.set_yticklabels(y_tick_labels, fontsize=10)

    ax.set_xlim(left=current_time)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    if total_duration <= 72:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    elif total_duration <= 168:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45, ha="right")
    ax.set_xlabel("真实时间轴", fontsize=12)
    ax.set_title(
        f"工人排产甘特图（基准日期：{state_manager.work_calendar.base_date.strftime('%Y-%m-%d')}）\n\\\\=人工锁定 ///=计划冻结",
        fontsize=14)

    ax.set_ylabel("工人编号", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.18)

    warnings.filterwarnings("ignore", category=UserWarning)
    plt.show()


def plot_operation_gantt(schedule_detail: List[dict], state_manager: ProductionStateManager):
    df = pd.DataFrame(schedule_detail)

    current_time = state_manager.current_system_time  # datetime

    df = df[df["start_time"] >= current_time]

    if df.empty:
        print("没有可显示的工序数据")
        return
    df["biz_no_int"] = df["business_op_no"].astype(int)
    df = df.sort_values(["job_id", "biz_no_int"], ascending=[True, True])
    unique_ops = df["op_id"].tolist()
    colors = plt.cm.tab20.colors
    job_color_map = {}
    unique_jobs = df["job_id"].unique()
    for idx, jid in enumerate(unique_jobs):
        job_color_map[jid] = colors[idx % len(colors)]

    canvas_height = max(10, len(unique_ops) * 0.6)
    total_duration = (df["end_time"].max() - df["start_time"].min()).total_seconds() / 3600
    canvas_width = max(20, total_duration / 8)
    fig, ax = plt.subplots(figsize=(canvas_width, canvas_height))
    y_tick_labels = []

    for y_idx, op_id in enumerate(unique_ops):
        op_data = df[df["op_id"] == op_id].iloc[0]
        start = op_data["start_time"]      # datetime
        end = op_data["end_time"]          # datetime
        duration = end - start             # timedelta
        job_id = op_data["job_id"]
        mid = op_data["machine_id"]
        wid = op_data["worker_id"]
        job_inner_idx = op_data["job_op_index"]
        is_frozen = op_data["is_frozen"]
        is_manual_locked = op_data["is_manual_locked"]
        rect_color = job_color_map[job_id]

        if 'business_op_no' in op_data and 'op_name' in op_data:
            tick_text = f"订单{job_id}-{op_data['business_op_no']}"
        else:
            tick_text = f"订单{job_id} · 第{job_inner_idx + 1}道"
        y_tick_labels.append(tick_text)

        if is_manual_locked:
            hatch_style = "\\\\"
        elif is_frozen:
            hatch_style = "///"
        else:
            hatch_style = ""

        ax.barh(y=y_idx, width=duration, left=start,
                color=rect_color, edgecolor="black", hatch=hatch_style, height=0.55)
        text_x = start + duration / 2

        machine_label = f"机床{mid}" if mid != -1 else "无设备"
        worker_label = f"工人{wid}" if wid != -1 else "无工人"
        label_text = f"{machine_label}\n{worker_label}"
        ax.text(x=text_x, y=y_idx, s=label_text, ha="center", va="center", fontsize=7, color="black", fontweight="bold")

        start_label = start.strftime("%m-%d %H:%M")
        end_label = end.strftime("%m-%d %H:%M")
        ax.text(start, y_idx + 0.25, start_label, ha="right", va="bottom", fontsize=7)
        ax.text(end, y_idx - 0.25, end_label, ha="left", va="top", fontsize=7)

    ax.set_yticks(range(len(unique_ops)))
    ax.set_yticklabels(y_tick_labels, fontsize=8.5)

    ax.set_xlim(left=current_time)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    if total_duration <= 72:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    elif total_duration <= 168:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45, ha="right")
    ax.set_xlabel("真实时间轴", fontsize=12)
    ax.set_title(
        f"全工序时间轴甘特图（基准日期：{state_manager.work_calendar.base_date.strftime('%Y-%m-%d')}）\n\\\\=人工锁定 ///=计划冻结",
        fontsize=14)

    ax.set_ylabel("订单及内部工序", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.18)

    warnings.filterwarnings("ignore", category=UserWarning)
    plt.show()