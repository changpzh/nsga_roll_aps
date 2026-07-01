# visual/plot_gantt.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from typing import Dict, List
import warnings
from datetime import datetime
from core.state_manager import ProductionStateManager


def plot_pareto_front(pareto_fits: List[List[float]]):
    makespan_list = [fit[2] * 24 for fit in pareto_fits]
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
            start = row["start_time"]  # datetime
            end = row["end_time"]  # datetime
            duration = end - start  # timedelta
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

            # 基础标签
            if 'business_op_no' in row and 'op_name' in row:
                label_text = f"J{row['job_id']}-{row['business_op_no']}\n{row['op_name']}"
            else:
                label_text = f"J{row['job_id']}-OP{row['op_id']}"

            # 批次信息（若拆分）
            if row.get('batch_total', 1) > 1:
                label_text += f"\n[批{row['batch_index'] + 1}/{row['batch_total']}]"

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
            start = row["start_time"]  # datetime
            end = row["end_time"]  # datetime
            duration = end - start  # timedelta
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

            # 基础标签
            if 'business_op_no' in row:
                label_text = f"J{row['job_id']}-{row['business_op_no']}\nM{row['machine_id']}"
            else:
                label_text = f"J{row['job_id']}\nM{row['machine_id']}"

            # 批次信息
            if row.get('batch_total', 1) > 1:
                label_text += f"\n[批{row['batch_index'] + 1}/{row['batch_total']}]"

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

    # 按订单和工序号排序（包含批次）
    df["biz_no_int"] = df["business_op_no"].astype(int)
    df = df.sort_values(["job_id", "biz_no_int", "batch_index"], ascending=[True, True, True])

    # 每一行是一个独立的子批次，直接遍历所有行
    all_rows = list(df.iterrows())
    colors = plt.cm.tab20.colors
    job_color_map = {}
    unique_jobs = df["job_id"].unique()
    for idx, jid in enumerate(unique_jobs):
        job_color_map[jid] = colors[idx % len(colors)]

    canvas_height = max(10, len(all_rows) * 0.6)
    total_duration = (df["end_time"].max() - df["start_time"].min()).total_seconds() / 3600
    canvas_width = max(20, total_duration / 8)
    fig, ax = plt.subplots(figsize=(canvas_width, canvas_height))
    y_tick_labels = []

    for y_idx, (_, op_data) in enumerate(all_rows):
        start = op_data["start_time"]
        end = op_data["end_time"]
        duration = end - start
        job_id = op_data["job_id"]
        mid = op_data["machine_id"]
        wid = op_data["worker_id"]
        is_frozen = op_data["is_frozen"]
        is_manual_locked = op_data["is_manual_locked"]
        rect_color = job_color_map[job_id]

        # y轴标签：订单-工序号，若有批次则追加批次信息
        if 'business_op_no' in op_data:
            tick_text = f"订单{job_id}-{op_data['business_op_no']}"
        else:
            tick_text = f"订单{job_id} · 第{op_data['job_op_index'] + 1}道"
        if op_data.get('batch_total', 1) > 1:
            tick_text += f" [批{op_data['batch_index'] + 1}/{op_data['batch_total']}]"
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

    ax.set_yticks(range(len(all_rows)))
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


def print_topsis_sorted_pareto_table(
    sorted_pareto_list: List[dict],
    sorted_fit_list: List[List[float]]
):
    """
    【窄屏紧凑版】打印TOPSIS从优到劣排序帕累托解集，小终端窗口也不会横向溢出、排版对齐
    :param sorted_pareto_list: 按名次升序排列的帕累托个体列表
    :param sorted_fit_list: 对应排序后的适应度二维列表
    """
    # 极致精简表头，压缩总宽度
    target_names = [
        "逾期订单数",
        "逾期惩罚成本",
        "最大完工时间(天)",
        "设备闲置率(%)",
        "设备负荷不均",
        "人员负荷不均",
        "在制品等待时长(天)"
    ]
    headers = ["综合名次"] + target_names

    # 窄屏专属紧凑列宽，刚好容纳数据，无多余冗余
    col_widths = [
        7,    # 名次
        12,   # 逾期订单数
        24,   # 惩罚成本（数值最长，适度加宽）
        15,   # 最大完工时间
        14,   # 设备闲置率
        16,   # 设备负荷不均
        16,   # 人员负荷不均
        18    # 在制品等待时长
    ]
    total_width = sum(col_widths)

    # 组装表格数据，统一保留4位小数
    table_data = []
    for rank, fit_vec in enumerate(sorted_fit_list, start=1):
        row = [rank] + [round(val, 4) for val in fit_vec]
        table_data.append(row)

    print(f"\n帕累托最优解集数量：{len(sorted_pareto_list)}")
    print("-" * total_width)
    print(f"{'【帕累托解集｜TOPSIS从优到劣排序】':^{total_width}}")
    print("-" * total_width)

    # 表头行：居中排版
    header_line = ""
    for w, name in zip(col_widths, headers):
        header_line += f"{name:^{w}}"
    print(header_line)
    print("-" * total_width)

    # 数据行：名次居中，所有数值右对齐，小数点垂直对齐
    for row in table_data:
        row_line = ""
        for col_idx, (width, cell_val) in enumerate(zip(col_widths, row)):
            if col_idx == 0:
                row_line += f"{cell_val:^{width}}"
            else:
                row_line += f"{cell_val:>{width}.4f}"
        print(row_line)

    print("-" * total_width + "\n")

