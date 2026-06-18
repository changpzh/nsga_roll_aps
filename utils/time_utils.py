from datetime import date, datetime
from typing import Tuple


def hour_to_hhmm(hour: float) -> str:
    """浮点数小时转HH:MM格式，用于甘特图显示"""
    h = int(hour)
    m = int((hour - h) * 60)
    return f"{h:02d}:{m:02d}"


def hhmm_to_hour(hhmm: str) -> float:
    """HH:MM字符串转浮点数小时"""
    h, m = map(int, hhmm.split(":"))
    return h + m / 60


def calculate_work_hours(start_hour: float, end_hour: float) -> float:
    """计算同一天内的有效工时（默认8:00-18:00）"""
    default_start = 8.0
    default_end = 18.0
    actual_start = max(start_hour, default_start)
    actual_end = min(end_hour, default_end)
    return max(0.0, actual_end - actual_start)


def date_to_str(dt: date) -> str:
    """date对象转标准日期字符串"""
    return dt.strftime("%Y-%m-%d")