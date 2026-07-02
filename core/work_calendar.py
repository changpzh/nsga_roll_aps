from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

from core.data_structs import Shift, ShiftSegment
from utils.log_utils import get_logger

logger = get_logger(__name__)


class ShiftCalendar:
    """
    基于绝对 datetime 的班次日历。
    每日最多 N 个班次，同一时刻不重叠。
    """

    def __init__(self,
                 base_date: date,
                 weekly_shifts: Dict[int, List[Shift]],
                 special_shifts: Optional[Dict[date, List[Shift]]] = None):
        self.base_date = base_date
        self._weekly = weekly_shifts
        self._special = special_shifts or {}
        self._base_zero = self._calc_base_zero()

    @property
    def special_shifts(self) -> Dict[date, List[Shift]]:
        return self._special

    def clear_cache(self) -> None:
        """清空所有日期区间缓存，班次配置变更后必须调用"""
        self._day_intervals.cache_clear()
        logger.info("班次日历缓存已清空")

    def _get_shifts(self, dt: date) -> List[Shift]:
        if dt in self._special:
            return self._special[dt]
        return self._weekly.get(dt.weekday(), [])

    def _get_segments(self, dt: date) -> List[ShiftSegment]:
        shifts = self._get_shifts(dt)
        all_segs = []
        for shift in shifts:
            all_segs.extend(shift.segments)
        all_segs.sort(key=lambda s: (s.start.hour * 60 + s.start.minute))
        return all_segs

    def _calc_base_zero(self) -> datetime:
        segs = self._get_segments(self.base_date)
        if not segs:
            return datetime.combine(self.base_date, time(0, 0))
        return datetime.combine(self.base_date, segs[0].start)

    @property
    def base_zero(self) -> datetime:
        return self._base_zero

    @lru_cache(maxsize=512)
    def _day_intervals(self, dt: date) -> Tuple[Tuple[datetime, datetime], ...]:
        segs = self._get_segments(dt)
        result = []
        for seg in segs:
            s_dt = datetime.combine(dt, seg.start)
            e_dt = datetime.combine(dt, seg.end)
            if seg.end <= seg.start:
                e_dt = datetime.combine(dt + timedelta(days=1), seg.end)
            result.append((s_dt, e_dt))
        return tuple(result)

    def is_working(self, dt: datetime) -> bool:
        for s, e in self._day_intervals(dt.date()):
            if s <= dt < e:
                return True
        return False

    def next_work_start(self, dt: datetime) -> Optional[datetime]:
        cur_date = dt.date()
        for _ in range(366):
            intervals = self._day_intervals(cur_date)
            for s, e in intervals:
                if dt < s:
                    return s
                if s <= dt < e:
                    return dt
            cur_date += timedelta(days=1)
            dt = datetime.combine(cur_date, time(0, 0))
        return None

    def add_work_hours(self, start: datetime, hours: float) -> datetime:
        if hours <= 0:
            return start
        cur = self.next_work_start(start)
        if cur is None:
            return start
        remain_sec = hours * 3600.0
        cur_date = cur.date()
        for _ in range(366):
            intervals = self._day_intervals(cur_date)
            if intervals and cur < intervals[0][0]:
                cur = intervals[0][0]
            for s, e in intervals:
                if cur >= e:
                    continue
                if cur < s:
                    cur = s
                avail = (e - cur).total_seconds()
                if remain_sec <= avail:
                    return cur + timedelta(seconds=remain_sec)
                remain_sec -= avail
                cur = e
            cur_date += timedelta(days=1)
            cur = datetime.combine(cur_date, time(0, 0))
        return cur

    def work_hours_between(self, start: datetime, end: datetime) -> float:
        if end <= start:
            return 0.0
        total_sec = 0.0
        cur = start
        cur_date = cur.date()
        while cur < end:
            intervals = self._day_intervals(cur_date)
            found = False
            for s, e in intervals:
                if e <= cur:
                    continue
                found = True
                overlap_start = max(cur, s)
                overlap_end = min(e, end)
                if overlap_end > overlap_start:
                    total_sec += (overlap_end - overlap_start).total_seconds()
                cur = e
                if cur >= end:
                    break
            if not found:
                cur_date += timedelta(days=1)
                cur = datetime.combine(cur_date, time(0, 0))
        return round(total_sec / 3600.0, 2)

    def get_work_hours_for_date(self, dt: date) -> float:
        segs = self._get_segments(dt)
        total = 0.0
        for seg in segs:
            s_min = seg.start.hour * 60 + seg.start.minute
            e_min = seg.end.hour * 60 + seg.end.minute
            if e_min <= s_min:
                e_min += 24 * 60
            total += (e_min - s_min) / 60.0
        return total

    def is_workday(self, dt: date) -> bool:
        return len(self._get_shifts(dt)) > 0