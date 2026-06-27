from datetime import date, timedelta, datetime, time
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from functools import lru_cache

from core.data_structs import ShiftSegment
from utils.log_utils import get_logger

logger = get_logger(__name__)


class WorkCalendar:
    def __init__(self,
                 base_date: date,
                 weekly_rules: Dict[int, List[ShiftSegment]],
                 special_rules: Optional[Dict[date, List[ShiftSegment]]] = None):
        """
        数据驱动的日历初始化
        :param base_date: 基准日期（**相对工时零点 = 该日第一个白班开始时刻**）
        :param weekly_rules: 每周固定班次，key=0~6(周一~周日), value=该天班次列表
        :param special_rules: 特殊日期班次（如节假日），key=日期, value=班次列表（空列表表示全天休息）
        """
        self.base_date = base_date
        self._segments_cache: Dict[date, List[Tuple[str, float, float]]] = {}

        # ========= 第一步：先初始化两个核心班次容器属性 =========
        self._weekly_natural_segments: Dict[int, List[Tuple[str, float, float]]] = {}
        self._special_natural_segments: Dict[date, List[Tuple[str, float, float]]] = {}

        # ---------- 解析填充周班次、特殊日期班次 ----------
        for weekday, seg_list in weekly_rules.items():
            natural_segs = []
            for seg in seg_list:
                natural_segs.extend(seg.to_natural_day_segments())
            natural_segs.sort(key=lambda x: x[1])
            self._weekly_natural_segments[weekday] = self._merge_segments(natural_segs)

        if special_rules:
            for dt, seg_list in special_rules.items():
                natural_segs = []
                for seg in seg_list:
                    natural_segs.extend(seg.to_natural_day_segments())
                natural_segs.sort(key=lambda x: x[1])
                self._special_natural_segments[dt] = self._merge_segments(natural_segs)

        # ========= 第二步：所有班次数据初始化完成后，再初始化基准零点 =========
        self._base_work_zero_datetime: Optional[datetime] = None
        self._init_base_work_zero_point()

    @staticmethod
    def _merge_segments(segments: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
        """合并相邻或重叠的同名段（如连续的夜班段）"""
        if not segments:
            return []
        merged = []
        current_name, current_start, current_end = segments[0]
        for name, s, e in segments[1:]:
            if name == current_name and s <= current_end + 1e-9:
                current_end = max(current_end, e)
            else:
                merged.append((current_name, current_start, current_end))
                current_name, current_start, current_end = name, s, e
        merged.append((current_name, current_start, current_end))
        return merged

    # ---------- 核心数据访问 ----------
    def get_segments_for_date(self, dt: date) -> List[Tuple[str, float, float]]:
        """返回该日期的有效班次列表 [(名称, 开始, 结束), ...]，空列表表示全天无班次"""
        if dt in self._segments_cache:
            return self._segments_cache[dt]

        if dt in self._special_natural_segments:
            segs = self._special_natural_segments[dt]
        else:
            segs = self._weekly_natural_segments.get(dt.weekday(), [])

        sorted_segs = sorted(segs, key=lambda x: x[1])
        valid_segs = [(name, s, e) for name, s, e in sorted_segs if s < e - 1e-9]

        if dt == self.base_date:
            valid_segs = [(name, s, e) for name, s, e in valid_segs if not (name == "夜班" and s < 12.0 - 1e-9)]

        self._segments_cache[dt] = valid_segs
        return valid_segs

    def _segments_to_datetime(self, dt: date):
        seg_list = self.get_segments_for_date(dt)
        res = []
        for name, start_h, end_h in seg_list:
            # 小时浮点数转时分
            start_total_min = start_h * 60
            s_h = int(start_total_min // 60)
            s_m = int(start_total_min % 60)
            dt_start = datetime.combine(dt, time(hour=s_h, minute=s_m))

            end_total_min = end_h * 60
            e_h = int(end_total_min // 60)
            e_m = int(end_total_min % 60)
            # 处理24点，转为次日0点
            if e_h >= 24:
                dt_end = datetime.combine(dt + timedelta(days=1), time(0, 0))
            else:
                dt_end = datetime.combine(dt, time(hour=e_h, minute=e_m))
            res.append((name, dt_start, dt_end))
        return res

    def get_work_hours_for_date(self, dt: date) -> float:
        """返回某天总有效工作小时数（基于班次配置）"""
        segs = self.get_segments_for_date(dt)
        return sum(e - s for _, s, e in segs)

    def is_workday(self, input_val: Union[int, date, str]) -> bool:
        """判断是否为工作日（当天有班次）"""
        if isinstance(input_val, int):
            dt = self.daynum_to_date(input_val)
        elif isinstance(input_val, str):
            dt = date.fromisoformat(input_val)
        else:
            dt = input_val
        segs = self.get_segments_for_date(dt)
        return len(segs) > 0

    # ---------- 相对时间与日期的转换（保留） ----------
    def date_to_daynum(self, dt: Union[date, str]) -> int:
        if isinstance(dt, str):
            dt = date.fromisoformat(dt)
        return (dt - self.base_date).days

    def daynum_to_date(self, day_num: int) -> date:
        return self.base_date + timedelta(days=day_num)

    # ---------- 辅助拆分/合并相对小时 ----------
    def _split_relative_hour(self, total_relative_hour: float) -> Tuple[date, float]:
        """
        将全局相对工时拆分为：基准偏移日期 + 当天小时
        入参：基于基准零点的总相对工时
        返回：(对应日期date, 该日期内小时0~24)
        """
        # 基准零点所在日期
        base_zero_date = self._base_work_zero_datetime.date()
        remain_h = total_relative_hour
        curr_date = base_zero_date

        while True:
            daily_work_h = self.get_work_hours_for_date(curr_date)
            if remain_h < daily_work_h + 1e-9:
                break
            remain_h -= daily_work_h
            curr_date += timedelta(days=1)

        # 剩余工时换算成当天的时刻小时
        day_cursor = 0.0
        segs = self.get_segments_for_date(curr_date)
        for name, s, e in segs:
            seg_len = e - s
            if remain_h <= seg_len + 1e-9:
                day_cursor = s + remain_h
                break
            remain_h -= seg_len
            day_cursor = e

        return curr_date, day_cursor


    def _combine_relative_hour(self, dt: date, hour_of_day: float) -> float:
        """将 (日期, 当天小时) 转换回相对小时，移除递归调用，直接走原始零点计算"""
        target_dt = datetime.combine(dt, time(0, 0)) + timedelta(hours=hour_of_day)
        # 直接调用内部无偏移计算函数，不再递归调用对外转换入口
        total_natural = self._calc_datetime_to_relative_hours_from_base_zero(target_dt)
        offset_base = self._calc_datetime_to_relative_hours_from_base_zero(self._base_work_zero_datetime)
        final_relative = total_natural - offset_base
        return max(round(final_relative, 1), 0.0)

    # ---------- 核心时间校正函数 ----------
    def get_valid_start_hours_skip_holidays(self, ideal_start: float) -> float:
        """将理论时间校正到合法开工时刻（向后顺延，不提前）"""
        if ideal_start <= 1e-9:
            return 0.0

        dt, hour = self._split_relative_hour(ideal_start)

        for _ in range(366):  # 最多跨年
            segs = self.get_segments_for_date(dt)
            if segs:
                for name, s, e in segs:
                    if s - 1e-9 <= hour < e - 1e-9:
                        return self._combine_relative_hour(dt, hour)
                # 找下一个段开始
                next_start = None
                for name, s, e in segs:
                    if hour < s:
                        next_start = s
                        break
                if next_start is not None:
                    return self._combine_relative_hour(dt, next_start)

            dt += timedelta(days=1)
            hour = 0.0

        return self._combine_relative_hour(dt, 0.0)

    def get_actual_work_end_hours_skip_holidays(self, start_time: float, work_duration: float) -> float:
        """从 start_time 开始消耗 work_duration 个有效工作小时，返回完工相对小时"""
        if work_duration <= 1e-9:
            return start_time

        current_time = self.get_valid_start_hours_skip_holidays(start_time)
        remaining = work_duration
        dt, hour = self._split_relative_hour(current_time)

        for _ in range(366):
            segs = self.get_segments_for_date(dt)
            if not segs:
                dt += timedelta(days=1)
                hour = 0.0
                continue

            # 计算当天剩余可用工时
            day_remaining = 0.0
            in_segment = False
            for name, s, e in segs:
                if hour < s:
                    hour = s
                    in_segment = True
                    day_remaining += (e - hour)
                    hour = e
                elif hour < e:
                    in_segment = True
                    day_remaining += (e - hour)
                    hour = e

            if not in_segment and segs:
                hour = segs[0][1]
                day_remaining = sum(e - s for _, s, e in segs)

            if remaining <= day_remaining + 1e-9:
                for name, s, e in segs:
                    if e <= hour:
                        continue
                    if hour < s:
                        hour = s
                    seg_avail = e - hour
                    if remaining <= seg_avail + 1e-9:
                        return self._combine_relative_hour(dt, hour + remaining)
                    else:
                        remaining -= seg_avail
                        hour = e
                # 浮点误差补偿
                remaining -= day_remaining
                dt += timedelta(days=1)
                hour = 0.0
            else:
                remaining -= day_remaining
                dt += timedelta(days=1)
                hour = 0.0

        return self._combine_relative_hour(dt, hour)

    # ===================== 新增工具：获取单日首个日间白班起始小时 =====================
    def get_first_daytime_shift_start(self, dt: date) -> Optional[float]:
        """
        获取指定日期【第一个白班起始小时（0~24）】
        过滤规则：名称为白班的第一个开始时间，没有白班，就取第一个开始时间；无排班，就是None
        有白班
            班次：[夜班(4,8),白班(9,12),夜班(20,24)] → 返回 9.0
        全天无白班
            班次：[夜班(13,22),夜班(22,26)] → 返回 13.0
        当日无排班
            空列表 → 返回 None
        """
        seg_sorted_list = self.get_segments_for_date(dt)
        if not seg_sorted_list:
            return None

        daytime_segs = [seg for seg in seg_sorted_list if seg[0] == "白班"]
        if daytime_segs:
            return daytime_segs[0][1]
        return seg_sorted_list[0][1]

    def _init_base_work_zero_point(self):
        """初始化全局相对工时零点：base_date 第一个白班时刻"""
        first_start_h = self.get_first_daytime_shift_start(self.base_date)
        day_start_dt = datetime.combine(self.base_date, time.min)
        if first_start_h is not None:
            self._base_work_zero_datetime = day_start_dt + timedelta(hours=first_start_h)
        else:
            # 兜底：基准日无班次，零点仍为 base_date 0点
            self._base_work_zero_datetime = day_start_dt

    # # ===================== 首白班为零点=====================
    def _calc_datetime_to_relative_hours_from_base_zero(self, dt: datetime) -> float:
        """内部辅助：计算时间相对于 base_date 自然午夜0点的有效工时"""
        base_natural = datetime.combine(self.base_date, time.min)
        if dt < base_natural:
            return 0.0
        total = 0.0
        curr = base_natural
        while curr.date() < dt.date():
            total += self.get_work_hours_for_date(curr.date())
            curr += timedelta(days=1)
        segs = self.get_segments_for_date(dt.date())
        h = (dt - datetime.combine(dt.date(), time.min)).total_seconds() / 3600.0
        for name, s, e in segs:
            if h <= s:
                break
            elif h >= e:
                total += e - s
            else:
                total += h - s
                break
        return total

    def work_hours_between_relative_hour(self, start_time: float, end_time: float) -> float:
        """
        计算两个相对工时之间的总有效上班时长
        :param start_time: 起始相对工时 float
        :param end_time: 结束相对工时 float
        :return: 区间有效工作总小时
        """
        if end_time <= start_time + 1e-9:
            return 0.0
        # 两个入参全部转为datetime对象
        current_dt = self.base_relative_hour_to_real_datetime(start_time)
        end_dt = self.base_relative_hour_to_real_datetime(end_time)
        total_work_h = 0.0

        while current_dt < end_dt - timedelta(seconds=1e-9):
            curr_date = current_dt.date()
            seg_list = self._segments_to_datetime(curr_date)
            if not seg_list:
                # 当日无班次，跳到次日零点
                current_dt = datetime.combine(curr_date + timedelta(days=1), time.min)
                continue

            for seg_name, seg_s_dt, seg_e_dt in seg_list:
                if current_dt >= end_dt - timedelta(seconds=1e-9):
                    break
                if seg_e_dt <= current_dt + timedelta(seconds=1e-9):
                    continue
                overlap_start = max(current_dt, seg_s_dt)
                overlap_end = min(seg_e_dt, end_dt)
                delta_h = (overlap_end - overlap_start).total_seconds() / 3600.0
                total_work_h += delta_h
                current_dt = overlap_end

            if current_dt < datetime.combine(curr_date + timedelta(days=1), time.min):
                current_dt = datetime.combine(curr_date + timedelta(days=1), time.min)

        return round(total_work_h, 1)

    def work_hours_between_datetime(self, start_dt: datetime, end_dt: datetime) -> float:
        """start_dt 到 end_dt 之间的工作总秒数（左闭右开）"""
        total = 0.0
        cur_datetime = start_dt
        while cur_datetime < end_dt:
            segs = self._segments_to_datetime(cur_datetime.date())
            found = False
            for _, s_dt, e_dt in segs:
                if e_dt > cur_datetime:  # 找到可能重叠的段
                    found = True
                    if cur_datetime < s_dt:  # 休息中，快进到工作起点
                        cur_datetime = s_dt
                    overlap_end = min(e_dt, end_dt)
                    if overlap_end > cur_datetime:
                        total += (overlap_end - cur_datetime).total_seconds()
                    cur_datetime = e_dt  # 跳到该段结束
                    break
            if not found:
                # 当天已无更多工作段，直接跳到次日 00:00
                cur_datetime = datetime.combine(cur_datetime.date() + timedelta(days=1), time(0, 0))
        return round(np.ceil(total/3600), 1)

    def add_work_hours_to_datetime(self, start_dt: datetime, work_hours: float) -> datetime:
        """
        从起始时间 start_dt 向后累加指定工作小时，返回消耗完工时的目标时间
        整体逻辑：
        外层 while：只要还有剩余工时就循环
            取当前游标所在日期的全部工作时段
            逐个遍历时段，判断当前时间是否落在该时段结束之前
                若当前时间在时段前面：先跳到班次开始时间
                计算当前时段剩余时长
                    剩余工时 ≥ 时段时长：耗完整班次，扣减工时，游标跳到班次结束，继续下一个时段
                    剩余工时 < 时段时长：在当前时段内叠加剩余秒数，直接返回结果
        当天遍历完所有时段都没消耗完工时：跳到次日 0 点，进入下一轮while
        :param start_dt: 起始时间点
        :param work_hours: 需要累加的工作总小时（正数）
        :return: 工时耗尽对应的 datetime
        """
        if work_hours <= 1e-9:
            return start_dt

        cur_datetime = start_dt
        remain_sec = work_hours * 3600  # 统一转秒运算，避免浮点小时误差

        while remain_sec > 1e-9:
            segs = self._segments_to_datetime(cur_datetime.date())
            found = False

            for _, s_dt, e_dt in segs:
                if e_dt > cur_datetime:
                    found = True
                    # 游标在班次开始前，先跳到班次起点
                    if cur_datetime < s_dt:
                        cur_datetime = s_dt

                    seg_total_sec = (e_dt - cur_datetime).total_seconds()
                    if seg_total_sec <= remain_sec + 1e-9:
                        # 整个班次工时不够，耗尽本时段，扣减剩余工时，游标跳到班次结束
                        remain_sec -= seg_total_sec
                        cur_datetime = e_dt
                        break
                    else:
                        # 当前班次可以消化剩余全部工时，算出精准落点，直接返回
                        offset_sec = remain_sec
                        target_dt = cur_datetime + timedelta(seconds=offset_sec)
                        return target_dt

            if not found:
                # 当日后续无班次，跳到次日零点
                cur_datetime = datetime.combine(cur_datetime.date() + timedelta(days=1), time(0, 0))

        return cur_datetime

    def datetime_to_base_relative_hour(self, target_datetime: Union[datetime, date, str]) -> float:
        """
        真实datetime → 以base_date首白班为零点的有效相对工时
        双向可逆：base_relative_hour_to_datetime(x) 反算等于x
        """
        if isinstance(target_datetime, str):
            target_datetime = datetime.fromisoformat(target_datetime)
        elif isinstance(target_datetime, datetime):
            pass
        elif isinstance(target_datetime, date):
            target_datetime = datetime.combine(target_datetime, time(0, 0))

        # 早于基准自然日，直接返回0
        if target_datetime < self._base_work_zero_datetime:
            return 0.0
        return self.work_hours_between_datetime(self._base_work_zero_datetime, target_datetime)

    def base_relative_hour_to_real_datetime(self, relative_hour: float) -> datetime:
        """
        将【以base_date首白班为零点的有效工作小时】转为真实datetime
        relative_hour = 0 → base_date 第一个白班开始时刻
        """
        # 边界：0工时直接返回基准白班零点
        if relative_hour <= 1e-9:
            return self._base_work_zero_datetime

        return self.add_work_hours_to_datetime(self._base_work_zero_datetime, relative_hour)

    def base_relative_hour_to_first_start_datetime(self, relative_hour: float) -> datetime:
        """
        将基准相对工时转为实际时间；若工时耗尽时刻刚好等于某班次结束点，自动返回下一个班次起始时间
        :param relative_hour: 基于基准零点的相对工作小时
        :return: 目标时间（班次末尾则跳转下一班起点，否则返回原耗尽时间）
        """
        # 边界：0工时直接返回基准零点
        if relative_hour <= 1e-9:
            return self._base_work_zero_datetime

        # 1. 算出工时耗尽的真实时刻
        real_datetime = self.add_work_hours_to_datetime(self._base_work_zero_datetime, relative_hour)
        target_date = real_datetime.date()
        day_segs = self._segments_to_datetime(target_date)
        result_dt = real_datetime  # 默认返回原时间

        # 2. 枚举当日所有班次，判断是否落在某个班次末尾
        for seg_idx, seg_item in enumerate(day_segs):
            seg_name, s_dt, e_dt = seg_item
            if abs((real_datetime - e_dt).total_seconds()) < 1e-9:
                # 命中：当前时间刚好为本班次结束时刻
                if seg_idx + 1 < len(day_segs):
                    # 当天还有下一个班次，取下一班起始
                    result_dt = day_segs[seg_idx + 1][1]
                else:
                    # 已是当日最后一班，取次日第一个班次的起始时间
                    next_day = target_date + timedelta(days=1)
                    next_day_segs = self._segments_to_datetime(next_day)
                    if next_day_segs:
                        result_dt = next_day_segs[0][1]
                break

        return result_dt

    # ---------- 新增：班次信息查询 ----------
    def get_shift_info_by_relative_hour(self, relative_hour: float) -> Optional[Tuple[str, datetime, datetime]]:
        """
        根据基准相对工时，查询该时间归属的班次信息
        规则：时间在 [班次起始, 班次结束] 闭区间内都属于本班次（等于结束时刻也归属本班）
        :param relative_hour: 基于基准零点的相对工作小时
        :return: (班次名称, 班次开始datetime, 班次结束datetime)；无匹配班次返回None
        """
        # 1. 相对工时换算成真实绝对时间
        current_dt = self.add_work_hours_to_datetime(self._base_work_zero_datetime, relative_hour)
        check_date = current_dt.date()

        # 2. 获取当日全部datetime格式排班
        seg_list = self._segments_to_datetime(check_date)

        # 3. 遍历匹配班次，闭区间判断：s ≤ t ≤ e
        for shift_name, s_dt, e_dt in seg_list:
            if s_dt <= current_dt <= e_dt:
                return shift_name, s_dt, e_dt

        # 兜底：跨前一天查找（适配跨天班次场景）
        prev_date = check_date - timedelta(days=1)
        prev_seg_list = self._segments_to_datetime(prev_date)
        for shift_name, s_dt, e_dt in prev_seg_list:
            if s_dt <= current_dt <= e_dt:
                return shift_name, s_dt, e_dt

        # 兜底：跨后一天查找
        next_date = check_date + timedelta(days=1)
        next_seg_list = self._segments_to_datetime(next_date)
        for shift_name, s_dt, e_dt in next_seg_list:
            if s_dt <= current_dt <= e_dt:
                return shift_name, s_dt, e_dt

        # 所有日期都未匹配到班次
        return None
