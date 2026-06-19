from datetime import date, timedelta, datetime
from typing import Dict, List, Set, Tuple, Union
# core/calendar.py
from utils.log_utils import get_logger

logger = get_logger(__name__)

class WorkCalendar:
    def __init__(self, base_date: date, default_work_start: float = 8.0, default_work_end: float = 18.0):
        self.base_date: date = base_date    # 基准日期：相对工时0点对应的真实日历日期
        self.default_work_start: float = default_work_start
        self.default_work_end: float = default_work_end
        self.special_date_work_map: Dict[date, bool] = {}   # 特殊日期映射：key=日期, value=True=上班 False=休息（覆盖周规则）
        # 每日工时映射：支持不同日期不同上下班时间, # key:date对象, value:(上班小时,下班小时)
        self.date_hour_map: Dict[date, Tuple[float, float]] = {}
        # 周工作日规则：默认双休,# datetime.date.weekday() 规则：0=周一，1=周二，2=周三，3=周四，4=周五，5=周六，6=周日
        self.week_work_set: Set[int] = {0, 1, 2, 3, 4}

    def date_to_daynum(self, dt: Union[date, str]) -> int:
        if isinstance(dt, str):
            dt = date.fromisoformat(dt)
        return (dt - self.base_date).days

    def daynum_to_date(self, day_num: int) -> date:
        return self.base_date + timedelta(days=day_num)

    def is_workday(self, input_val: Union[int, date, str]) -> bool:
        """判断是否为工作日，支持3种入参"""
        # 第一步：统一转成date对象
        if isinstance(input_val, int):
            dt = self.daynum_to_date(input_val)
        elif isinstance(input_val, str):
            dt = date.fromisoformat(input_val)
        else:
            dt = input_val

        # 第二步：优先查特殊日期配置（节假日/调休）
        if dt in self.special_date_work_map:
            return self.special_date_work_map[dt]

        # 第三步：没有特殊配置，用周规则判断
        return dt.weekday() in self.week_work_set

    def add_calendar_item(self, dt: date, is_work: bool, work_start: float = None, work_end: float = None):
        self.special_date_work_map[dt] = is_work
        s = work_start if work_start is not None else self.default_work_start
        e = work_end if work_end is not None else self.default_work_end
        self.date_hour_map[dt] = (s, e)

    def set_week_rule(self, work_week_list: List[int]):
        self.week_work_set = set(work_week_list)


    def add_work_days(self, start_input: Union[int, date, str], work_days: int) -> int:
        """
        从开始日期向后/向前加指定个工作日，返回结束日期的天数偏移量
        :param start_input: 开始日期（支持天数偏移/date对象/日期字符串）
        :param work_days: 需要增加的工作日数量（正数向后，负数向前）
        :return: 结束日期的天数偏移量
        """
        current_day = self.date_to_daynum(start_input)
        remaining = abs(work_days)
        step = 1 if work_days > 0 else -1
        max_loop = 365 * 3  # 最多遍历3年，防死循环
        loop_cnt = 0
        while remaining > 0 and loop_cnt < max_loop:
            current_day += step
            if self.is_workday(current_day):
                remaining -= 1
            loop_cnt += 1
        return current_day

    def get_day_work_hour(self, day_num: int) -> Tuple[float, float]:
        target_date = self.daynum_to_date(day_num)
        if target_date in self.date_hour_map:
            return self.date_hour_map[target_date]
        return self.default_work_start, self.default_work_end

    def get_valid_start_time_skip_holidays(self, ideal_start: float) -> float:
        """
                将理论理想开工小时时间校正为符合车间工作日历的合法投产开工时刻
                校正规则：仅向后顺延，绝不提前开工
                1. 无工作日历配置时，直接返回原始ideal_start，不做时间偏移校正
                2. 第一步日期校正：若ideal_start所属日期为休息日/节假日，自动向后遍历找到第一个有效工作日
                3. 第二步当日时段校正（原日期本身是工作日时）：
                   - 理想时刻早于当日上班时间：顺延至当日正式开工时刻
                   - 理想时刻晚于当日下班时间：顺延至下一个有效工作日的开工时刻
                   - 理想时刻落在当日上下班区间内：时间完全合法，直接返回原值
                输入ideal_start：距离基准零点的总小时数（浮点型）
                返回值：校正后可正式投产的合法总小时数
                """

        # 时间拆分：总小时数拆成【第几天 + 当天几点】
        day_num = int(ideal_start // 24)  # 距离基准时间的总天数（整数，1天=24h）
        day_hour = ideal_start % 24  # 当天内的小时时刻（0~23.999）

        # 第一步：向后查找第一个有效的工作日
        current_day = day_num
        while not self.is_workday(current_day):  # 循环：如果当前天不是工作日，天数+1往后顺延
            current_day += 1

        # 第二步：根据当日时刻判断是否需要顺延
        # 场景A：已经顺延到后面的天数（原日期是非工作日）
        if current_day > day_num:
            # 直接取这个合法工作日的【上班起始时间】作为开工点
            work_start, _ = self.get_day_work_hour(current_day)
            return current_day * 24.0 + work_start
        # 场景B：原日期本身就是工作日，校验当天小时时段
        else:
            # 理想时刻早于上班时间 → 等到当天上班点再开工
            work_start, work_end = self.get_day_work_hour(day_num)
            if day_hour < work_start:
                return day_num * 24.0 + work_start
            # 理想时刻晚于下班时间 → 当天无法开工，找下一个工作日的上班点
            elif day_hour >= work_end:
                next_day = current_day + 1
                while not self.is_workday(next_day):
                    next_day += 1
                next_start, _ = self.get_day_work_hour(next_day)
                return next_day * 24.0 + next_start
            # 场景C：理想时刻正好落在当日上班~下班区间内，时间完全合法，原样返回
            else:
                return ideal_start

    def calculate_actual_work_end_time_skip_holidays(self, start_time: float, work_duration: float) -> float:
        remaining = work_duration
        current_time = self.get_valid_start_time_skip_holidays(start_time)
        while remaining > 0:
            day_num = int(current_time // 24)
            day_hour = current_time % 24

            # 跳过非工作日，直接跳转到下一个工作日上班时间
            if not self.is_workday(day_num):
                next_day = day_num + 1
                while not self.is_workday(next_day):
                    next_day += 1
                s, _ = self.get_day_work_hour(next_day)
                current_time = next_day * 24.0 + s
                continue

            # 获取当日上下班时间，计算当日剩余可用工时
            s, e = self.get_day_work_hour(day_num)
            available_today = e - day_hour
            if remaining <= available_today:
                current_time += remaining
                remaining = 0
            else:
                # 当日无法完成，耗满当日工时，剩余顺延至下一个工作日
                current_time = day_num * 24.0 + e
                remaining -= available_today
                next_day = day_num + 1
                while not self.is_workday(next_day):
                    next_day += 1
                ns, _ = self.get_day_work_hour(next_day)
                current_time = next_day * 24.0 + ns
        return current_time