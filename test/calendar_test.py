# test/calendar_test.py
import unittest
from core.calendar import WorkCalendar
from datetime import datetime, date
from data.test_dataset import HOLIDAYS_2026, MAKEUP_DAYS_2026
from config import DATE_WORK_MAP

class TestWorkCalendar(unittest.TestCase):
    # 每个test方法执行前自动运行，初始化实例
    def setUp(self):
        self.cal = WorkCalendar(
            base_date=datetime(2026, 6, 15).date(),
            default_work_start=8.0,
            default_work_end=20.0
        )
        # 预设节假日
        self.cal.holiday_list = ["2026-01-01", "2026-05-01"]
    #
    # # 测试判断工作日逻辑
    # def test_is_workday_normal(self):
    #     # 普通工作日，预期返回True
    #     res = self.cal.is_day_work("2026-06-15")
    #     self.assertEqual(res, True)
    #
    # # 测试节假日，预期返回False
    # def test_is_workday_holiday(self):
    #     res = self.cal.is_day_work("2026-01-01")
    #     self.assertEqual(res, False)
    #
    # # 测试边界空节假日场景
    # def test_empty_holiday(self):
    #     self.cal.holiday_list = []
    #     res = self.cal.is_day_work("2026-01-01")
    #     self.assertTrue(res)

    # def test_add_work_days(self):
    #     cal = WorkCalendar(base_date=date.today())
    #     # 批量添加节假日
    #     for holiday in HOLIDAYS_2026:
    #         cal.date_work_map[date.fromisoformat(holiday)] = False
    #     # 批量添加调休
    #     for makeup in MAKEUP_DAYS_2026:
    #         cal.date_work_map[date.fromisoformat(makeup)] = True

        # # 普通工作日向后加
        # self.assertEqual(cal.daynum_to_date(cal.add_work_days("2026-06-15", 3)), datetime(2026, 6, 18).date())
        # # 跨周末
        # self.assertEqual(cal.daynum_to_date(cal.add_work_days("2026-06-14", 5)), datetime(2026, 6, 21).date())
        # # 向前减工作日
        # self.assertEqual(cal.daynum_to_date(cal.add_work_days("2026-06-17", -2)), datetime(2026, 6, 13).date())

    def test_real_workday_logic(self):
        cal = WorkCalendar(base_date=date(2026, 6, 15))
        cal.date_work_map = DATE_WORK_MAP
        # 测试普通工作日
        self.assertTrue(cal.is_workday("2026-06-16"))  # 周二
        # 测试周末
        self.assertFalse(cal.is_workday("2026-06-21"))  # 周日
        # 测试周末
        self.assertFalse(cal.is_workday("2026-06-20"))  # 周日
        # 测试节假日
        # cal.date_work_map[date(2026, 10, 1)] = False
        self.assertFalse(cal.is_workday("2026-06-19"))
        # 测试调休
        # cal.date_work_map[date(2026, 10, 8)] = True
        self.assertTrue(cal.is_workday("2026-06-27"))
        # 测试加工作日
        # end_day = cal.add_work_days("2026-09-30", 3)
        # self.assertEqual(cal.daynum_to_date(end_day), date(2026, 10, 8))


# 脚本直接运行时执行全部测试
if __name__ == "__main__":
    unittest.main()