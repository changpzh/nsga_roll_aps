# nsga_roll_aps

基于NSGA-II/NSGA-III算法的多目标滚动式车间APS排程系统

## 项目结构

- config：全局超参、配置项
- core：调度内核、日历、状态管理、NSGA-II算子
- trigger：滚动调度事件触发器
- visual：甘特图、帕累托前沿绘图
- data：测试数据集、Excel读写
- test：单元测试脚本

## 运行方式

1. 激活venv虚拟环境: venvScriptsactivate.bat
2. 执行 `python main.py` 启动排程运算

## 依赖

见 requirements.txt

## 单元测试运行方式

运行单个测试文件

python -m unittest test.calendar_test -v

v 代表详细打印每一条用例结果

# 后续优化点

1. 优化工序排程时找不到设备，找不到工人时的处理逻辑
2. 优化排程算法为nsga3型
3. 优化工作日有多个休息时段的
4. 优化订单给的是交付日期情形
5. 优化人员在某天请假，某天不请假情形 worker_aviable_dict={WorkId:{Datetime:bool}},
6. get_machine_overload_penalty,这个函数中过载应该是需要算出排程总的工作日数，应该判定：总的时间 > 排程工作日数 * 每天计划小时
7. wo 更改了
