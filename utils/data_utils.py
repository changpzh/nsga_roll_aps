import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from utils.log_utils import get_logger

logger = get_logger(__name__)

def read_excel_sheet(
        file_path: str | Path,
        sheet_name: str = "Sheet1",
        fill_na: Any = ""
) -> pd.DataFrame:
    """
    读取Excel单个Sheet，自动处理空值和文件异常
    :param file_path: Excel文件路径
    :param sheet_name: 工作表名称
    :param fill_na: 空值填充值
    :return: pandas DataFrame
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Excel文件不存在：{file_path}")
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df = df.fillna(fill_na)
        logger.debug(f"成功读取Excel Sheet：{sheet_name}，共{len(df)}行数据")
        return df

    except Exception as e:
        logger.error(f"读取Excel失败：{str(e)}", exc_info=True)
        raise


def read_order_data(file_path: str | Path, sheet_name: str = "订单") -> List[Dict]:
    """
    专门读取排程订单数据，自动转换为字典列表
    :param file_path: 订单Excel文件路径
    :param sheet_name: 订单工作表名称
    :return: 订单字典列表
    """
    df = read_excel_sheet(file_path, sheet_name)
    orders = df.to_dict("records")

    # 自动转换数值类型
    for order in orders:
        if "order_id" in order:
            order["order_id"] = str(order["order_id"])
        if "quantity" in order:
            order["quantity"] = int(order["quantity"])
        if "priority" in order:
            order["priority"] = str(order["priority"]).lower()

    logger.info(f"成功读取订单数据：{len(orders)}条")
    return orders


def read_device_data(file_path: str | Path, sheet_name: str = "设备") -> List[Dict]:
    """
    专门读取设备基础数据
    :param file_path: 设备Excel文件路径
    :param sheet_name: 设备工作表名称
    :return: 设备字典列表
    """
    df = read_excel_sheet(file_path, sheet_name)
    devices = df.to_dict("records")

    for device in devices:
        if "device_id" in device:
            device["device_id"] = str(device["device_id"])
        if "capacity" in device:
            device["capacity"] = float(device["capacity"])

    logger.info(f"成功读取设备数据：{len(devices)}条")
    return devices


def write_schedule_result(
        schedule_result: List[Dict],
        output_path: str | Path,
        sheet_name: str = "排程结果"
) -> None:
    """
    将排程结果写入Excel文件
    :param schedule_result: 排程结果字典列表
    :param output_path: 输出文件路径
    :param sheet_name: 工作表名称
    """
    try:
        output_path = Path(output_path)
        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(schedule_result)
        # 按开始时间排序
        if "start_time" in df.columns:
            df = df.sort_values("start_time")

        df.to_excel(output_path, index=False, sheet_name=sheet_name)
        logger.info(f"排程结果已写入：{output_path}")

    except Exception as e:
        logger.error(f"写入排程结果失败：{str(e)}", exc_info=True)
        raise


def write_excel_with_sheets(
        data_dict: Dict[str, List[Dict]],
        output_path: str | Path
) -> None:
    """
    写入多Sheet Excel文件
    :param data_dict: 键=Sheet名，值=数据列表
    :param output_path: 输出文件路径
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, data in data_dict.items():
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"多Sheet Excel已写入：{output_path}")

    except Exception as e:
        logger.error(f"写入多Sheet Excel失败：{str(e)}", exc_info=True)
        raise


def validate_order_data(orders: List[Dict]) -> tuple[bool, List[str]]:
    """
    校验订单数据完整性，返回校验结果和错误信息列表
    :param orders: 订单列表
    :return: (是否全部合法, 错误信息列表)
    """
    required_fields = ["job_id", "quantity", "priority", "due_warn_time", "due_contract_time"]
    errors = []

    for idx, order in enumerate(orders):
        missing_fields = [f for f in required_fields if f not in order]
        if missing_fields:
            errors.append(f"订单{idx + 1}缺失必填字段：{', '.join(missing_fields)}")

        if order.get("quantity", 0) <= 0:
            errors.append(f"订单{order.get('order_id', idx + 1)}数量必须大于0")

    if errors:
        logger.warning(f"订单数据校验发现{len(errors)}个问题")
        return False, errors

    logger.debug("订单数据校验通过")
    return True, []


def validate_device_data(devices: List[Dict]) -> tuple[bool, List[str]]:
    """
    校验设备数据完整性
    """
    required_fields = ["device_id", "device_name", "process_capability"]
    errors = []

    for idx, device in enumerate(devices):
        missing_fields = [f for f in required_fields if f not in device]
        if missing_fields:
            errors.append(f"设备{idx + 1}缺失必填字段：{', '.join(missing_fields)}")

    if errors:
        logger.warning(f"设备数据校验发现{len(errors)}个问题")
        return False, errors

    logger.debug("设备数据校验通过")
    return True, []


def clean_nan_values(data: List[Dict], fill_value: Any = "") -> List[Dict]:
    """
    清理字典列表中的空值
    """
    cleaned = []
    for item in data:
        cleaned_item = {k: (v if pd.notna(v) else fill_value) for k, v in item.items()}
        cleaned.append(cleaned_item)
    return cleaned


def df_to_list_dict(df: pd.DataFrame) -> List[Dict]:
    """DataFrame转字典列表（兼容pandas各版本）"""
    return df.to_dict("records")


def list_dict_to_df(data: List[Dict]) -> pd.DataFrame:
    """字典列表转DataFrame"""
    return pd.DataFrame(data)

def fill_default_quantity(job_list: List[Dict], default_qty: int = 10) -> List[Dict]:
    """兼容旧订单，自动补全quantity和op_quantity"""
    for job in job_list:
        if "quantity" not in job or not isinstance(job["quantity"], int) or job["quantity"] <=0:
            job["quantity"] = default_qty
        job_qty = job["quantity"]
        for op in job["op_info_list"]:
            if "op_quantity" not in op or not isinstance(op["op_quantity"], int) or op["op_quantity"] <=0:
                op["op_quantity"] = job_qty
    return job_list