# utils/log_utils.py
import logging
from logging.handlers import RotatingFileHandler
import os
from config.settings import LOG_CONFIG


class LogManager:
    """全局日志管理器，单例模式，保证整个项目只有一个日志配置"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._setup_global_logger()
        self._initialized = True

    def _setup_global_logger(self):
        """初始化全局日志配置"""
        # 创建logs目录
        log_dir = os.path.dirname(LOG_CONFIG["log_file_path"])
        os.makedirs(log_dir, exist_ok=True)

        # 获取根logger
        root_logger = logging.getLogger()
        root_logger.setLevel(LOG_CONFIG["global_level"])
        # 清空原有handler，避免重复输出
        root_logger.handlers.clear()

        # 1. 添加控制台输出handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_CONFIG["console_level"])
        console_formatter = logging.Formatter(
            LOG_CONFIG["formats"]["console"],
            datefmt=LOG_CONFIG["date_format"]
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # 2. 添加滚动文件输出handler
        file_handler = RotatingFileHandler(
            filename=LOG_CONFIG["log_file_path"],
            maxBytes=LOG_CONFIG["max_file_size"],
            backupCount=LOG_CONFIG["backup_count"],
            encoding=LOG_CONFIG["encoding"]
        )
        file_handler.setLevel(LOG_CONFIG["file_level"])
        file_formatter = logging.Formatter(
            LOG_CONFIG["formats"]["file"],
            datefmt=LOG_CONFIG["date_format"]
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    @staticmethod
    def get_logger(module_name: str = None) -> logging.Logger:
        """
        获取指定模块的logger实例
        :param module_name: 模块名，传__name__自动获取当前模块名
        :return: 对应模块的logger
        """
        if module_name is None:
            # 不传模块名默认返回根logger
            return logging.getLogger()
        # 返回对应模块的logger，自动继承全局配置
        return logging.getLogger(module_name)


# 对外暴露的快捷方法
def get_logger(module_name: str = None) -> logging.Logger:
    """获取模块logger的快捷方法，一行代码调用"""
    LogManager()  # 保证初始化
    return LogManager.get_logger(module_name)