"""
Minimalist logger focusing on progress, errors, and format issues.
"""
import json
import logging
import sys
from datetime import datetime
from tqdm import tqdm


def setup_logger(name: str = "MedicalGPT"):
    """配置标准日志格式"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

# 全局系统日志实例
logger = setup_logger()


class SynthesisLogger:
    """业务进度日志：负责进度条和错误持久化"""
    
    def __init__(self, total_tasks: int, log_file: str | None = None):
        self.progress_bar = tqdm(total=total_tasks, desc="数据合成")
        self.error_log: list[dict] = []
        self.format_errors: list[dict] = []
        self.log_file = log_file
    
    def log_success(self):
        """成功处理一条"""
        self.progress_bar.update(1)
    
    def log_error(self, task_id: str, error: str):
        """关键错误"""
        entry = {"task_id": task_id, "error": error, "time": datetime.now().isoformat()}
        self.error_log.append(entry)
        
        # 如果是 .jsonl 结尾，实时写入
        if self.log_file and self.log_file.endswith('.jsonl'):
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def log_format_error(self, task_id: str, reason: str):
        """格式提取失败"""
        entry = {"task_id": task_id, "reason": reason, "time": datetime.now().isoformat()}
        self.format_errors.append(entry)
        
        if self.log_file and self.log_file.endswith('.jsonl'):
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        

    
    def summary(self):
        """最终统计"""
        self.progress_bar.close()
        print(f"\n成功: {self.progress_bar.n}")
        print(f"错误: {len(self.error_log)}")
        print(f"格式问题: {len(self.format_errors)}")
        
        # 保存错误日志到文件
        if self.log_file and (self.error_log or self.format_errors):
            if not self.log_file.endswith('.jsonl'):
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "errors": self.error_log,
                        "format_errors": self.format_errors
                    }, f, ensure_ascii=False, indent=2)
            print(f"\n错误详情已记录: {self.log_file}")
