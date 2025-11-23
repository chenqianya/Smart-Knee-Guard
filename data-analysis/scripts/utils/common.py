import os
import datetime

def create_dir_if_not_exists(path):
    """如果目录不存在，则创建"""
    if not os.path.exists(path):
        os.makedirs(path)

def log(msg):
    """简单日志打印函数"""
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
