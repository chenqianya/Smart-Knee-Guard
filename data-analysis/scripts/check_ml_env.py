import os
import sys
import subprocess
import importlib
import tensorflow as tf

# 彩色输出
class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'

def check_and_install(package, import_name=None):
    """检测并自动安装库"""
    import_name = import_name or package
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', '未知版本')
        print(f"{Color.GREEN}✅ {package} 已安装，版本：{version}{Color.END}")
    except ImportError:
        print(f"{Color.RED}❌ {package} 未安装，正在安装...{Color.END}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{Color.GREEN}✅ {package} 安装完成！{Color.END}")

def check_tensorflow_device():
    """检查 TensorFlow 设备"""
    print(f"\n{Color.HEADER}💡 TensorFlow 设备检测:{Color.END}")
    try:
        devices = tf.config.list_physical_devices()
        if not devices:
            print(f"{Color.YELLOW}⚠️ 未检测到任何可用设备，TensorFlow 可能无法正常使用。{Color.END}")
        else:
            for d in devices:
                print(f"{Color.GREEN}✅ 检测到设备: {d}{Color.END}")
        # 测试计算
        result = tf.reduce_sum(tf.random.normal([3, 3]))
        print(f"\n🧮 简单计算测试成功：{result.numpy()}")
    except Exception as e:
        print(f"{Color.RED}❌ TensorFlow 测试失败: {e}{Color.END}")

def check_gpu_support():
    """检查 GPU 是否可用并提供建议"""
    print(f"\n{Color.HEADER}⚙️ GPU 支持检测:{Color.END}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"{Color.GREEN}✅ 检测到 GPU 设备: {gpus}{Color.END}")
    else:
        print(f"{Color.YELLOW}⚠️ 未检测到 GPU，可忽略（TensorFlow 将使用 CPU）。{Color.END}")
        print(f"{Color.BLUE}👉 如果你想使用 GPU 加速，可参考：https://www.tensorflow.org/install/gpu{Color.END}")

def print_env_info():
    """打印 Python 环境信息"""
    print("=" * 60)
    print(f"{Color.HEADER}📦 当前 Python 环境信息{Color.END}")
    print("=" * 60)
    print(f"Python 路径: {sys.executable}")
    print(f"Python 版本: {sys.version.split()[0]}")
    print("=" * 60)

def main():
    print_env_info()

    # 检查常用库
    packages = [
        "tensorflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ]
    for pkg in packages:
        check_and_install(pkg)

    check_tensorflow_device()
    check_gpu_support()

    print(f"\n{Color.GREEN}✅ 环境检测与修复完成！{Color.END}")
    print("=" * 60)

if __name__ == "__main__":
    main()
