import pickle
import sys

print("当前 Python:", sys.version)

filename = "models/scaler.pkl"

try:
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    print("Scaler 加载成功！说明是当前环境生成的。")
except Exception as e:
    print("加载失败。错误信息：")
    print(e)
