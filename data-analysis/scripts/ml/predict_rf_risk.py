"""
功能：
- 加载多输出随机森林模型 (动作, 危险)
- 支持三种输入方式：
    1) 随机样本 (--random)
    2) 从 CSV 中读取一行 (--csv path --idx N)
    3) 从命令行直接传入 54 个数 (--values v1 v2 ... v54)
- 校验输入长度 = 54
- 输出：
    - 动作编号 + 中文名
    - 动作类别的概率分布（每个类别的概率）
   ）
- 出错时给出友好提示
 - 危险概率（0/1 的概率
用法示例：
    python predict_rf_risk.py --random
    python predict_rf_risk.py --csv ../../processed/features_train_with_risk.csv --idx 10
    python predict_rf_risk.py --values 0.1 0.2 ... (54 个数)
"""

import argparse
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# 配置：动作映射（与训练脚本一致）

ACTION_MAP = {
    1: "站立",
    2: "行走",
    3: "跑步",
    4: "下蹲准备",
    5: "深蹲",
    6: "跳跃"
}

MODEL_PATH = Path("models/rf_risk_model.pkl")
EXPECTED_DIM = 54   # 必须是 54 维特征

# 辅助函数；加载模型和解析输入

def load_model(path: Path):  #定义一个函数来加载模型，输入是Path类型的路径
    if not path.exists():    #检查文件是否存在
        raise FileNotFoundError(f"模型文件不存在：{path}. 请先运行 train_rf_risk.py 生成模型。")
    model = joblib.load(path)
    return model

def parse_values_list(values_list):
    """把字符串列表转为浮点数组并校验长度"""
    try:
        arr = np.array([float(x) for x in values_list], dtype=float) #逐项把字符串转float
    except Exception as e:
        raise ValueError("无法将传入值转换为 float。请检查命令行输入。") from e #捕获原始异常并抛出更语义化的ValueError,提示用户是输入值转换的问题
    if arr.size != EXPECTED_DIM:  #严格校验元素数量，避免无效输入
        raise ValueError(f"输入的特征数量不对：期望 {EXPECTED_DIM}, 实际 {arr.size}")
    return arr.reshape(1, -1)

def predict_and_show(model, features_row):
    """对单条样本进行预测并打印详细信息（包含概率）"""
    # 1) 预测类别（硬预测）
    pred = model.predict(features_row)    #对单条样本做硬分类，对于MultiOutputClassifier,返回每个输入的预测值构成的数组
    action_pred = int(pred[0][0]) #返回每个输出的预测值构成的数组，形如[[action_label,risk_label]]
    risk_pred = int(pred[0][1]) #把危险预测值取出

    # 2) 尝试获取概率（MultiOutputClassifier（多输出分类） 返回的是每个输出的概率列表）
    action_proba = None
    risk_proba = None
    try:
        proba_list = model.predict_proba(features_row)
        # predict_proba 对于 MultiOutputClassifier 是 list-of-arrays:
        # proba_list[0] => 动作的概率矩阵 (1, n_classes_action)
        # proba_list[1] => 危险的概率矩阵 (1, 2)
        if isinstance(proba_list, list) and len(proba_list) >= 2:
            action_proba = proba_list[0][0]   # 形如 [p_class1, p_class2, ...]
            risk_proba = proba_list[1][0]     # 形如 [p_not_risk, p_risk] 或 [p0, p1]
    except Exception:
        # 如果基础模型或 sklearn 版本不支持 predict_proba（极少见），则跳过
        action_proba = None
        risk_proba = None

    # 3) 输出结果
    action_name = ACTION_MAP.get(action_pred, "未知动作") #将数值标签映为中文动作名，若找不到则显示“未知动作”
    print("\n=== 预测结果 ===")
    print(f"动作编号：{action_pred}  →  {action_name}")
    if action_proba is not None:
        # 打印每个动作概率（并显示最可能的动作概率）
        n_cls = action_proba.shape[0]
        probs_str = ", ".join([f"{i+1}:{action_proba[i]:.3f}" for i in range(n_cls)])
        print(f"动作概率分布（标签:概率）: {probs_str}")
        print(f"动作预测置信度: {action_proba[action_pred-1]:.3f}")
    else:
        print("动作概率信息不可用（predict_proba 不支持）")

    if risk_proba is not None:
        # 假设 risk_proba = [p0, p1], p1 即为“危险”的概率
        p_risk = float(risk_proba[1]) if len(risk_proba) > 1 else float(risk_proba[0])
        print(f"危险概率: {p_risk:.3f}  → {'危险' if p_risk>=0.5 else '正常'} (阈值 0.5)")
        print(f"危险 硬预测: {'危险' if risk_pred==1 else '正常'}")
    else:
        print(f"危险 硬预测: {'危险' if risk_pred==1 else '正常'}")
        print("危险概率信息不可用（predict_proba 不支持）")

# 主流程：命令行解析与输入准备

def main(argv): #主函数，argv是传入的命令行参数，通常是sys.argv[1:]
    parser = argparse.ArgumentParser(description="Predict action + risk using rf_risk_model.") #创建命令行解析器并写明用途
    group = parser.add_mutually_exclusive_group(required=True) #建立互斥组，一个用户必须选择其中一个输入方式，且只能选择一个
    group.add_argument("--random", action="store_true", help="use a random sample (for quick test)")
    group.add_argument("--csv", type=str, help="CSV path (read features from this CSV)")
    group.add_argument("--values", nargs="+", help="54 numeric features provided inline")
    parser.add_argument("--idx", type=int, default=0, help="when --csv is used, which row index to read (0-based)")
    args = parser.parse_args(argv)


    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print("加载模型失败：", e)
        sys.exit(1)

    # 依据输入方式准备 features_row (shape (1,54))
    try:
        if args.random:
            sample = np.random.rand(EXPECTED_DIM).astype(float).reshape(1, -1)
            features_row = sample
            print("使用随机样本进行测试。")
        elif args.csv:
            csv_path = Path(args.csv)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV 文件不存在：{csv_path}")
            df = pd.read_csv(csv_path)
            # 我们期望 CSV 中前 54 列为特征；也支持若 CSV 包含 label/risk 等列，则只取前 EXPECTED_DIM 的列
            if df.shape[1] < EXPECTED_DIM:
                raise ValueError(f"CSV 列数不足：期望至少 {EXPECTED_DIM} 列特征，实际 {df.shape[1]}")
            row_idx = args.idx
            if row_idx < 0 or row_idx >= len(df):
                raise IndexError(f"索引越界：csv 有 {len(df)} 行，无法读取 idx={row_idx}")
            values = df.iloc[row_idx, :EXPECTED_DIM].values.astype(float)
            features_row = values.reshape(1, -1)
            print(f"从 CSV {csv_path} 读取第 {row_idx} 行作为输入。")
        else:
            # --values 模式
            features_row = parse_values_list(args.values)
            print("使用命令行传入的特征向量。")
    except Exception as e:
        print("准备输入特征时出错：", e)
        sys.exit(1)

    # 最后预测并显示
    try:
        predict_and_show(model, features_row)
    except Exception as e:
        print("预测时发生异常：", e)
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])
