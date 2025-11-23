import pandas as pd
import numpy as np

# 1. 读取数据
df = pd.read_csv("processed/features_train.csv")
print("原始数据 shape:", df.shape)

# 2-1. 列划分
feature_cols = df.columns[:-1]     # 特征列 0~53
label_col = "label"                # 动作类别列

#2-2 创建新的风险列（全部先设为 0 = 正常）
df["risk_label"] = 0

print("=== 开始风险检测（方法：偏离平均值 3σ） ===")

#2-3 记录危险和正常的数量
total_risky = 0
total_normal = 0

# 3. 对每一种动作类别进行风险检测
for action in sorted(df[label_col].unique()):
    print(f"▶ 动作类别 {action}")

    # 抽取该动作的数据
    sub_df = df[df[label_col] == action]

    # 均值 & 标准差
    mean = sub_df[feature_cols].mean()
    std = sub_df[feature_cols].std()

    # 计算 Z-Score（偏离程度）
    z_score = (sub_df[feature_cols] - mean) / std

    # 判断危险：任意一列超过 3σ
    is_risky = (np.abs(z_score) > 3).any(axis=1)

    # 标记危险
    df.loc[sub_df.index[is_risky], "risk_label"] = 1

    # 数量统计
    risky_count = is_risky.sum()
    normal_count = len(sub_df) - risky_count

    total_risky += risky_count
    total_normal += normal_count

    print(f"    - 正常样本数：{normal_count}")
    print(f"    - 危险样本数：{risky_count}")

# 4. 保存新数据
output_path = "processed/features_train_with_risk.csv"
df.to_csv(output_path, index=False)

print("=== 风险标签生成完成 ===")
print("新文件已保存：", output_path)

print("=== 全局统计 ===")
print(f"总正常样本数：{total_normal}")
print(f"总危险样本数：{total_risky}")
print("数据集最终 shape:", df.shape)
