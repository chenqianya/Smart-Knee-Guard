import pandas as pd

# 加载你实际训练用的数据文件
df = pd.read_csv("processed/features_train.csv")

# 如果你已经使用 features_train_with_risk.csv，也可以改成下面这一行：
# df = pd.read_csv("processed/features_train_with_risk.csv")

print("标签列（label）唯一值：", sorted(df["label"].unique()))

print("\n每个标签出现的次数：")
print(df["label"].value_counts())

# 查看每个标签的数据特征分布是否类似
print("\n按标签分组后取前 2 行（用于人工快速检查）：")
for label, group in df.groupby("label"):
    print(f"\n--- label = {label} 的样本前 2 行 ---")
    print(group.iloc[:2, :10])   # 只显示前10列，便于阅读
