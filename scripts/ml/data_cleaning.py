import os
import pandas as pd

# 数据集目录
DATA_DIR = "../../datasets/UCI_HAR_Dataset"

# 1. 读取特征名称
features = pd.read_csv(
    os.path.join(DATA_DIR, "features.txt"),
    sep="\s+",
    header=None,
    names=["id", "feature"]
)

# 2. 去重（防止重复列名引起 ValueError）
features["feature"] = features["feature"].astype(str)
features["feature"] = features["feature"] + features.groupby("feature").cumcount().replace(0, '', regex=True).astype(str)

# 3. 加载训练数据
X_train = pd.read_csv(
    os.path.join(DATA_DIR, "train", "X_train.txt"),
    sep="\s+",
    header=None,
    names=features["feature"]
)
y_train = pd.read_csv(
    os.path.join(DATA_DIR, "train", "y_train.txt"),
    sep="\s+",
    header=None,
    names=["label"]
)

# 4. 打印信息确认
print("✅ 数据加载成功")
print("训练集形状:", X_train.shape)
print("标签分布:\n", y_train["label"].value_counts())
print("是否存在缺失值:", X_train.isna().sum().sum())
