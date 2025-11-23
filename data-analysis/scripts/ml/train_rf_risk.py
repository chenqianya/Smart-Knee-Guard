import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 动作编号 → 中文名称映射表

ACTION_MAP = {
    1: "站立",
    2: "行走",
    3: "跑步",
    4: "下蹲准备",
    5: "深蹲",
    6: "跳跃"
}

# 1. 读取数据

data_path = "processed/features_train_with_risk.csv"
df = pd.read_csv(data_path)

print("数据加载成功")
print("数据 shape:", df.shape)

# 2. 分离特征与标签

feature_cols = df.columns[:-2]  # 0~53
label_cols = ["label", "risk_label"]  # 输出两个标签

X = df[feature_cols]
y = df[label_cols]

# 输出示例说明（动作ID对应中文）
print("动作类别对应关系：")
for k, v in ACTION_MAP.items():
    print(f"  {k} → {v}")

# 3. 划分训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y["label"]
)

print("划分完成：")
print("训练集大小:", X_train.shape)
print("测试集大小:", X_test.shape)

# 4. 构建多输出随机森林模型

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model = MultiOutputClassifier(rf)

print("开始训练模型...")
model.fit(X_train, y_train)

print("训练完成！")

# 5. 测试集评估

y_pred = model.predict(X_test)

acc_action = accuracy_score(y_test["label"], y_pred[:, 0])
acc_risk = accuracy_score(y_test["risk_label"], y_pred[:, 1])

print("=== 测试结果 ===")
print(f"动作预测准确率：{acc_action:.4f}")
print(f"危险预测准确率：{acc_risk:.4f}")

# 6. 保存模型

os.makedirs("models", exist_ok=True)
save_path = "models/rf_risk_model.pkl"
joblib.dump(model, save_path)

print("模型已保存：", save_path)

# 7. 训练结束提示

print("=== 训练总结 ===")
print("动作类别对应：")
for k, v in ACTION_MAP.items():
    print(f"  {k} → {v}")

print("你现在可以使用 predict_rf_risk.py 进行预测。")
