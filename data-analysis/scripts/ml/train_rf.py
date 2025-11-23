import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1.读取数据

df = pd.read_csv("processed/features_train.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

# 2.分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3.训练随机森林模型
print("训练随机森林模型中...")
model = RandomForestClassifier(n_estimators=100,
                               random_state=42,
                               n_jobs=-1) #使用多核加速

model.fit(X_train, y_train)

#4.模型评估
print("\n===训练完成，开始评估模型===\n")
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#5.保存模型

joblib.dump(model, "models/rf_model.pkl")

print("\n模型已保存：models/rf_model.pkl")
