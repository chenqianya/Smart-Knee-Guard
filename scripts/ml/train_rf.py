import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

df = pd.read_csv("../../processed/features_train.csv")
X = df.drop("label", axis=1)
y = df["label"]

scaler = StandardScaler().fit(X)
X_s = scaler.transform(X)

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_s, y)

y_pred = model.predict(X_s)
print("✅ 模型训练完成\n")
print(classification_report(y, y_pred))

os.makedirs("../../models", exist_ok=True)
joblib.dump(model, "../../models/rf_model.pkl")
joblib.dump(scaler, "../../models/scaler.pkl")
print("✅ 模型与scaler已保存至 ../models/")
