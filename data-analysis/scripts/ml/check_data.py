import pandas as pd
p = "processed/features_train.csv"
df = pd.read_csv(p)
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("labels unique:", sorted(df["label"].unique()))
print(df.dtypes)
print("null:",df.isnull().sum().sum())