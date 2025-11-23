import os
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

DATA_DIR = "../../datasets/UCI_HAR_Dataset"

SENSORS = [
    "body_acc_x","body_acc_y","body_acc_z",
    "body_gyro_x","body_gyro_y","body_gyro_z",
    "total_acc_x","total_acc_y","total_acc_z"
]

def load_inertial(split="train"):
    data = []
    for s in SENSORS:
        arr = np.loadtxt(os.path.join(DATA_DIR, split, "Inertial Signals", f"{s}_{split}.txt"))
        data.append(arr)
    return np.stack(data, axis=-1)  # (n_samples, 128, 9)

def extract_features(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        feats += [np.mean(x), np.std(x), np.min(x), np.max(x)]
        yf = np.abs(rfft(x))
        feats.append(np.sum(yf**2))
        feats.append(np.argmax(yf))
    return np.array(feats)

def main():
    X_raw = load_inertial("train")
    X_feats = np.vstack([extract_features(w) for w in X_raw])
    y = np.loadtxt(os.path.join(DATA_DIR, "train", "y_train.txt")).astype(int)
    df = pd.DataFrame(X_feats)
    df["label"] = y
    os.makedirs("../../processed", exist_ok=True)
    df.to_csv("../processed/features_train.csv", index=False)
    print("✅ 特征数据已保存至 ../processed/features_train.csv")

if __name__ == "__main__":
    main()
