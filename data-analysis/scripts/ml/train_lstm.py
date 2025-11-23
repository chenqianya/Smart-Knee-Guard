import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "../../datasets/UCI_HAR_Dataset"

def load_raw(split="train"):
    sensors = [
        "body_acc_x","body_acc_y","body_acc_z",
        "body_gyro_x","body_gyro_y","body_gyro_z",
        "total_acc_x","total_acc_y","total_acc_z"
    ]
    data = [np.loadtxt(os.path.join(DATA_DIR, split, "Inertial Signals", f"{s}_{split}.txt")) for s in sensors]
    X = np.stack(data, axis=-1)
    y = np.loadtxt(os.path.join(DATA_DIR, split, f"y_{split}.txt")).astype(int)
    return X, y

X_train, y_train = load_raw("train")
X_test, y_test = load_raw("test")

# 将标签从 [1, 6] 转为 [0, 5]
y_train = y_train - 1
y_test = y_test - 1


model = models.Sequential([
    layers.Input(shape=(128, 9)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y_train)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

os.makedirs("../../models", exist_ok=True)
model.save("../models/lstm_har_tf29.h5")
print("✅ LSTM 模型已保存至 ../models/lstm_har_tf29.h5")
