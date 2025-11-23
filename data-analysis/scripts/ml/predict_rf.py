import joblib
import numpy as np

#1.加载模型
model = joblib.load("models/rf_model.pkl")

print("模型加载成功！")


#2-1.构造一条示例特征（等待我们进行数据采集ing！！！）
#这里先用随机数代替，共54维
sample = np.random.rand(54)

print("输入特征shape:", sample.shape)

#2-2.转成模型需要的格式（1，54）
sample = sample.reshape(1, -1)

#3.使用模型进行预测
prediction = model.predict(sample)

#4.输出预测结果
print("预测动作类别：", prediction[0])