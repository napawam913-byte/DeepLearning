import os
import pandas as pd
import torch

data_dir = r"D:\learning\DeepLearning\d2l_practice\data"
os.makedirs(data_dir, exist_ok=True)

csv_path = os.path.join(data_dir, "tiny_house.csv")

with open(csv_path, "w", encoding="utf-8") as f:
    f.write("NumRooms,Alley,Price\n")
    f.write("NA,Pave,127500\n")
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")

data = pd.read_csv(csv_path)
print("原始数据：")
print(data)

inputs = data.iloc[:, 0:2]
outputs = data.iloc[:, 2]

print("输入 inputs：")
print(inputs)

print("输出 outputs：")
print(outputs)

numeric_inputs = inputs[["NumRooms"]]
numeric_inputs = numeric_inputs.fillna(numeric_inputs.mean())

print("数值列缺失值填充后：")
print(numeric_inputs)

category_inputs = pd.get_dummies(inputs["Alley"], dummy_na=True)

print("类别列 one-hot 后：")
print(category_inputs)

processed_inputs = pd.concat([numeric_inputs, category_inputs], axis=1)

print("处理后的输入：")
print(processed_inputs)

X = torch.tensor(processed_inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))

print("X =")
print(X)
print("X.shape =", X.shape)

print("y =")
print(y)
print("y.shape =", y.shape)