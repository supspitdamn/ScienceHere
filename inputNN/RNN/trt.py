import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import pandas as pd

# Читаем файлы. Чтобы считать 
df = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\УИРС\\SEM5\\filtered_robot_data.csv", encoding="cp1251", sep=";", decimal=',')

print(df.info())

for col in df.columns:
    if col != 'surf' and col != 'movedir':
        df[col] = df[col].astype(np.float32)

df = df.ffill()

df_sorted = df.sort_values(by=['movedir', 't'])

print(df_sorted)

print(df["movedir"].unique())

plt.hist(df["movedir"], align = "mid", label = "Гистограмма распределения movedir")
plt.xlabel("Значение movedir")
plt.ylabel("Частота упоминаний")
plt.title("Гистограмма распределения movedir")
# plt.show()

print(df.sort_values(by = "t"))
