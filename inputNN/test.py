import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\УИРС\\SEM5\\filtered_robot_data.csv", encoding="cp1251", sep=";")

surf_cols = ["surf"]
vel_cols = ["m1vel", "m2vel", "m3vel"]
setvel_cols = ["m1setvel", "m2setvel", "m3setvel"]

print(f"{df[df["w1linslip"].abs() < 0.01].shape[0]} из {df.shape[0]}")
print(max(df["w1slip"]))

df["w1slip"].hist(bins=50)  # bins — это количество столбиков
plt.title("Распределение проскальзывания w1slip")
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.show()