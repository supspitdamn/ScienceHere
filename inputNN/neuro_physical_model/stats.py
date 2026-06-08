import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

root_path = r"C:\Users\User\Documents\MyPythonProjects\inputNN\neuro_physical_model\NPM_comparison"
df_info = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\УИРС\\SEM5\\filtered_robot_data.csv", encoding="cp1251", sep=";")

df_info.columns = [col.strip() for col in df_info.columns]

df = pd.read_excel(os.path.join(root_path, "FINAL_NPM_metrics_test.xlsx"))

models = df.iloc[:, 0].to_list()
columns = list(df.columns[1:-1])

ordered_features_for_mean = [
    "xpos", "ypos", "ang",
    "vx", "vy", "omega",
    "w1slip", "w2slip", "w3slip",
    "m1cur", "m2cur", "m3cur",
    "m1vel", "m2vel", "m3vel"
]

for col in ordered_features_for_mean:
    if col in df_info.columns:
        if df_info[col].dtype == 'object':
            df_info[col] = df_info[col].astype(str).str.replace(',', '.')
        df_info[col] = pd.to_numeric(df_info[col], errors='coerce').fillna(0.0)

mean_value = torch.tensor(df_info[ordered_features_for_mean].values.astype(np.float32)).abs().mean(dim=0)

plt.figure(figsize=(15, 8))
for i in range(df.shape[0]):
    row_values = df[columns].iloc[i].values
    y_values = [row_values[idx] / (abs(mean_value[idx].item()) + 1e-8) for idx in range(len(columns))]
    
    line, = plt.plot(columns, y_values, marker='o', linewidth=2, label=models[i])
    line_color = line.get_color()

    for idx, col in enumerate(columns):
        val = y_values[idx]
        base_step = 0.02

        if i % 2 == 0:
            direction_multiplier = 1 + (i // 2)
            final_y = val + (base_step * direction_multiplier)
            va_align = 'bottom'
        else:
            direction_multiplier = 1 + (i // 2)
            final_y = val - (base_step * direction_multiplier)
            va_align = 'top'
            
        plt.text(
            x=idx,
            y=final_y,    
            s=f"{val:.2f}",
            ha='center',                       
            va=va_align,
            fontsize=8,             
            fontweight='bold',
            color=line_color        
        )

plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel("Параметры стадий обучения")
plt.ylabel("Относительное значение MAE (MAE / Mean_Abs_Value)")
plt.title("Динамика изменения относительного MAE по стадиям обучения моделей")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


categories = ['X', 'Y', 'Фи', 'Дельта', 'Проскальзывание', 'Ток', 'Скорость']

plt.figure(figsize=(11, 6))

for i in range(df.shape[0]):
    row = df.iloc[i]
    y_values = []
    
    for cat in categories:
        cat_cols = [col for col in columns if col.startswith(cat)]
        mean_val = row[cat_cols].mean()
        y_values.append(mean_val)
        
    plt.plot(categories, y_values, marker='o', linewidth=2, label=models[i])

plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel("Группы физических параметров")
plt.ylabel("Среднее абсолютное значение MAE")
plt.title("Динамика изменения усредненного абсолютного MAE по стадиям моделирования")
plt.tight_layout()
plt.show()

