import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import itertools

root_path = r"C:\Users\User\Documents\MyPythonProjects\inputNN\neuro_physical_model\NPM_comparison"
df_info = pd.read_csv(r"RNN\Phi\robot_data_with_chunks.csv")

df_info.columns = [col.strip() for col in df_info.columns]

df = pd.read_excel(os.path.join(root_path, "FINAL_NPM_metrics_test.xlsx"))

models = df.iloc[:, 0].to_list()
columns = list(df.columns[1:-1])

ordered_features_for_mean = [
    "xpos", "ypos", "sin(ang)", "cos(ang)",
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

glued_relative_features = []

plt.figure(figsize=(15, 8))
for i in range(df.shape[0]):
    row_values = df[columns].iloc[i].values
    y_values = [row_values[idx] / (abs(mean_value[idx].item()) + 1e-8) for idx in range(len(columns))]
    
    if i == 0:
        glued_relative_features = list(y_values)
    
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

categories = ['Дельта', 'Проскальзывание', 'Ток', 'Скорость']

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

columns = [
    "Тип соединений",
    "Скорости",
    "Токи",
    "Проскальзывания",
    "Дельта координаты",
    "Координаты",
]
r2_train = [
    "R2 отдельных моделей",
    0.9996549,
    0.9750801,
    0.67339,
    0.99,
    0.999,
]


r2_glued = [
    "R2 отдельных моделей как частей NPM",
    0.9989,
    0.9690,
    0.6010,
    0.8715,
    0.9984,
]

table = pd.DataFrame(columns=columns, data=np.vstack((r2_train, r2_glued)))
print(table)

plt.figure(figsize=(10, 5))
x_labels = columns[1:]

for i in range(table.shape[0]):
    plt.plot(
        x_labels,
        table.iloc[i][1:].values.astype(float),
        marker="o",
        linewidth=2,
        label=table.iloc[i]["Тип соединений"],
    )

plt.grid(True, linestyle="--", alpha=0.7)
plt.xlabel("Группы параметров по цепочке NPM")
plt.ylabel("Коэффициент детерминации R2")
plt.title("Сравнение R2 на изолированном обучении и в составе склейки")
plt.legend(loc="best")
plt.tight_layout()
plt.show()



