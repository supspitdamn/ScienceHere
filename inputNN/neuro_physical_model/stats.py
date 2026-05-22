import matplotlib.pyplot as plt
import pandas as pd
import os
import torch

root_path = r"C:\Users\User\Documents\MyPythonProjects\inputNN\neuro_physical_model\NPM_comparison"
df_info = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\УИРС\\SEM5\\filtered_robot_data.csv", encoding="cp1251", sep=";")

df_info[["m1vel", "m2vel", "m3vel", "m1cur", "m2cur", "m3cur", "w1slip", "w2slip", "w3slip", "vx", "vy", "omega"][::-1]].describe().to_csv("features_stats.csv", sep=";")

df = pd.read_excel(os.path.join(root_path, "FINAL_NPM_metrics_test.xlsx"))

print(df_info[["vx", "vy", "omega", "w1slip", "w2slip", "w3slip", "m1vel", "m2vel", "m3vel", "m1cur", "m2cur", "m3cur"]])
print(df)

models = df.iloc[:-1, 0].to_list()
print(models)

columns = list(df.columns[1:-1])
mean_value = torch.tensor(df_info[["vx", "vy", "omega", "w1slip", "w2slip", "w3slip", "m1cur", "m2cur", "m3cur", "m1vel", "m2vel", "m3vel"]].values).abs().mean(dim=0)
print(mean_value)
print(columns)

plt.figure(figsize=(14, 7))
for i in range(df.shape[0] - 1):

    row_values = df[columns].iloc[i].values
    y_values = [row_values[idx] / abs(mean_value[idx].item()) for idx in range(len(columns))]
    
    line, = plt.plot(columns, y_values, marker='o', linewidth=2, label=models[i])
    line_color = line.get_color()

    for idx, col in enumerate(columns):
        val = y_values[idx]

        base_step = 0.005

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

plt.gca().invert_xaxis()
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel("Стадии обучения")
plt.ylabel("Значения MAE")
plt.title("Динамика изменения MAE по стадиям обучения")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import os

root_path = r"C:\Users\User\Documents\MyPythonProjects\inputNN\neuro_physical_model\NPM_comparison"
df = pd.read_excel(os.path.join(root_path, "FINAL_NPM_metrics_test.xlsx"))

models = df.iloc[:-1, 0].to_list()
print(models)

categories = ['Дельта', 'Проскальзывание', 'Скорость', 'Ток']

plt.figure(figsize=(10, 6))

for i in range(df.shape[0] - 1):
    row = df.iloc[i]
    y_values = []
    
    for cat in categories:
        cat_cols = [col for col in df.columns if col.startswith(cat)]
        mean_val = row[cat_cols].mean()
        y_values.append(mean_val)
        
    plt.plot(categories, y_values, marker='o', linewidth=2, label=models[i])

plt.gca().invert_xaxis()
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel("Группы параметров")
plt.ylabel("Среднее значение MAE")
plt.title("Динамика изменения усредненного MAE по группам параметров")
plt.tight_layout()
plt.show()
