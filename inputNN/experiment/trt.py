import sys
import os
from RNN.trt import ROBLSTM, chunk_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

HOME_FOLDER = r"./experiment"
os.makedirs(HOME_FOLDER, exist_ok=True)

targets_dict = {"1": "xpos", "2": "ypos", "3": ["sin(ang)","cos(ang)"]}

choice = str(input("1. Xpos\n2. Ypos\n3. Phi\nВыбор: "))
if choice not in targets_dict:
    sys.exit()

target_column_name = targets_dict[choice]
print(f"Запуск эксперимента для целевой переменной: {target_column_name}")

def autoregressive_forecast(model, test_scaled_df, test_raw_df, feature_cols, choice_key, start_row, total_steps_to_predict, window_size=25):
    model.eval()
    device = next(model.parameters()).device
    
    if choice_key == "1":
        target_cols = ["xpos"]
    elif choice_key == "2":
        target_cols = ["ypos"]
    elif choice_key == "3":
        target_cols = ["sin(ang)", "cos(ang)"]
    else:
        raise ValueError(f"Unknown choice_key: {choice_key}")
        
    target_idxs = [feature_cols.index(col) for col in target_cols]
    base_features = test_scaled_df[feature_cols].values.astype(np.float32)
    raw_targets = test_raw_df[target_cols].values.astype(np.float32)
    
    predictions = []
    for i in range(window_size):
        predictions.append(raw_targets[start_row + i])

    for step in range(start_row + window_size, start_row + window_size + total_steps_to_predict):
        if step >= len(test_scaled_df):
            break
            
        window_base = base_features[step - window_size : step].copy()
        
        for i in range(window_size):
            hist_step = step - window_size + i
            if hist_step >= start_row + window_size:
                pred_idx = hist_step - start_row
                for t_idx, base_idx in enumerate(target_idxs):
                    window_base[i, base_idx] = predictions[pred_idx][t_idx]

        current_window_tensor = torch.tensor(window_base, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_tensor = model(current_window_tensor)
            pred_numpy = pred_tensor.detach().cpu().numpy().flatten()
            
            if choice_key == "3":
                mag = np.sqrt(pred_numpy[0]**2 + pred_numpy[1]**2)
                if mag > 1e-6:
                    pred_numpy[0] /= mag
                    pred_numpy[1] /= mag
                    
            predictions.append(pred_numpy)

    final_forecast = np.array(predictions[window_size:])
    return final_forecast if choice_key == "3" else final_forecast.flatten()

df = pd.read_csv(r"RNN\Phi\robot_data_with_chunks.csv")

if "sin(ang)" not in df.columns:
    df["sin(ang)"] = np.sin(df["ang"])
if "cos(ang)" not in df.columns:
    df["cos(ang)"] = np.cos(df["ang"])

CHUNK_SIZE = 300
group_cols = ["movedir", "speedamp", "surf_copy"]
target_cols = [target_column_name]

full_group_cols = group_cols + ["unique_chunk_key", "surf_copy"]

targets_all = ["xpos", "ypos", "sin(ang)", "cos(ang)", "vx", "vy", "omega", 
               "m1vel", "m2vel", "m3vel", "w1slip", "w2slip", "w3slip", "m1cur", "m2cur", "m3cur"]

temp, train = chunk_split(df, strat="surf_copy", group_cols=full_group_cols, target_cols=targets_all, train_size=0.7)

deltas = ["vx", "vy", "omega"]
speeds = ["m1vel", "m2vel", "m3vel"]
slips = ["w1slip", "w2slip", "w3slip"]
currents = ["m1cur", "m2cur", "m3cur"]
surfaces = ["type_brown", "type_gray", "type_green", "type_table"]
coords_inputs = ["xpos", "ypos", "sin(ang)", "cos(ang)"]

features_to_standartize = deltas + speeds + slips + currents

SC_X = StandardScaler()
train_scaled = train.copy()
temp_scaled = temp.copy()

train_scaled[features_to_standartize] = SC_X.fit_transform(train[features_to_standartize])
temp_scaled[features_to_standartize] = SC_X.transform(temp[features_to_standartize])

all_features = features_to_standartize + surfaces + coords_inputs

model = ROBLSTM(input_dim=20,
                hidden_dim=32,
                num_layers=1,
                output_dim=1,
                dropout=0.2)

weights_map = {
    "1": r"RNN\X\Full_Context_with_Environments_And_Global_Coords_REMAKE\best_RNN_config.pth",
    "2": r"RNN\Y\Full_Context_with_Environments_And_Global_Coords_REMAKE\best_RNN_config.pth",
    "3": r"RNN\Phi\Full_Context_with_Environments_And_Global_Coords_REMAKE\best_RNN_config.pth"
}

model.load_state_dict(torch.load(weights_map[choice]))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

WINDOW_SIZE = 15
STEPS_TO_PREDICT = 30

available_chunks = train["unique_chunk_key"].unique()
print("\n=== ДОСТУПНЫЕ КЛЮЧИ ЗАЕЗДОВ В TRAIN (Первые 10 штук) ===")
for i, chunk_key in enumerate(available_chunks[:10]):
    print(f"[{i+1}] {chunk_key}")
print("=" * 55)

selected_key = str(input("Введите точный уникальный ключ чанка (например, '12_0'): "))
if selected_key not in available_chunks:
    print(f"Ошибка: Ключ '{selected_key}' не найден в обучающей выборке train!")
    sys.exit()

train_chunk_df = train[train["unique_chunk_key"] == selected_key].sort_index()
train_scaled_chunk_df = train_scaled[train_scaled["unique_chunk_key"] == selected_key].sort_index()

print(f"\nДлина выбранного чанка: {len(train_chunk_df)} шагов.")
start_row_user = int(input(f"Выберите начальный шаг окна (от 1 до {len(train_chunk_df) - WINDOW_SIZE}): "))

if start_row_user < 1 or start_row_user > (len(train_chunk_df) - WINDOW_SIZE):
    print("Ошибка: выбранный шаг выходит за допустимые границы чанка!")
    sys.exit()

START_ROW = start_row_user

print(f"\n=== ТАБЛИЦА ПАРАМЕТРОВ ДЛЯ ОКНА ИСТОРИИ (Чанк: {selected_key}, Шаги: {START_ROW} - {START_ROW + WINDOW_SIZE - 1}) ===")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(train_chunk_df[["t"] + target_cols].iloc[START_ROW : START_ROW + WINDOW_SIZE])
print("=" * 100 + "\n")

initial_10_points = train_chunk_df[target_column_name].values[START_ROW : START_ROW + WINDOW_SIZE]
y_pred = autoregressive_forecast(
    model=model, 
    test_scaled_df=train_scaled_chunk_df, 
    test_raw_df=train_chunk_df,
    feature_cols=all_features,  
    choice_key=choice, 
    start_row=START_ROW, 
    total_steps_to_predict=STEPS_TO_PREDICT,
    window_size=WINDOW_SIZE
)

y_true = train_chunk_df[target_column_name].values[START_ROW + WINDOW_SIZE : START_ROW + WINDOW_SIZE + len(y_pred)]

if len(y_pred) > 0:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"\n[{target_column_name.upper()}] Метрики чистой авторегрессии:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(WINDOW_SIZE), initial_10_points, label="Исходное окно истории", color="green", marker="o")
    plt.plot(range(WINDOW_SIZE, WINDOW_SIZE + len(y_true)), y_true, label="Реальная траектория", color="black", linestyle="--")
    plt.plot(range(WINDOW_SIZE, WINDOW_SIZE + len(y_pred)), y_pred, label="Прогноз LSTM (Авторегрессия)", color="red", marker="x")
    plt.axvline(x=WINDOW_SIZE-1, color='gray', linestyle=':')
    
    plt.title(f"Авторегрессия для {target_column_name}")
    plt.xlabel("Шаги времени")
    plt.ylabel(f"{target_cols}, м")
    plt.legend(loc="best")
    plt.grid(True)

    plot_path = os.path.join(HOME_FOLDER, f"pure_forecast_{target_column_name}.png")
    plt.savefig(plot_path)
    print(f"График успешно сохранен в: {plot_path}")
    plt.show()
else:
    print("Ошибка: не было сделано ни одного предсказания.")
