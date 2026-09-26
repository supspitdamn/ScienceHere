from RNN.trt import ROBLSTM, chunk_split, RobotDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import sys

def autoregressive_forecast(model, test_scaled_df, feature_cols, start_row, total_steps_to_predict, window_size=10):
    model.eval()
    device = next(model.parameters()).device
    
    sin_idx = feature_cols.index("sin(ang)")
    cos_idx = feature_cols.index("cos(ang)")
    
    base_features = test_scaled_df[feature_cols].values.astype(np.float32)
    predictions_history = list(base_features[start_row : start_row + window_size, [sin_idx, cos_idx]])
    
    for step_idx in range(total_steps_to_predict):
        current_step = start_row + window_size + step_idx
        if current_step >= len(test_scaled_df):
            break
            
        window_input = base_features[current_step - window_size : current_step].copy()
        
        for i in range(window_size):
            simulated_global_step = current_step - window_size + i
            if simulated_global_step >= start_row + window_size:
                pred_idx = simulated_global_step - start_row
                window_input[i, sin_idx] = predictions_history[pred_idx][0]
                window_input[i, cos_idx] = predictions_history[pred_idx][1]
        
        current_window_tensor = torch.tensor(window_input, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_tensor = model(current_window_tensor)
            pred_numpy = pred_tensor.detach().cpu().numpy().flatten()
            
            sin_val, cos_val = pred_numpy[0], pred_numpy[1]
            magnitude = np.sqrt(sin_val**2 + cos_val**2)
            if magnitude > 1e-6:
                sin_val /= magnitude
                cos_val /= magnitude
            
            predictions_history.append(np.array([sin_val, cos_val]))

    return np.array(predictions_history[window_size:])

HOME_FOLDER = r"RNN\experiment"
os.makedirs(HOME_FOLDER, exist_ok=True)

print("Запуск эксперимента для целевой переменной: ФИ (sin/cos)")
df = pd.read_csv(r"RNN\Phi\robot_data_with_chunks.csv")

if "sin(ang)" not in df.columns:
    df["sin(ang)"] = np.sin(df["ang"])
if "cos(ang)" not in df.columns:
    df["cos(ang)"] = np.cos(df["ang"])

full_group_cols = ["movedir", "speedamp", "surf_copy", "unique_chunk_key"]
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
all_features = features_to_standartize + surfaces + coords_inputs

SC_X = StandardScaler()
train_scaled = train.copy()
train_scaled[features_to_standartize] = SC_X.fit_transform(train[features_to_standartize])

model = ROBLSTM(input_dim=20, hidden_dim=256, num_layers=3, output_dim=2, dropout=0.0)
weights_path = r"RNN\Phi\Full_Context_with_Environments_And_Global_Coords_REMAKE\best_RNN_config.pth"
model.load_state_dict(torch.load(weights_path))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

WINDOW_SIZE = 20
STEPS_TO_PREDICT = 30

available_chunks = train["unique_chunk_key"].unique()
print(f"\nДоступно чанков: {len(available_chunks)}. Первый доступный ключ: {available_chunks[0]}")
user_input_key = input("Введите уникальный ключ чанка: ")

try:
    if df["unique_chunk_key"].dtype in [np.int64, np.int32]:
        selected_key = int(user_input_key)
    elif df["unique_chunk_key"].dtype in [np.float64, np.float32]:
        selected_key = float(user_input_key)
    else:
        selected_key = str(user_input_key)
except ValueError:
    selected_key = user_input_key

if selected_key not in available_chunks:
    print("Ошибка: Ключ не найден!")
    sys.exit()

train_chunk_df = train[train["unique_chunk_key"] == selected_key].sort_index()
train_scaled_chunk_df = train_scaled[train_scaled["unique_chunk_key"] == selected_key].sort_index()

start_row_user = int(input(f"Выберите начальный шаг окна (от 1 до {len(train_chunk_df) - WINDOW_SIZE}): "))
START_ROW = start_row_user

initial_points_raw = train_chunk_df[["sin(ang)", "cos(ang)"]].values[START_ROW : START_ROW + WINDOW_SIZE]

y_pred = autoregressive_forecast(
    model=model, 
    test_scaled_df=train_scaled_chunk_df, 
    feature_cols=all_features,  
    start_row=START_ROW, 
    total_steps_to_predict=STEPS_TO_PREDICT,
    window_size=WINDOW_SIZE
)

y_true = train_chunk_df[["sin(ang)", "cos(ang)"]].values[START_ROW + WINDOW_SIZE : START_ROW + WINDOW_SIZE + len(y_pred)]

if len(y_pred) > 0:
    mae_sin = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_cos = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    
    print(f"\n[SIN(ANG)] MAE: {mae_sin:.4f} | RMSE: {np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0])):.4f}")
    print(f"[COS(ANG)] MAE: {mae_cos:.4f} | RMSE: {np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1])):.4f}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    
    ax1.plot(range(WINDOW_SIZE), initial_points_raw[:, 0], label="История sin", color="green", marker="o")
    ax1.plot(range(WINDOW_SIZE, WINDOW_SIZE + len(y_true)), y_true[:, 0], label="Реальный sin", color="black", linestyle="--")
    ax1.plot(range(WINDOW_SIZE, WINDOW_SIZE + len(y_pred)), y_pred[:, 0], label="Прогноз sin", color="red", marker="x")
    ax1.axvline(x=WINDOW_SIZE-1, color='gray', linestyle=':')
    ax1.grid(True)
    ax1.legend(loc="best")
    ax1.set_title("Авторегрессия: sin(ang)")
    
    ax2.plot(range(WINDOW_SIZE), initial_points_raw[:, 1], label="История cos", color="green", marker="o")
    ax2.plot(range(WINDOW_SIZE, WINDOW_SIZE + len(y_true)), y_true[:, 1], label="Реальный cos", color="black", linestyle="--")
    ax2.plot(range(WINDOW_SIZE, WINDOW_SIZE + len(y_pred)), y_pred[:, 1], label="Прогноз cos", color="blue", marker="x")
    ax2.axvline(x=WINDOW_SIZE-1, color='gray', linestyle=':')
    ax2.grid(True)
    ax2.legend(loc="best")
    ax2.set_title("Авторегрессия: cos(ang)")
    
    plt.xlabel("Шаги времени")
    plot_path = os.path.join(HOME_FOLDER, "pure_forecast_angle.png")
    plt.savefig(plot_path)
    print(f"График сохранен в: {plot_path}")
    plt.show()
