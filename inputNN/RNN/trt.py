import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import pandas as pd
from tqdm import tqdm
import os
import optuna

def objective(trial, current_features, targets_cols, df_train, df_val,  device, root_path):

    sequence_length = trial.suggest_int("sequence_length", 5, 30, step=5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_dataset = RobotDataset(df_train, sequence_length, current_features, targets_cols)
    val_dataset = RobotDataset(df_val, sequence_length, current_features, targets_cols)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, pin_memory=True)

    model = ROBLSTM(
        input_dim=len(current_features),
        hidden_dim=hidden_dim,
        output_dim=len(targets_cols),
        num_layers=num_layers,
        dropout=dropout
    )

    model.to(device)

    trial_folder = os.path.join(root_path, f"trial_{trial.number}")

    os.makedirs(trial_folder, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_losses, val_losses = model.fit(
        op=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10, 
        root_path=trial_folder,
        device=device,
        patience=3,
    )

    return min(val_losses)

class RobotDataset(Dataset):

    def __init__(self, grouped_df, sequence_length, feature_cols, target_cols):

        self.sequence_length = sequence_length
        self.features = grouped_df[feature_cols].values.astype(np.float32)
        self.target = grouped_df[target_cols].values.astype(np.float32)

        try:
            movedir_labels = grouped_df["movedir"].values
        except (AttributeError, KeyError) as e:
            raise AttributeError(f"Ошибка: {e}")

        self.valid_indices = []
        num_rows = len(movedir_labels)

        for i in range(num_rows - sequence_length):

            if movedir_labels[i] == movedir_labels[i + sequence_length]:

                self.valid_indices.append(i)
        
    def __len__(self):

        return len(self.valid_indices)

    def __getitem__(self, idx):

        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        x = self.features[start_idx:end_idx]
        y = self.target[end_idx]

        return torch.tensor(x), torch.tensor(y)

def train_val_test_split(df: pd.DataFrame, train_size, val_size):

    train_frames = []
    val_frames = []
    test_frames = []

    for _, group in df.groupby("movedir", sort=False):

        n = len(group)
        train_end_idx = int(n*train_size)
        val_end_idx = train_end_idx + int(n*val_size)

        train_frames.append(group.iloc[:train_end_idx])
        val_frames.append(group.iloc[train_end_idx:val_end_idx])
        test_frames.append(group.iloc[val_end_idx:])

    df_train = pd.concat(train_frames).reset_index(drop = True)
    df_val = pd.concat(val_frames).reset_index(drop = True)
    df_test = pd.concat(test_frames).reset_index(drop = True)

    return df_train, df_val, df_test

class ROBLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout = 0.2):

        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(
            hidden_dim, output_dim
        )
    
    def forward(self, x):

        x = self.input_norm(x)

        lstm_out, _ = self.lstm(x)

        last_time_step_out = lstm_out[:, -1, :]

        out = self.fc(last_time_step_out)

        return out

    def fit(self, op, criterion, scheduler, train_loader, val_loader, epochs, root_path, device, patience = 10):

        train_loss_avg = []
        val_loss_avg = []
        lr_change = []
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(root_path, "best_RNN_config.pth")

        for epoch in range(epochs):

            self.train()
            
            tqdm_train_loader = tqdm(train_loader, desc = f"Эпоха {epoch+1}/{epochs} [Обучение]")

            train_loss = 0
            val_loss = 0
            status_msg = ""

            for x, y in tqdm_train_loader:
                
                x, y = x.to(device), y.to(device)
                op.zero_grad()
                pred = self(x)

                loss = criterion(pred, y)

                loss.backward()
                op.step()

                train_loss += loss.item()

                tqdm_train_loader.set_postfix(batch_loss = loss.item())

            train_loss_avg.append(train_loss/len(train_loader))

            self.eval()

            tqdm_val_loader = tqdm(val_loader, desc = f"Эпоха {epoch+1}/{epochs} [Валидация]")

            with torch.no_grad():

                for x, y in tqdm_val_loader:

                    x, y = x.to(device), y.to(device)

                    pred = self(x)

                    loss = criterion(pred, y)

                    val_loss += loss.item()
                    tqdm_val_loader.set_postfix(batch_loss = loss.item())

            val_loss_avg.append(val_loss/len(val_loader))

            scheduler.step(val_loss_avg[-1])    
            lr_change.append(op.param_groups[0]["lr"])

            if val_loss_avg[-1] < best_val_loss:

                best_val_loss = val_loss_avg[-1]
                patience_counter = 0

                torch.save(self.state_dict(), best_model_path)

                status_msg = f"-> Лосс снизился. Модель сохранена"
            
            else:

                patience_counter += 1
                status_msg = f"-> Лосс не изменился. Терпение {patience_counter}/{patience}"
            
            print(status_msg + "\n")

            if patience_counter >= patience:

                print(f"Остановка на эпохе {epoch+1}/{epochs}")

                self.load_state_dict(torch.load(best_model_path))
            
                break
        
        plt.figure()
        plt.suptitle("Процесс обучения")

        plt.subplot(1,2,1)
        plt.title("Изменение лосс")
        plt.plot(range(epochs)[:len(val_loss_avg)], val_loss_avg, label = "Валидация")
        plt.plot(range(epochs)[:len(train_loss_avg)], train_loss_avg, label = "Обучение")
        plt.xlabel("Эпохи")
        plt.ylabel("Лосс")
        plt.legend(loc = "best")

        plt.subplot(1, 2, 2)
        plt.title("Изменение шага обучения")
        plt.step(range(epochs)[:len(lr_change)], lr_change, label = "Шаг обучения")
        plt.xlabel("Эпохи")

        plt.savefig(os.path.join(root_path, "learning_info.png"), dpi = 300)
        plt.close()

        return train_loss_avg, val_loss_avg
        
    def evaluate_all(self, loaders: dict, save_path: str, device: str = "cpu") -> pd.DataFrame:

        data_dict = {}
        self.eval()
        
        with torch.no_grad():
            for loader_name, loader in loaders.items():
                all_pred = []
                all_true = []
                
                for x, y in tqdm(loader, desc=f"Оценка [{loader_name}]"):
                    x, y = x.to(device), y.to(device)
                    predict = self(x).cpu().numpy()
                    true_value = y.cpu().numpy()
                    
                    all_pred.append(predict)
                    all_true.append(true_value)
                    
                all_pred = np.vstack(all_pred)
                all_true = np.vstack(all_true)
                
                mse = mean_squared_error(all_true, all_pred, multioutput="raw_values")
                mae = mean_absolute_error(all_true, all_pred, multioutput="raw_values")
                r2 = r2_score(all_true, all_pred, multioutput="raw_values")
                
                os.makedirs(save_path, exist_ok=True)
                
                if loader_name == "test":
                    with open(os.path.join(save_path, "LOG.txt"), "a", encoding="utf-8") as log_txt:
                        log_txt.write(20 * "-" + "\n")
                        log_txt.write("Результаты для тестовой выборки:\n")
                        log_txt.write(f"Абсолютная ошибка (Дельта Х, Дельта У, Дельта Фи): {'  '.join(map(str, np.round(mae, 4)))}\n")
                
                data_dict[loader_name] = {
                    ("Дельта Х", "MSE"): mse[0], ("Дельта Х", "MAE"): mae[0], ("Дельта Х", "R2"): r2[0],
                    ("Дельта У", "MSE"): mse[1], ("Дельта У", "MAE"): mae[1], ("Дельта У", "R2"): r2[1],
                    ("Дельта Фи", "MSE"): mse[2], ("Дельта Фи", "MAE"): mae[2], ("Дельта Фи", "R2"): r2[2]
                }

        summary_df = pd.DataFrame.from_dict(data_dict, orient="index")
        summary_df.index.name = "Набор данных"
        
        # Сохраняем в CSV (мультииндекс запишется корректно в две верхние строчки)
        summary_df.to_csv(os.path.join(save_path, "FINAL_metrics_all.csv"), encoding="utf-8-sig")
        
        return summary_df

df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\УИРС\SEM5\filtered_robot_data_csv.csv", encoding="cp1251", sep = ";")

home_folder = r"C:\Users\User\Documents\MyPythonProjects\inputNN\RNN"

os.makedirs(home_folder, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Обучение на {device}")

print(df.info())

df.columns = [column.strip() for column in df.columns]

plt.hist(df["movedir"], align = "mid", label = "Гистограмма распределения movedir")
plt.xlabel("Значение movedir")
plt.ylabel("Частота упоминаний")
plt.title("Гистограмма распределения movedir")

df_grouped = df.sort_values(by = ["movedir", "t"])

df_train, df_val, df_test = train_val_test_split(df_grouped, 0.7, 0.2)

deltas = ["vx", "vy", "omega"]
speeds = ["m1vel", "m2vel", "m3vel"]
slips = ["w1slip", "w2slip", "w3slip"]
currents = ["m1cur", "m2cur", "m3cur"]
surfaces = ["type_brown", "type_gray", "type_green", "type_table"]

feature_expirements = {
    "Base_Odometry" : deltas + speeds,
    "Odometry_with_Slippage" : deltas + speeds + slips,
    "Odometry_with_Currents" : deltas + speeds + currents,
    "Full_motor_Physics" : deltas + speeds + slips + currents,
    "Full_Context_wwith_Environments" : deltas + speeds + slips + currents + surfaces
}

targets_cols = ["xcur", "ycur", "rotcur"]

for exp_name, current_features in feature_expirements.items():

    root_path = os.path.join(home_folder, f"{"-".join(current_features)}")

    os.makedirs(root_path, exist_ok=True)

    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective(
            trial, current_features, targets_cols, df_train, df_val, device, root_path
        ),
        n_trials=20,
    )

    print("\n" + "=" * 50)
    print(f"ПОДБОР ЗАВЕРШЕН ДЛЯ ЭКСПЕРИМЕНТА: {exp_name}")
    print(f"Лучший достигнутый Val Loss: {study.best_value:.6f}")
    print("Лучшие параметры архитектуры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 50)

    best = study.best_params

    final_train_dataset = RobotDataset(
        df_train, best["sequence_length"], current_features, targets_cols
    )
    final_val_dataset = RobotDataset(
        df_val, best["sequence_length"], current_features, targets_cols
    )
    final_test_dataset = RobotDataset(
        df_test, best["sequence_length"], current_features, targets_cols
    )

    final_train_loader = DataLoader(
        final_train_dataset,
        batch_size=best["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    final_val_loader = DataLoader(
        final_val_dataset, batch_size=512, shuffle=False, pin_memory=True
    )
    final_test_loader = DataLoader(
        final_test_dataset, batch_size=512, shuffle=False, pin_memory=True
    )

    final_model = ROBLSTM(
        input_dim=len(current_features),
        hidden_dim=best["hidden_dim"],
        output_dim=len(targets_cols),
        num_layers=best["num_layers"],
        dropout=best["dropout"],
    )
    final_model.to(device)

    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best["lr"])
    final_criterion = nn.MSELoss()
    final_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        final_optimizer, mode="min", factor=0.5, patience=4
    )

    final_model.fit(
        op=final_optimizer,
        criterion=final_criterion,
        scheduler=final_scheduler,
        train_loader=final_train_loader,
        val_loader=final_val_loader,
        epochs=40,
        root_path=root_path,
        device=device,
        patience=10,
    )

    loaders_dict = {
        "train": final_train_loader,
        "val": final_val_loader,
        "test": final_test_loader,
    }

    summary_table = final_model.evaluate_all(
        loaders=loaders_dict, save_path=root_path, device=device
    )

    print(f"\nИтоговая таблица метрик для {exp_name}:")
    print(summary_table.to_string())
