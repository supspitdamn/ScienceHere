import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import pandas as pd
from tqdm import tqdm
import os
import optuna
import json

def chunk_split(df: pd.DataFrame, strat: str, group_cols: list[str], target_cols: list[str], train_size: float, random_seed: int = 67):
    df = df.copy()

    all_cols = list(set(group_cols + [strat]))
    chunk_metadata = df[all_cols].drop_duplicates().reset_index(drop=True)
    
    if train_size == 1.0:
        df_train = df.reset_index(drop=True)
        df_temp = None
    else:
        train_ids, temp_ids = train_test_split(
            chunk_metadata,
            train_size=train_size,
            stratify=chunk_metadata[strat],  
            random_state=random_seed
        )

        train_ids_clean = train_ids[all_cols].drop_duplicates()
        temp_ids_clean = temp_ids[all_cols].drop_duplicates()

        df_train = df.merge(train_ids_clean, on=all_cols, how='inner').reset_index(drop=True)
        df_temp = df.merge(temp_ids_clean, on=all_cols, how='inner').reset_index(drop=True)

    return df_train, df_temp

def objective(trial, current_features, targets_cols, df_train, df_val,  device, root_path):

    sequence_length = trial.suggest_int("sequence_length", 5, 30, step=5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = 32

    train_dataset = RobotDataset(df_train, sequence_length, current_features, targets_cols)
    val_dataset = RobotDataset(df_val, sequence_length, current_features, targets_cols)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              pin_memory=True,
                              drop_last=True)
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=512, 
                            shuffle=False)
    
    actual_input_dim = len(all_features)

    model = ROBLSTM(
        input_dim=actual_input_dim ,
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

import numpy as np
import torch
from torch.utils.data import Dataset

class RobotDataset(Dataset):
    def __init__(self, grouped_df, sequence_length, feature_cols, target_cols):
        self.sequence_length = sequence_length
        self.features = grouped_df[feature_cols].values.astype(np.float32)
        self.target = grouped_df[target_cols].values.astype(np.float32)
        
        chunk_keys = grouped_df["unique_chunk_key"].values
        self.valid_indices = []
        
        for i in range(0, len(chunk_keys) - sequence_length):
            if chunk_keys[i] == chunk_keys[i + sequence_length]:
                self.valid_indices.append(i)
        
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        x = self.features[start_idx:end_idx]
        y = self.target[end_idx]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

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
                
                # if epoch == 0:
                    # print(f"Длина последовательности {len(x[0])}. Количество признаков {len(x[0, 0])}")
                    # print(f"X : {x[0, 0, :]}")
                    # print(f"Y : {y[0, :]}")

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
        plt.tight_layout()
        plt.legend(loc = "best")

        plt.subplot(1, 2, 2)
        plt.title("Изменение шага обучения")
        plt.step(range(epochs)[:len(lr_change)], lr_change, label = "Шаг обучения")
        plt.xlabel("Эпохи")
        plt.tight_layout()
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
                
                data_dict[loader_name] = {
                    ("X", "MSE"): mse[0], ("X", "MAE"): mae[0], ("X", "R2"): r2[0]
                    # ("У", "MSE"): mse[1], ("У", "MAE"): mae[1], ("У", "R2"): r2[1],
                    # ("Фи", "MSE"): mse[2], ("Фи", "MAE"): mae[2], ("Фи", "R2"): r2[2]
                }

        summary_df = pd.DataFrame.from_dict(data_dict, orient="index")
        summary_df.index.name = "Набор данных"

        summary_df.to_csv(os.path.join(save_path, "FINAL_metrics_all.csv"), encoding="utf-8-sig")
        
        return summary_df

if __name__ == "__main__":

    # Подготовка данных
    df = pd.read_csv(r"RNN\Phi\robot_data_with_chunks.csv")

    group_cols = ["surf_copy", "speedamp", "movedir"]
    full_group_cols = group_cols + ["unique_chunk_key"]

    targets_cols = ["ypos"]
    
    home_folder = r"C:\Users\User\Documents\MyPythonProjects\inputNN\RNN\Y"

    os.makedirs(home_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Обучение на {device}")

    # Функция деления по чанкам
    df_train, df_temp = chunk_split(df = df,
                                    strat = "surf_copy",
                                    group_cols = full_group_cols,
                                    target_cols = targets_cols,
                                    train_size = 0.7)

    df_val, df_test = chunk_split(df = df_temp,
                                    strat = "surf_copy",
                                    group_cols = full_group_cols,
                                    target_cols = targets_cols,
                                    train_size = 0.5)

    plt.figure(figsize=(10, 6))

    deltas = ["vx", "vy", "omega"]
    speeds = ["m1vel", "m2vel", "m3vel"]
    slips = ["w1slip", "w2slip", "w3slip"]
    currents = ["m1cur", "m2cur", "m3cur"]
    surfaces = ["type_brown", "type_gray", "type_green", "type_table"]
    coords_inputs = ["xpos", "ypos", "sin(ang)", "cos(ang)"]

    columns_to_standardize = deltas + speeds + slips + currents

    all_features = columns_to_standardize + surfaces + coords_inputs

    SC_X = StandardScaler()

    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    df_train_scaled[columns_to_standardize] = SC_X.fit_transform(df_train[columns_to_standardize])
    df_val_scaled[columns_to_standardize]   = SC_X.transform(df_val[columns_to_standardize])
    df_test_scaled[columns_to_standardize]  = SC_X.transform(df_test[columns_to_standardize])

    feature_expirements = {
        "Full_Context_with_Environments_And_Global_Coords_REMAKE": all_features
    }

    for exp_name, current_features in feature_expirements.items():

        root_path = os.path.join(home_folder, exp_name)

        os.makedirs(root_path, exist_ok=True)

        study = optuna.create_study(direction="minimize")

        study.optimize(
            lambda trial: objective(
                trial, current_features, targets_cols, df_train_scaled, df_val_scaled, device, root_path
            ),
            n_trials=10,
        )

        print("\n" + "=" * 50)
        print(f"ПОДБОР ЗАВЕРШЕН ДЛЯ ЭКСПЕРИМЕНТА: {exp_name}")
        print(f"Лучший достигнутый Val Loss: {study.best_value:.6f}")
        print("Лучшие параметры архитектуры:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("=" * 50)

        best = study.best_params

        best_config_meta = {
            "experiment_name": exp_name,
            "best_val_loss": study.best_value,
            "input_features": current_features,
            "target_columns": targets_cols,
            "hyperparameters": best,
        }

        with open(
            os.path.join(root_path, "best_model_params.json"),
            "w",
            encoding="utf-8",
        ) as json_file:
            json.dump(best_config_meta, json_file, ensure_ascii=False, indent=4)


        final_train_dataset = RobotDataset(
            df_train_scaled, best["sequence_length"], current_features, targets_cols
        )
        final_val_dataset = RobotDataset(
            df_val_scaled, best["sequence_length"], current_features, targets_cols
        )
        final_test_dataset = RobotDataset(
            df_test_scaled, best["sequence_length"], current_features, targets_cols
        )

        final_train_loader = DataLoader(
            final_train_dataset,
            batch_size=32,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        final_val_loader = DataLoader(
            final_val_dataset,
              batch_size=512,
                shuffle=False,
                pin_memory=True
        )
        final_test_loader = DataLoader(
            final_test_dataset,
            batch_size=512,
            shuffle=False,
            pin_memory=True
        )

        final_model = ROBLSTM(
            input_dim=len(all_features),
            hidden_dim=best["hidden_dim"],
            output_dim=len(targets_cols),
            num_layers=best["num_layers"],
            dropout=best["dropout"],
        )
        final_model.to(device)

        final_optimizer = torch.optim.Adam(final_model.parameters(), best["lr"])
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
            epochs=100,
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
