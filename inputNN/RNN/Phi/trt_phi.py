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
from RNN.trt import chunk_split, RobotDataset, ROBLSTM

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

if __name__ == "__main__":

    # Подготовка данных
    df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\УИРС\SEM5\filtered_robot_data_csv.csv", encoding="cp1251", sep = ";")

    home_folder = r"C:\Users\User\Documents\MyPythonProjects\inputNN\RNN\Phi"

    os.makedirs(home_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Обучение на {device}")

    print(df.info())

    df.columns = [column.strip() for column in df.columns]

    cols_to_convert = ["xcur", "ycur", "ang", "m1setvel", "m2setvel", "m3setvel", "m1pos", "m2pos", "m3pos"]

    # Конвертация того, что не должно быть строкой
    for col in cols_to_convert:
        if col in df.columns:

            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.')

            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # One Hot Encoding
    if 'surf' in df.columns and df['surf'].dtype == 'str':
        df["surf_copy"] = df["surf"].copy()
        df = pd.get_dummies(df, columns=['surf'], prefix='type', dtype=int)

    df["ang"] = df["ang"] * np.pi / 180
    df["ang"] = np.where(df["ang"] < np.pi, df["ang"], df["ang"] - 2 * np.pi)

    df["cos(ang)"] = np.cos(df["ang"])
    df["sin(ang)"] = np.sin(df["ang"])

    # print(df.info())

    # Распределение speedamp
    plt.hist(df["ang"], align = "mid")
    plt.title("Distribution [ang]")
    plt.xlabel("value")
    plt.ylabel("freq")
    # plt.show()

    df = df.sort_index()

    group_cols = ["surf_copy", "speedamp", "movedir"]

    # Группируем эксперименты
    df_grouped = df.groupby(by = group_cols)

    # # Пример группы
    # print(df_grouped.get_group(("table", 0.1, 0)))

    # Внутри каждой группы выделяем сессии (от 0 до n сек)
    df["session_id"] = df_grouped["t"].transform(lambda x : (x.diff() < 0)).cumsum()

    # Размер чанка (подможество сессий)
    CHUNK_SIZE = 300
    # В каждой сессии чанки начинаются с 0 до m
    df["chunk_id"] = df.groupby("session_id").cumcount() // CHUNK_SIZE

    # Уникальный ключ чанка
    df["unique_chunk_key"] = df["session_id"].astype(str) + "_" + df["chunk_id"].astype(str)

    targets_cols = ['sin(ang)', 'cos(ang)']

    df_processed = df.copy()

    # # группируем по чанкам. Внутри каждой группы чанков выбираем первый элемент (таргеты) и центрируем относительно значения t = 0
    # chunk_first = df_processed.groupby("unique_chunk_key", sort = False)[targets_cols].transform("first")
    # df_processed[targets_cols] = df_processed[targets_cols] - chunk_first

    df_processed.to_csv(os.path.join(home_folder, "robot_data_with_chunks.csv"), encoding="utf-8-sig")

    # print(df_processed[["movedir", "speedamp", "t", "xpos", "ypos", "ang"]].head(15))

    full_group_cols = group_cols + ["unique_chunk_key", "surf_copy"]

    # Функция деления по чанкам
    df_train, df_temp = chunk_split(df = df_processed,
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

        final_optimizer = torch.optim.Adam(final_model.parameters(), 0.001)
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
