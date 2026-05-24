import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import os
import optuna
import datetime
import json
from tqdm import tqdm

class ResBlock(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.LeakyReLU(0.01)

        self.ln2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x

        out = self.linear1(self.ln1(x))
        out = self.relu(out)
        out = self.linear2(self.ln2(out))
        
        out += identity
        return self.relu(out)

class MLP_RES(nn.Module):

    def __init__(self, in_features, hidden_dim, num_blocks, out_features):

        super().__init__()
        self.input_norm = nn.LayerNorm(in_features)
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.01)
        )
        self.struct = [in_features, hidden_dim, num_blocks, out_features]

        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(hidden_dim, out_features)
    
    def forward(self, vec: torch.Tensor):

        vec = vec.float()

        vec = self.input_norm(vec)

        vec = self.in_proj(vec)

        for block in self.res_blocks:

            vec = block(vec)

        return self.out_proj(vec)

    def teaching(self, epochs: int, op: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, train_loader: DataLoader, val_loader: DataLoader, save_path: str, model_state_dict: dict, verbose: bool = True, patience: int = 15, loss_func = None, lr: float = None, batch_size: int = None) -> float:

        device = next(self.parameters()).device

        log_txt_path = os.path.join(save_path, "LOG.txt")
        trigger = 0

        best_val_loss = float('inf') 
        train_loss_res = []
        val_loss_res = []
        lr_history = []

        with open(log_txt_path, "w", encoding="utf-8") as log_file:

            log_file.write(f"Обучаемая сеть: {type(self).__name__}, стукутра: {"->".join(map(str,self.struct))}, оптимизатор - {type(op).__name__}, шаг обучения - {lr}, функция потерь - {type(loss_func).__name__}, Эпох обучения: {epochs}, Размер батча: {batch_size}\n")

            for _ in range(0, epochs):

                train_epoch_loss = 0
                val_epoch_loss = 0

                self.train()

                train_bar = tqdm(train_loader)

                for x, y in train_bar:

                    x, y = x.to(device), y.to(device)

                    res = self(x)

                    loss = loss_func(res, y)

                    op.zero_grad()
                    loss.backward()
                    op.step()

                    train_epoch_loss += loss.item()

                    train_bar.set_postfix({"Epoch [Обучение]" : f"{_+1}/{epochs}", "Loss" : f"{loss.item():.2f}"})
                
                avg_loss = train_epoch_loss/len(train_loader)

                train_loss_res.append(avg_loss)

                self.eval()

                val_bar = tqdm(val_loader)

                for x, y in val_bar:

                    x, y = x.to(device), y.to(device)

                    with torch.no_grad():

                        res = self(x)
                        loss = loss_func(res, y)
                        val_epoch_loss += loss.item()
                    
                    val_bar.set_postfix({"Epoch [Валидация]" : f"{_+1}/{epochs}", "Loss" : f"{loss.item():.2f}"})
                
                avg_val_loss = val_epoch_loss/len(val_loader)
                val_loss_res.append(avg_val_loss)

                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    trigger = 0 

                    model_state_dict["model"] = self.state_dict()
                    model_state_dict["optimizer"] = op.state_dict()
                    file_path = os.path.join(save_path, "MLPconfig.pth")
                    torch.save(model_state_dict, file_path)
                    
                    log_file.write(f"-> Лучшая модель сохранена на эпохе {_} с лоссом {avg_val_loss:.6f}\n")
                else:
                    trigger += 1

                lr_history.append(op.param_groups[0]['lr'])
                log_str = f"Эпоха: {_}, Лосс обучения: {avg_loss}, Лосс валидации: {avg_val_loss}. Шаг обучения : {op.param_groups[0]['lr']}\n"

                if _ % 5 == 0:
                    log_file.write(log_str)

                if _ % 20 == 0 and verbose:

                    print(f"Эпоха: {_}, Лосс обучения: {avg_loss}")
                    print(f"Эпоха: {_}, Лосс валидации: {avg_val_loss}")
                
                if trigger == patience:

                    log_file.write(f"Остановка алг. Early Stopping\n")
                    log_file.write(log_str)
                    print(f"Останов. Эпоха : {_}, Лосс обучения: {avg_loss}")
                    print(f"Останов. Эпоха : {_}, Лосс валидации: {avg_val_loss}")
                    break

            else:
                log_file.write(f"Штатный останов на эпохе {_}\n")
                log_file.write(f"Лосс обучения: {avg_loss},  Лосс валидации: {avg_val_loss}\n")
                print(f"Запланированный конец обучения. Эпоха : {_}, Лосс обучения: {avg_loss}")
                print(f"Запланированный конец обучения. Эпоха : {_}, Лосс валидации: {avg_val_loss}")
            
            
        # График Loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_res, color="blue", label="Обучение")
        plt.plot(val_loss_res, color="red", label="Валидация")
        plt.legend(); plt.grid(True); plt.title("Кривые обучения")
        plt.savefig(os.path.join(save_path, "Loss_MLP.png"), dpi=300, bbox_inches="tight")
        if verbose: plt.show()
        plt.close()

        # График Learning Rate
        plt.figure(figsize=(10, 5))
        plt.plot(lr_history, color="green")
        plt.title(f"Изменение шага обучения (LR). Последний: {op.param_groups[0]["lr"]}")
        plt.xlabel("Эпохи"); plt.ylabel("LR"); plt.grid(True)
        plt.yscale('log')
        plt.savefig(os.path.join(save_path, "LR_history.png"), dpi=300, bbox_inches="tight")
        if verbose: plt.show()
        plt.close()
    
        return best_val_loss
    
    def evaluate(self, data_loader: DataLoader, save_path: str, name: str, device: str = "cpu") -> dict:

        all_pred = []
        all_true = []
        self.eval()
        for x, y in tqdm(data_loader, "Оценка результатов: "):

            x, y = x.to(device), y.to(device)

            with torch.no_grad():

                predict = self.forward(x).detach().cpu().numpy()
                true_value = y.detach().cpu().numpy()

                all_true.append(true_value)
                all_pred.append(predict)

        all_pred = np.vstack(all_pred)
        all_true = np.vstack(all_true)

        with open(os.path.join(save_path, "LOG.txt"), "a", encoding="utf-8") as log_txt:

            mse = mean_squared_error(all_pred, all_true, multioutput="raw_values")
            mae = mean_absolute_error(all_pred, all_true, multioutput="raw_values")
            r2 = r2_score(all_true, all_pred, multioutput="raw_values")
            mape = mean_absolute_percentage_error(all_true, all_pred, multioutput="raw_values")
            
            if name == "test":
                log_txt.write(20*"-"+"\n")
                log_txt.write("Результаты для тестовой выборки:\n")
                log_txt.write(f"Абсолютная ошибка (Дельта Х, Дельта У, Дельта Фи): {'  '.join(map(str, np.round(mae, 4)))}\n")

        return {"MSE": tuple(mse), "MAE" : tuple(mae), "R2" : tuple(r2), "MAPE" : tuple(mape)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Обучение на {device}")

df_raw = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\УИРС\\SEM5\\filtered_robot_data.csv", encoding="cp1251", sep=";")

cols_to_use = ["m1vel", "m2vel", "m3vel", "surf", "m1cur", "m2cur", "m3cur", "w1slip", "w2slip", "w3slip"]
df_base = df_raw[cols_to_use].dropna().reset_index(drop=True)

train_raw, temp_raw = train_test_split(df_base, test_size=0.2, random_state=42, shuffle=True, stratify=df_base["surf"])
val_raw, test_raw = train_test_split(temp_raw, test_size=0.5, random_state=42, shuffle=True, stratify=temp_raw["surf"])

df_processed = pd.get_dummies(df_base, columns=["surf"], prefix="type").astype(float)

train = df_processed.loc[train_raw.index]
val = df_processed.loc[val_raw.index]
test = df_processed.loc[test_raw.index]

filter_cols = ["m1cur", "m2cur", "m3cur", "w1slip", "w2slip", "w3slip"]
train = train[(train[filter_cols] > 1e-2).all(axis=1)]
val = val[(val[filter_cols] > 1e-2).all(axis=1)]
test = test[(test[filter_cols] > 1e-2).all(axis=1)]

surf_features = [col for col in df_processed.columns if col.startswith("type_")]
features = ["m1cur", "m2cur", "m3cur", "m1vel", "m2vel", "m3vel"] + surf_features
targets = ["w1slip", "w2slip", "w3slip"]

print(f"После фильтрации: {len(train)}. До фильтрации: {df_processed.shape[0]}. Потери: {(1-len(train)/df_processed.shape[0])*100:.2f}%")

x_train = torch.tensor(train[features].values, dtype=torch.float32)
y_train = torch.tensor(train[targets].values, dtype=torch.float32)

x_val = torch.tensor(val[features].values, dtype=torch.float32)
y_val = torch.tensor(val[targets].values, dtype=torch.float32)

x_test = torch.tensor(test[features].values, dtype=torch.float32)
y_test = torch.tensor(test[targets].values, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=512, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=512, shuffle=False)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
root_path = f".//Currents_to_Slippage//MLP_RES_study_{timestamp}"
os.makedirs(root_path, exist_ok=True)

model = MLP_RES(10, 128, 3, 3).to(device)

op = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=9.55638933553592e-09)
loss = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op, factor=0.5, patience=10)

model_state_dict = {"model": None, "optimizer": None}

# model.teaching(400, op, scheduler, train_loader, val_loader, root_path, 
#                model_state_dict=model_state_dict, patience=25, loss_func=loss, verbose=True, lr=1e-3, batch_size=128)

st = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\Currents_to_Slippage\MLP_RES_study_20260524_101812\MLPconfig.pth")
model.load_state_dict(st["model"])

data = {"train": train_loader, "val": val_loader, "test": test_loader}

for key, value in data.items():

    res = model.evaluate(value, name=key, save_path=root_path, device=device)

    metrics_df = pd.DataFrame(res, index=["Двигатель 1", "Двигатель 2", "Двигатель 3"]).T

    metrics_df.loc["MAPE, %"] = metrics_df.loc["MAPE"]*100
    metrics_df.drop("MAPE", inplace=True)

    table_path = os.path.join(root_path, f"FINAL_MLP_metrics_{key}.csv")

    metrics_df.to_csv(table_path, index_label="Метрики", encoding="utf-8-sig")
