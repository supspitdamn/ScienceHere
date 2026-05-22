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

                for x, y in tqdm(train_loader, "Тренировка"):

                    x, y = x.to(device), y.to(device)

                    res = self(x)

                    loss = loss_func(res, y)

                    op.zero_grad()
                    loss.backward()
                    op.step()

                    train_epoch_loss += loss.item()
                
                avg_loss = train_epoch_loss/len(train_loader)

                train_loss_res.append(avg_loss)

                self.eval()

                for x, y in tqdm(val_loader,  "Валидация"):

                    x, y = x.to(device), y.to(device)

                    with torch.no_grad():

                        res = self(x)
                        loss = loss_func(res, y)
                        val_epoch_loss += loss.item()
                
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
            
            if name == "test":
                log_txt.write(20*"-"+"\n")
                log_txt.write("Результаты для тестовой выборки:\n")
                log_txt.write(f"Абсолютная ошибка (Дельта Х, Дельта У, Дельта Фи): {'  '.join(map(str, np.round(mae, 4)))}\n")

        return {"MSE": tuple(mse), "MAE" : tuple(mae), "R2" : tuple(r2)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Обучение на {device}")

df_raw = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\УИРС\\SEM5\\filtered_robot_data.csv", encoding="cp1251", sep=";")

for col in ["m1setvel", "m2setvel", "m3setvel", "m1vel", "m2vel", "m3vel"]:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
df_raw = df_raw.dropna(subset=["m1setvel", "m2setvel", "m3setvel", "m1vel", "m2vel", "m3vel"]).reset_index(drop=True)

train_raw, temp_raw = train_test_split(df_raw, test_size=0.2, random_state=42, shuffle=True, stratify=df_raw["surf"])
val_raw, test_raw = train_test_split(temp_raw, test_size=0.5, random_state=42, shuffle=True, stratify=temp_raw["surf"])

def prepare_split(df_split):
    surf_cols = ["surf"]
    vel_cols = ["m1vel", "m2vel", "m3vel"]
    setvel_cols = ["m1setvel", "m2setvel", "m3setvel"]

    df_vel = df_split.melt(id_vars=surf_cols, value_vars=vel_cols, value_name="mvel")
    df_setvel = df_split.melt(value_vars=setvel_cols, value_name="msetvel")

    final_split = pd.concat([df_vel[["mvel", "surf"]], df_setvel["msetvel"]], axis=1)
    final_split = pd.get_dummies(final_split, columns=["surf"], prefix="type", prefix_sep="_")
    return final_split

train = prepare_split(train_raw)
val = prepare_split(val_raw)
test = prepare_split(test_raw)

features = ["msetvel", "type_brown", "type_gray", "type_green", "type_table"]
targets = ["mvel"]

x_train = torch.tensor(train[features].values.astype(np.float32), dtype=torch.float32).to(device)
y_train = torch.tensor(train[targets].values.astype(np.float32), dtype=torch.float32).to(device)

x_val = torch.tensor(val[features].values.astype(np.float32), dtype=torch.float32).to(device)
y_val = torch.tensor(val[targets].values.astype(np.float32), dtype=torch.float32).to(device)

x_test = torch.tensor(test[features].values.astype(np.float32), dtype=torch.float32).to(device)
y_test = torch.tensor(test[targets].values.astype(np.float32), dtype=torch.float32).to(device)

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
root_path = f".//SetVelocity_to_RealVelocity//MLP_study_{timestamp}"
os.makedirs(root_path, exist_ok=True)

model = MLP_RES(5, 32, 3, 1).to(device)

op = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-6)
loss = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op, factor=0.5, patience=10)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

model_state_dict = {"model": None, "optimizer": None}

model.teaching(200, op, scheduler, train_loader, val_loader, root_path, 
               model_state_dict=model_state_dict, patience=25, loss_func=loss, verbose=True, lr=1e-3, batch_size=128)

data = {"train": train_loader, "val": val_loader, "test": test_loader}

for key, value in data.items():
    res = model.evaluate(value, name=key, save_path=root_path, device=device)
    metrics_df = pd.DataFrame(res, index=["Общая по двигателям"]).T
    table_path = os.path.join(root_path, f"FINAL_MLP_metrics_{key}.csv")
    metrics_df.to_csv(table_path, index_label="Метрики", encoding="utf-8-sig")
