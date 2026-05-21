import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import os
import datetime
import json
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Обучение на {device}")

df = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\УИРС\\SEM5\\filtered_robot_data.csv", encoding="cp1251", sep=";")

df = df.query("w1slip <= 1 and w2slip <= 1 and w3slip <= 1")

df["m1setvel"] = pd.to_numeric(df["m1setvel"], errors="coerce")
df["m2setvel"] = pd.to_numeric(df["m2setvel"], errors="coerce")
df["m3setvel"] = pd.to_numeric(df["m3setvel"], errors="coerce")
df = df.dropna(subset=["m1setvel", "m2setvel", "m3setvel"])

cols_to_keep = ["m1setvel", "m2setvel", "m3setvel", "m1cur", "m2cur", "m3cur", "surf", 
                "m1vel", "m2vel", "m3vel", "w1slip", "w2slip", "w3slip", "vx", "vy", "omega"]
df = df[cols_to_keep]

df = pd.get_dummies(df, columns=["surf"], prefix="type", prefix_sep="_")

train, temp = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df["surf"] if "surf" in df.columns else None)
val, test = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True, stratify=None)

features = ["m1setvel", "m2setvel", "m3setvel", "type_brown", "type_gray", "type_green", "type_table"]
targets_extra = ["w1slip", "w2slip", "w3slip", "m1vel", "m2vel", "m3vel", "m1cur", "m2cur", "m3cur"]
targets = ["vx", "vy", "omega"]

x_train = torch.tensor(train[features].values.astype(np.float32), dtype=torch.float32)
y_train = torch.tensor(train[targets + targets_extra].values.astype(np.float32), dtype=torch.float32)

x_val = torch.tensor(val[features].values.astype(np.float32), dtype=torch.float32)
y_val = torch.tensor(val[targets + targets_extra].values.astype(np.float32), dtype=torch.float32)

x_test = torch.tensor(test[features].values.astype(np.float32), dtype=torch.float32)
y_test = torch.tensor(test[targets + targets_extra].values.astype(np.float32), dtype=torch.float32)

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

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
    
    @staticmethod

    def objective(trial, input_size, output_dim, train_dataset, val_dataset, root_path, device = "cpu")->float:

        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        hidden_size = trial.suggest_int("hidden_size", 16, 128, step=16)
        b_size = trial.suggest_int("batch_size", 16, 128, step=16)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log = True)

        layers_struct = [input_size] + [hidden_size]*num_layers + [output_dim]

        trial_model = MLP(*layers_struct).to(device=device)

        trial_op = torch.optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
        trial_loss = nn.MSELoss()
        trial_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=trial_op, factor = 0.5, patience = 10)

        v_load = DataLoader(val_dataset, 
                            batch_size=512,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)
        
        t_load = DataLoader(train_dataset, 
                            batch_size=b_size, 
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True)

        save_path = os.path.join(root_path, f"MLP_{trial.number}_{"-".join(map(str, trial_model.struct))}_{type(trial_op).__name__}_{lr}_{type(trial_loss).__name__}_Batch_{b_size}")
        os.makedirs(save_path, exist_ok = True)

        best_val_loss = trial_model.teaching(
            epochs=100, 
            train_loader=t_load, 
            val_loader=v_load, 
            model_state_dict={},
            op=trial_op,
            loss_func=trial_loss,
            scheduler = trial_scheduler,
            lr=lr,
            batch_size=b_size,
            save_path=save_path,
            verbose=False,
            patience=30
        )

        return best_val_loss

class MLP_NORM(nn.Module):

    def __init__(self, *args):

        super().__init__()
        self.struct = args
        self.__layers = nn.ModuleList()

        self.input_norm = nn.LayerNorm(args[0])

        for i in range(len(args) - 1):

            self.__layers.append(nn.Linear(args[i], args[i+1]))

            if i < len(args) - 2:

                self.__layers.append(nn.ReLU())
    
    def forward(self, vec: torch.Tensor):

        vec = vec.float()

        vec = self.input_norm(vec)

        for layer in self.__layers:

            vec = layer(vec)

        return vec

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
    
    @staticmethod

    def objective(trial, input_size, output_dim, train_dataset, val_dataset, root_path, device = "cpu")->float:

        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        hidden_size = trial.suggest_int("hidden_size", 16, 128, step=16)
        b_size = trial.suggest_int("batch_size", 16, 128, step=16)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log = True)

        layers_struct = [input_size] + [hidden_size]*num_layers + [output_dim]

        trial_model = MLP(*layers_struct).to(device=device)

        trial_op = torch.optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
        trial_loss = LogCoshLoss()
        trial_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=trial_op, factor = 0.5, patience = 10)

        v_load = DataLoader(val_dataset, 
                            batch_size=512,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)
        
        t_load = DataLoader(train_dataset, 
                            batch_size=b_size, 
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True)

        save_path = os.path.join(root_path, f"MLP_{trial.number}_{"-".join(map(str, trial_model.struct))}_{type(trial_op).__name__}_{lr}_{type(trial_loss).__name__}_Batch_{b_size}")
        os.makedirs(save_path, exist_ok = True)

        best_val_loss = trial_model.teaching(
            epochs=100, 
            train_loader=t_load, 
            val_loader=v_load, 
            model_state_dict={},
            op=trial_op,
            loss_func=trial_loss,
            scheduler = trial_scheduler,
            lr=lr,
            batch_size=b_size,
            save_path=save_path,
            verbose=False,
            patience=30
        )

        return best_val_loss

class MLP(nn.Module):

    def __init__(self, *args):

        super().__init__()
        self.struct = args
        self.__layers = nn.ModuleList()

        for i in range(len(args) - 1):

            self.__layers.append(nn.Linear(args[i], args[i+1]))

            if i < len(args) - 2:

                self.__layers.append(nn.ReLU())
    
    def forward(self, vec: torch.Tensor):

        for layer in self.__layers:

            vec = layer(vec)

        return vec

    def teaching(self, epochs: int, op: torch.optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader, save_path: str, model_state_dict: dict, verbose: bool = True, patience: int = 15, loss_func = None, lr: float = None, batch_size: int = None) -> float:

        device = next(self.parameters()).device
        best_val_loss = float("inf")

        log_txt_path = os.path.join(save_path, "LOG.txt")
        trigger = 0

        epochs = range(epochs)

        train_loss_res = []
        val_loss_res = []

        with open(log_txt_path, "w", encoding="utf-8") as log_file:

            log_file.write(f"Обучаемая сеть: {type(self).__name__}, стукутра: {"->".join(map(str,self.struct))}, оптимизатор - {type(op).__name__}, шаг обучения - {lr}, функция потерь - {type(loss_func).__name__}, Эпох обучения: {epochs}, Размер батча: {batch_size}\n")

            for _ in epochs:

                train_epoch_loss = 0
                val_epoch_loss = 0

                self.train()

                for x, y in train_loader:

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

                for x, y in val_loader:

                    x, y = x.to(device), y.to(device)

                    with torch.no_grad():

                        res = self(x)
                        loss = loss_func(res, y)
                        val_epoch_loss += loss.item()
                
                avg_val_loss = val_epoch_loss/len(val_loader)
                val_loss_res.append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    trigger = 0

                    log_file.write(f"->Модель сохранена на эпохе {_}\n")
                    model_state_dict["model"] = self.state_dict()
                    model_state_dict["optimizer"] = op.state_dict()
                    model_state_dict["epoch"] = _
                    file_path = os.path.join(save_path, f"MLPconfig.pth")
                    torch.save(model_state_dict, file_path)
                else:
                    trigger += 1

                
                log_str = f"Эпоха: {_}, Лосс обучения: {avg_loss}, Лосс валидации: {avg_val_loss}\n"

                if _ % 5 == 0:
                    log_file.write(log_str)

                if _ % 20 == 0 and verbose:

                    print(f"Эпоха: {_}, Лосс обучения: {avg_loss}")
                    print(f"Эпоха: {_}, Лосс валидации: {avg_val_loss}")
                
                if trigger >= patience:

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
            
        plt.plot(range(len(train_loss_res)), train_loss_res, color="blue", label = "Обучение")
        plt.plot(range(len(val_loss_res)), val_loss_res, color = "red", label = "Валидация")

        plt.legend()
        plt.title(f"Функция потерь {type(self).__name__}_{'-'.join(map(str, self.struct))}_{type(op).__name__}_{lr}_{type(loss_func).__name__}")
        plt.xlabel("Эпохи обучения")
        plt.ylabel(f"Лосс {type(loss_func).__name__}")
        plt.grid(visible=True)

        png_path = os.path.join(save_path, f"Loss_MLP.png")
        plt.savefig(png_path, dpi = 300, bbox_inches = "tight")

        if verbose:
            plt.show()
        
        plt.close()
        
        return min(val_loss_res)
    
    def evaluate(self, data_loader: DataLoader, scaler_y : StandardScaler, save_path: str, name: str, device: str = "cpu") -> dict:

        all_pred = []
        all_true = []
        self.eval()
        for x, y in data_loader:

            x, y = x.to(device), y.to(device)

            with torch.no_grad():

                predict = self.forward(x).detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                predict = (scaler_y.inverse_transform(predict))
                true_value = (scaler_y.inverse_transform(y))

                all_true.append(true_value)
                all_pred.append(predict)

        all_pred = np.vstack(all_pred)
        all_true = np.vstack(all_true)

        with open(os.path.join(save_path, "LOG.txt"), "w", encoding="utf-8") as log_txt:

            mse = mean_squared_error(all_pred, all_true, multioutput="raw_values")
            mae = mean_absolute_error(all_pred, all_true, multioutput="raw_values")
            mape = mean_absolute_percentage_error(all_pred, all_true, multioutput="raw_values")
            r2 = r2_score(all_true, all_pred, multioutput="raw_values")
            
            if name == "test":
                log_txt.write(20*"-"+"\n")
                log_txt.write("Результаты для тестовой выборки:\n")
                log_txt.write(f"Абсолютная ошибка (M1, M2, M3): {'  '.join(map(str, np.round(mae, 4)))}\n")
                log_txt.write(f"Относительная ошибка (M1, M2, M3): {'  '.join(map(str, np.round(mape * 100, 4)))}\n")

        return {"MSE": tuple(mse), "MAE" : tuple(mae), "MAPE" : tuple(mape), "R2" : tuple(r2)}
    
    @staticmethod

    def objective(trial, train_dataset, val_dataset, root_path, device = "cpu")->float:

        lr = trial.suggest_float("lr", 1e-4, 1e-2)
        num_layers = trial.suggest_int("num_layers", 1, 5)
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        b_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 0, 1e-3)

        layers_struct = [5] + [hidden_size]*num_layers + [1]

        trial_model = MLP(*layers_struct).to(device=device)

        trial_op = torch.optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
        trial_loss = nn.MSELoss()

        v_load = DataLoader(val_dataset, batch_size=b_size, shuffle=False)
        t_load = DataLoader(train_dataset, batch_size=b_size, shuffle=True)

        save_path = os.path.join(root_path, f"MLP_{trial.number}_{"-".join(map(str, trial_model.struct))}_{type(trial_op).__name__}_{lr}_{type(trial_loss).__name__}_Batch_{b_size}")
        os.makedirs(save_path, exist_ok = True)

        best_val_loss = trial_model.teaching(
            epochs=100, 
            train_loader=t_load, 
            val_loader=v_load, 
            model_state_dict={},
            op=trial_op,
            loss_func=trial_loss,
            lr=lr,
            batch_size=b_size,
            save_path=save_path,
            verbose=False,
            patience=10
        )

        return best_val_loss

class NPM(nn.Module):

    def __init__(self, seq_neur: list, device: str = "cuda", kaiman_weight_init = False) -> None:

        super().__init__()
        
        self.stage_1 = nn.ModuleList(seq_neur[0])
        self.stage_2 = seq_neur[1]
        self.stage_3 = seq_neur[2]
        self.stage_4 = seq_neur[3]

        if kaiman_weight_init:
            self.apply(self._init_kaiming)

        self.to(device)
    
    def _init_kaiming(self, module):

        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, vec: torch.Tensor):

        device = next(self.parameters()).device
        vec = vec.to(device).float()

        surfs = vec[:, 3:7]

        v1_s = self.stage_1[0](torch.cat((vec[:, 0:1], surfs), dim=1))
        v2_s = self.stage_1[1](torch.cat((vec[:, 1:2], surfs), dim=1))
        v3_s = self.stage_1[2](torch.cat((vec[:, 2:3], surfs), dim=1))

        v_st1 = torch.cat((v1_s, v2_s, v3_s), dim=1) # Полученные скорсти

        in_st2 = torch.cat((v_st1, surfs), dim=1)

        cur_st2 = self.stage_2(in_st2) # Полученные токи

        in_st3 = torch.concat((cur_st2, v_st1, surfs), dim=1)

        slip_st3 = self.stage_3(in_st3) # Полученные проскальзывания

        in_st4 = torch.cat((v_st1, slip_st3, surfs), dim = 1)

        delta_st3 = self.stage_4(in_st4) # Полученные дельта-координаты

        return delta_st3, slip_st3, cur_st2, v_st1

    def evaluate(self, data_loader: DataLoader, name: str, save_path: str, device: str = "cpu") -> dict:
        all_pred = []
        all_true = []
        self.eval()
        
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                pred_delta, pred_slip, pred_cur, pred_vel = self(x) 

                predict = torch.cat([pred_delta, pred_slip, pred_vel, pred_cur], dim=1).cpu().numpy()
                true_value = y.cpu().numpy()

                all_true.append(true_value)
                all_pred.append(predict)

        all_pred = np.vstack(all_pred)
        all_true = np.vstack(all_true)

        mse = mean_squared_error(all_true, all_pred, multioutput="raw_values")
        mae = mean_absolute_error(all_true, all_pred, multioutput="raw_values")
        r2 = r2_score(all_true, all_pred, multioutput="raw_values")

        with open(os.path.join(save_path, "LOG.txt"), "a", encoding="utf-8-sig") as log_txt:

            if name == "test":
                log_txt.write(20*"-"+"\n")
                log_txt.write("Результаты для тестовой выборки:\n")
                
                log_txt.write(f"MAE Delta (X, Y, Phi):   {'  '.join(map(str, np.round(mae[0:3], 4)))}\n")
                log_txt.write(f"MAE Slip (S1, S2, S3):    {'  '.join(map(str, np.round(mae[3:6], 4)))}\n")
                log_txt.write(f"MAE Скор. (V1, V2, V3):   {'  '.join(map(str, np.round(mae[6:9], 4)))}\n")
                log_txt.write(f"MAE Токи (M1, M2, M3):    {'  '.join(map(str, np.round(mae[9:12], 4)))}\n")

        return {
            "MSE": tuple(mse), 
            "MAE": tuple(mae), 
            "R2": tuple(r2)
        }


    def fit(self, optimizer, loss, scheduler, train_loader, val_loader, epochs, root_path, patience = 10):

        sum_train_losses = []
        sum_val_losses = []

        best_val_loss = float('inf')
        counter = 0
        best_model_path = os.path.join(root_path, "best_model.pth")

        with open(os.path.join(root_path, "LOG.txt"), "a", encoding = "utf-8") as log_txt:

            for idx in range(epochs):

                train_losses = []   
                val_losses = []

                self.train()

                train_tqdm = tqdm(train_loader, desc=f"Эпоха {idx+1}/{epochs} [Обучение]")
                val_tqdm = tqdm(val_loader, desc=f"Эпоха {idx+1}/{epochs} [Валидация]")

                for x, y in train_tqdm:

                    current_lr = optimizer.param_groups[0]['lr']

                    x, y = x.to(device), y.to(device)

                    pred_delta, pred_slip, pred_cur, pred_vel = self(x)

                    loss_delta = loss(pred_delta, y[:, 0:3])
                    loss_slip = loss(pred_slip, y[:, 3:6])
                    loss_vel = loss(pred_vel, y[:, 6:9])
                    loss_cur = loss(pred_cur, y[:, 9:])

                    summary_loss = loss_cur + loss_vel + loss_slip + loss_delta

                    train_losses.append(summary_loss.item())

                    optimizer.zero_grad()
                    summary_loss.backward()
                    optimizer.step()

                    train_tqdm.set_postfix(batch_loss=f"{summary_loss.item():.4f}", lr = f"{current_lr:.6f}")
                
                res_loss_train = sum(train_losses)/len(train_loader)
                sum_train_losses.append(res_loss_train)
                
                self.eval()

                for x, y in val_tqdm:

                    x, y = x.to(device), y.to(device)

                    with torch.no_grad():

                        pred_delta, pred_slip, pred_cur, pred_vel = self(x)

                        loss_delta = loss(pred_delta, y[:, 0:3])
                        loss_slip = loss(pred_slip, y[:, 3:6])
                        loss_vel = loss(pred_vel, y[:, 6:9])
                        loss_cur = loss(pred_cur, y[:, 9:])

                        summary_loss = loss_cur + loss_vel + loss_slip + loss_delta

                        val_losses.append(summary_loss.item())
                        val_tqdm.set_postfix(batch_loss=f"{summary_loss.item():.4f}")
                
                res_loss_val = sum(val_losses)/len(val_loader)
                sum_val_losses.append(res_loss_val)

                scheduler.step(res_loss_val)

                current_lr = optimizer.param_groups[0]['lr']

                epoch_log = (f"Эпоха {idx+1}/{epochs} | "
                             f"Train Loss: {res_loss_train:.6f} | "
                             f"Val Loss: {res_loss_val:.6f} | "
                             f"LR: {current_lr:.8f}\n")
                
                print(epoch_log)
                log_txt.write(epoch_log)
                log_txt.flush()

                if res_loss_val < best_val_loss:
                    best_val_loss = res_loss_val
                    counter = 0
                    torch.save(self.state_dict(), best_model_path)
                    msg = f"--- Найдена лучшая модель на эпохе {idx+1} (Loss: {best_val_loss:.6f}) ---\n"
                else:
                    counter += 1
                    msg = f"Терпение: {counter} из {patience}\n"

                print(msg)
                log_txt.write(msg)
                log_txt.flush()

                if counter >= patience:
                    stop_msg = f"Early stopping на эпохе {idx+1}. Возвращаемся к лучшим весам.\n"
                    print(stop_msg)
                    log_txt.write(stop_msg)
                    log_txt.flush()
                    self.load_state_dict(torch.load(best_model_path))
                    break 

        plt.figure(figsize=(10, 5))
        plt.plot(sum_train_losses, label='Лосс тренировки')
        plt.plot(sum_val_losses, label='Лосс валидации')
        plt.xlabel('Эпохи')
        plt.ylabel('Лосс')
        plt.title('Лосс тренировки и валидации')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(root_path, "training_res.png"), dpi=300)
        plt.close()

best_npm_par = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\neuro_physical_model\NPM_study_21052026_104059_FINE_TUNING_DELTA\best_model.pth")
best_npm_par_zero = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\neuro_physical_model\NPM_study_20260521_125352_ZERO_COND\best_model.pth")
best_npm_par_zero_res = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\neuro_physical_model\NPM_study_20260521_152100\best_model.pth")

best_par_sv_v = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\SetVelocity_To_RealVelocity\MLP_study_20260413_204155\MLP_11_5-32-32-32-32-32-1_Adam_0.0001735565808786231_MSELoss_Batch_32\MLPconfig.pth")
best_par_v_c = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\RealVelocity_To_Current\MLP_study_20260412_181402\MLP_34_7-64-64-64-64-3_Adam_0.0006238342122664613_MSELoss_Batch_64\MLPconfig.pth")
best_par_v_c_s__sl = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\Currents_to_Slippage\Seeking_for_best_features_05-05-2026_18-18-28\feat_count_m1cur_m2cur_m3cur_m1vel_m2vel_m3vel_surfs\MLP_31_10-64-64-64-3_Adam_0.0009816432611570356_MSELoss_Batch_64\MLPconfig.pth")
best_par_v_sl_s__delta = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\Slippage_to_DeltaCoords\Seeking_for_best_features_19-05-2026_08-52-39\feat_count_m1vel_m2vel_m3vel_w1slip_w2slip_w3slip_type__brown_type__gray_type__green_type__table\MLP_17_10-128-128-128-3_Adam_0.0019297113996488578_LogCoshLoss_Batch_128\MLPconfig.pth")

def get_base_mlps():
    m1 = MLP(5, 32, 32, 32, 32, 32, 1)
    m2 = MLP(5, 32, 32, 32, 32, 32, 1)
    m3 = MLP(5, 32, 32, 32, 32, 32, 1)
    mvc = MLP(7, 64, 64, 64, 64, 3)
    mvcsl = MLP(10, 64, 64, 64, 3)
    mvslsd = MLP_NORM(10, 128, 128, 128, 3)

    m1.load_state_dict(best_par_sv_v["model"])
    m2.load_state_dict(best_par_sv_v["model"])
    m3.load_state_dict(best_par_sv_v["model"])
    mvc.load_state_dict(best_par_v_c["model"])
    mvcsl.load_state_dict(best_par_v_c_s__sl["model"])

    old_state_dict = best_par_v_sl_s__delta["model"]
    new_state_dict = {}
    for key, value in old_state_dict.items():
        new_key = key.replace("_MLP__layers", "_MLP_NORM__layers")
        new_state_dict[new_key] = value

    mvslsd.load_state_dict(new_state_dict)
    return [m1, m2, m3], mvc, mvcsl, mvslsd

def get_norm_base_mlps():
    m1 = MLP_NORM(5, 32, 32, 32, 32, 32, 1)
    m2 = MLP_NORM(5, 32, 32, 32, 32, 32, 1)
    m3 = MLP_NORM(5, 32, 32, 32, 32, 32, 1)
    mvc = MLP_NORM(7, 64, 64, 64, 64, 3)
    mvcsl = MLP_NORM(10, 64, 64, 64, 3)
    mvslsd = MLP_NORM(10, 128, 128, 128, 3)
    return [m1, m2, m3], mvc, mvcsl, mvslsd

def get_res_base_mlps():
    # Параметры: (входные_фичи, скрытая_размерность, количество_резидуальных_блоков, выходы)
    m1 = MLP_RES(5, 32, 3, 1)      # 3 блока (6 линейных слоев)
    m2 = MLP_RES(5, 32, 3, 1)      
    m3 = MLP_RES(5, 32, 3, 1)      
    mvc = MLP_RES(7, 64, 3, 3)     # 3 блока (6 линейных слоев)
    mvcsl = MLP_RES(10, 64, 3, 3)  # 3 блока (6 линейных слоев)
    mvslsd = MLP_RES(10, 128, 4, 3) # 4 блока (8 линейных слоев) для самого глубокого узла

    return [m1, m2, m3], mvc, mvcsl, mvslsd

mlps_t, mvc_t, mvcsl_t, mvslsd_t = get_base_mlps()
npm_trained = NPM([mlps_t, mvc_t, mvcsl_t, mvslsd_t], device="cuda")
npm_trained.load_state_dict(best_npm_par)

mlps_s, mvc_s, mvcsl_s, mvslsd_s = get_base_mlps()
npm_simple = NPM([mlps_s, mvc_s, mvcsl_s, mvslsd_s], device="cuda")

mlps_blank, mvc_blank, mvcsl_blank, mvslsd_blank = get_norm_base_mlps()
npm_zero_cond = NPM([mlps_blank, mvc_blank, mvcsl_blank, mvslsd_blank], device="cuda", kaiman_weight_init=True)
npm_zero_cond.load_state_dict(best_npm_par_zero)

mlps_res_blank, mvc_s_res_blank, mvcsl_s_res_blank, mvslsd_res_blank = get_res_base_mlps()
npm_zero_cond_res = NPM([mlps_res_blank, mvc_s_res_blank, mvcsl_s_res_blank, mvslsd_res_blank], device = "cuda", kaiman_weight_init=True)
npm_zero_cond_res.load_state_dict(best_npm_par_zero_res)

# root_path = r"C:\Users\User\Documents\MyPythonProjects\inputNN\neuro_physical_model"
# folder_path = os.path.join(root_path, f"NPM_study_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}")
# os.makedirs(folder_path, exist_ok=True)

train_loader = DataLoader(train_dataset, batch_size=128, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, pin_memory=True)

op = torch.optim.Adam(params=npm_zero_cond_res.parameters(), lr=1e-3, weight_decay=1e-5)
loss = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op, factor=0.5, patience=10)

# npm_zero_cond_res.fit(op, loss, scheduler, train_loader, val_loader, epochs=200, root_path=folder_path, patience=25)

root_path = f".//neuro_physical_model//NPM_comparison"
os.makedirs(root_path, exist_ok=True)

data = {"test": test_loader}

columns_names = ["Дельта Х", "Дельта У", "Дельта Фи", "Проскальзывание М1", "Проскальзывание М2", "Проскальзывание М3", "Ток М1", "Ток М2", "Ток М3", "Скорость М1", "Скорость М2", "Скорость М3", "Сумма"]
rows_names = ["Склеенная", "Тренированная", "Обученная с нуля", "Обученная с нуля + остаточные связи", "Разность (Скл - Трен)"]

for key, value in data.items():
    res_t = list(npm_trained.evaluate(value, name=key, save_path=root_path, device=device).get("MAE", [0]*12))
    res_s = list(npm_simple.evaluate(value, name=key, save_path=root_path, device=device).get("MAE", [0]*12))
    res_z = list(npm_zero_cond.evaluate(value, name=key, save_path=root_path, device=device).get("MAE", [0]*12))
    res_z_res = list(npm_zero_cond_res.evaluate(value, name=key, save_path=root_path, device=device).get("MAE", [0]*12))

    res_t.append(sum(res_t))
    res_s.append(sum(res_s))
    res_z.append(sum(res_z))
    res_z_res.append(sum(res_z_res))

    res_diff = np.array(res_s) - np.array(res_t)

    final_df = pd.DataFrame(
        data=[res_s, res_t, res_z, res_z_res, list(res_diff)], 
        index=rows_names, 
        columns=columns_names
    )

    final_df.to_excel(os.path.join(root_path, f"FINAL_NPM_metrics_{key}.xlsx"), index=True)