import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

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

        self.stage_1 = nn.ModuleList(seq_neur[0]) # Список из 3 подсетей для скоростей
        self.stage_2 = seq_neur[1]                # Сеть для токов
        self.stage_3 = seq_neur[2]                # Сеть для проскальзываний
        self.stage_4 = seq_neur[3]                # Сеть для дельта-координат

        if kaiman_weight_init:
            self.apply(self._init_kaiming)

        self.to(device)
    
    def _init_kaiming(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, vec: torch.Tensor):

        surfs = vec[:, 3:]

        # Стадия 1
        v1_s = self.stage_1[0](torch.cat([vec[:, 0:1], surfs], dim=1))
        v2_s = self.stage_1[1](torch.cat([vec[:, 1:2], surfs], dim=1))
        v3_s = self.stage_1[2](torch.cat([vec[:, 2:3], surfs], dim=1))

        v_st1 = torch.cat((v1_s, v2_s, v3_s), dim=1)

        # Стадия 2
        in_st2 = torch.cat((v_st1, surfs), dim=1)
        cur_st2 = self.stage_2(in_st2)

        # Стадия 3
        in_st3 = torch.concat((cur_st2, v_st1, surfs), dim=1)
        slip_st3 = self.stage_3(in_st3)

        # Стадия 4
        in_st4 = torch.cat((v_st1, slip_st3, surfs), dim=1)
        delta_st4 = self.stage_4(in_st4) 



        return delta_st4, slip_st3, cur_st2, v_st1

    def evaluate(self, data_loader: DataLoader, name: str, save_path: str, device: str = "cpu") -> dict:
        all_pred = []
        all_true = []
        self.eval()
        
        for x, y in data_loader:

            x, y = x.to(device), y.to(device)

            with torch.no_grad():

                delta_st4, slip_st3, cur_st2, v_st1 = self(x) 

                true_ordered = y.cpu().numpy()

                predict_ordered = np.hstack([
                    v_st1.cpu().numpy(),
                    cur_st2.cpu().numpy(),   
                    slip_st3.cpu().numpy(),
                    delta_st4.cpu().numpy()  
                ])

                all_true.append(true_ordered)
                all_pred.append(predict_ordered)

        all_pred = np.vstack(all_pred)
        all_true = np.vstack(all_true)

        mse = mean_squared_error(all_true, all_pred, multioutput="raw_values")
        mae = mean_absolute_error(all_true, all_pred, multioutput="raw_values")
        r2 = r2_score(all_true, all_pred, multioutput="raw_values")

        return {
            "MSE": list(mse), 
            "MAE": list(mae), 
            "R2": list(r2)
        }

    def fit(self, optimizer, loss, scheduler, train_loader, val_loader, epochs, root_path, patience = 10):

        sum_train_losses = []
        sum_val_losses = []

        best_val_loss = float('inf')
        counter = 0
        best_model_path = os.path.join(root_path, "best_model.pth")
        
        device = next(self.parameters()).device
        with open(os.path.join(root_path, "LOG.txt"), "a", encoding = "utf-8") as log_txt:

            for idx in range(epochs):
                train_losses = []   
                val_losses = []

                self.train()

                train_tqdm = tqdm(train_loader, desc=f"Эпоха {idx+1}/{epochs} [Обучение]")

                for x, y in train_tqdm:

                    current_lr = optimizer.param_groups[0]['lr']
                    x, y = x.to(device), y.to(device)

                    delta_st4, slip_st3, cur_st2, v_st1 = self(x)

                    loss_delta  = loss(delta_st4, y[:, 6:])
                    loss_vel    = loss(v_st1, y[:, 0:3])
                    loss_slip = loss(slip_st3, y[:]) # нет данных по данному таргету
                    loss_cur    = loss(cur_st2, y[:, 3:6])

                    summary_loss = loss_delta + loss_vel + loss_slip + loss_cur

                    train_losses.append(summary_loss.item())

                    optimizer.zero_grad()
                    summary_loss.backward()
                    optimizer.step()

                    train_tqdm.set_postfix(batch_loss=f"{summary_loss.item():.4f}", lr = f"{current_lr:.6f}")
                
                res_loss_train = sum(train_losses)/len(train_loader)
                sum_train_losses.append(res_loss_train)
                
                self.eval()

                val_tqdm = tqdm(val_loader, desc=f"Эпоха {idx+1}/{epochs} [Валидация]")

                for x, y in val_tqdm:
                    x, y = x.to(device), y.to(device)
                    batch_size = x.shape[0]

                    with torch.no_grad():

                        delta_st4, slip_st3, cur_st2, v_st1 = self(x)

                        loss_delta  = loss(delta_st4, y[:, 6:])
                        loss_vel    = loss(v_st1, y[:, 0:3])
                        loss_slip = loss(slip_st3, y[:]) # нет данных по данному таргету
                        loss_cur    = loss(cur_st2, y[:, 3:6])

                        summary_loss = loss_delta + loss_vel + loss_slip + loss_cur

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

def load_models():

    vels = MLP(5, 32, 32, 32, 32, 32, 1)
    curs = MLP(7, 64, 64, 64, 64, 3)
    slips = MLP(10, 64, 64, 64, 3)
    deltas = MLP_NORM(10, 128, 128, 128, 3)

    best_vels = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\SetVelocity_To_RealVelocity\MLP_study_20260413_204155\MLP_11_5-32-32-32-32-32-1_Adam_0.0001735565808786231_MSELoss_Batch_32\MLPconfig.pth")

    best_curs = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\RealVelocity_To_Current\MLP_study_20260412_181402\MLP_34_7-64-64-64-64-3_Adam_0.0006238342122664613_MSELoss_Batch_64\MLPconfig.pth")

    best_slips = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\Currents_to_Slippage\Seeking_for_best_features_05-05-2026_18-18-28\feat_count_m1cur_m2cur_m3cur_m1vel_m2vel_m3vel_surfs\MLP_31_10-64-64-64-3_Adam_0.0009816432611570356_MSELoss_Batch_64\MLPconfig.pth")

    best_deltas = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\Slippage_to_DeltaCoords\peredelka_modeli_izza_privat_24-05-2026_21-40-36\MLPconfig.pth")

    vels.load_state_dict(best_vels["model"])
    curs.load_state_dict(best_curs["model"])
    slips.load_state_dict(best_slips["model"])
    deltas.load_state_dict(best_deltas["model"])

    return [vels, vels, vels], curs, slips, deltas

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = [
    "m1setvel", "m2setvel", "m3setvel",
    "type__brown", "type__gray", "type__green", "type__table"
]

targets = [
    "m1vel", "m2vel", "m3vel",
    "m1cur", "m2cur", "m3cur",
    "w1slip", "w2slip", "w3slip", 
    "vx", "vy", "omega"
]

effvels = ["w1effvel", "w2effvel", "w3effvel"]
real_vels = ["w1vel", "w2vel", "w3vel"]

save_path = "testing_trajectories"
os.makedirs(save_path, exist_ok=True)

first_expirement = r"neuro_physical_model\NPM_comparison\FINAL_NPM_metrics_test.xlsx"
res_first_exp = pd.read_excel(first_expirement)
df_first_exp = pd.read_csv(r"RNN\Phi\robot_data_with_chunks.csv")

means_first_exp = df_first_exp[targets].abs().mean().to_numpy()
raw_maes_first_exp = res_first_exp.iloc[1][-2:-14:-1].to_numpy()

ready_maes_first_exp = []
for i in range(0, len(raw_maes_first_exp), 3):
    sliceCount = raw_maes_first_exp[i:(i+3 if i+3 < len(raw_maes_first_exp) else len(raw_maes_first_exp))]
    for j in range(len(sliceCount) - 1, -1, -1):
        ready_maes_first_exp.append(sliceCount[j])

try:
    if len(ready_maes_first_exp) != len(means_first_exp):
        raise IndexError
    first_exp_points = tuple(ready_maes_first_exp[i]/means_first_exp[i] for i in range(len(ready_maes_first_exp)))
except IndexError:
    print("Не совпали размеры базового эксперимента")
    sys.exit()

df_square = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\УИРС\SEM6\square.csv")
df_circle = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\УИРС\SEM6\circle.csv")

for df_tmp in [df_square, df_circle]:
    df_tmp["t, ms"] = df_tmp["t"]
    df_tmp.drop(columns=["t"], inplace=True)
    df_tmp["t, s"] = df_tmp["t, ms"].apply(lambda x: x * 1e-3)

df_square_surfs = pd.get_dummies(df_square, columns=["surf"], prefix="type_")
df_circle_surfs = pd.get_dummies(df_circle, columns=["surf"], prefix="type_")

for df_surf in [df_square_surfs, df_circle_surfs]:
    for col in ["type__green", "type__gray"]:
        if col in df_surf.columns:
            df_surf[col] = df_surf[col].astype(np.float64)

dfs = {
       "Square": df_square_surfs, 
       "Circle": df_circle_surfs
}


for shape, df in dfs.items():
    print(f"Эксперимент: {shape} начат\n")

    df.loc[:, "type__brown"] = 0.0
    df.loc[:, "type__table"] = 0.0

    for i, (effective, real) in enumerate(zip(effvels, real_vels)):
        division = df[effective].div(df[real]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df[f"w{i+1}slip"] = 1.0 - division

    local_folder = "prepared_data"
    directory = os.path.join(save_path, local_folder + shape)
    os.makedirs(directory, exist_ok=True)
    df.to_csv(os.path.join(directory, f"{shape}_route.csv"))

    df["original_index"] = df.index

    X_np = df[features].to_numpy(dtype=np.float32)
    y_np_clean = df[targets].to_numpy(dtype=np.float32) 
    y_np_graphs = df[targets + ["original_index"]].to_numpy(dtype=np.float32)

    test_dataset_eval = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np_clean))
    test_loader_eval = DataLoader(test_dataset_eval, batch_size=1, shuffle=False)

    test_dataset_graphs = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np_graphs))
    test_loader_graphs = DataLoader(test_dataset_graphs, batch_size=512, shuffle=False, pin_memory=True)

    mean_values = dict(zip(targets, df[targets].abs().mean().to_numpy()))

    vels, curs, slips, deltas = load_models()
    model = NPM([vels, curs, slips, deltas])
    model.to(device)
    model.eval()

    metrics = model.evaluate(test_loader_eval, name="Test", save_path=save_path, device=str(device))

    columns = ["Метрика", "Скорость М1", "Скорость М2", "Скорость М3", 
               "Ток М1", "Ток М2", "Ток М3",
               "Проскальзывание М1", "Проскальзывание М2", "Проскальзывание М3",
               "Дельта Х", "Дельта У", "Дельта Фи"]
    
    data = [["MAE"] + list(metrics["MAE"]), 
            ["MSE"] + list(metrics["MSE"]), 
            ["R2"] + list(metrics["R2"])]

    resulting_table = pd.DataFrame(columns=columns, data=data)
    resulting_table.to_csv(os.path.join(save_path, f"{shape}_results.csv"), encoding="utf-8-sig", index=False)
    print(resulting_table, "\n\n")

    points = [None] * len(targets)
    for i in range(len(points)):
        points[i] = data[0][i+1] / mean_values[targets[i]]

    plt.figure(figsize=(10, 5))
    plt.plot(columns[1:], points, marker='o')
    plt.plot(columns[1:], first_exp_points, marker='^')
    plt.xticks(rotation=45)
    plt.ylabel("Относительная ошибка (MAE / Среднее модуля)")
    plt.title(f"Эксперимент: {shape}")
    plt.grid(True)
    plt.legend(["Эксперимент по траектории", "Исходный результат"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{shape}_test.png"), dpi=300)
    plt.close()

    print(f"Сбор динамики для графиков {shape}...")
    predict_list, output_list, index_list = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader_graphs:
            x_batch = x_batch.to(device)
            current_batch_size = x_batch.shape[0]

            delta_st4, slip_st3, cur_st2, v_st1 = model(x_batch)

            delta_3d = delta_st4.reshape(current_batch_size, -1, 3)
            slip_3d = slip_st3.reshape(current_batch_size, -1, 3)
            vel_3d = v_st1.reshape(current_batch_size, -1, 3)
            cur_3d = cur_st2.reshape(current_batch_size, -1, 3)

            predict_ordered = np.hstack([
                vel_3d[:, -1, :].cpu().numpy(),    # 1-3: Скорости
                cur_3d[:, -1, :].cpu().numpy(),    # 4-6: Токи
                slip_3d[:, -1, :].cpu().numpy(),   # 7-9: Проскальзывания
                delta_3d[:, -1, :].cpu().numpy()   # 10-12: Дельты
            ])

            true_value = y_batch.numpy()
            true_ordered = true_value[:, :12]  
            batch_indices = true_value[:, -1].astype(int) 

            predict_list.append(predict_ordered)
            output_list.append(true_ordered)
            index_list.append(batch_indices)

    predict_np = np.vstack(predict_list)
    output_true_ordered = np.vstack(output_list)
    all_windows_indices = np.concatenate(index_list)

    timestamps = df.loc[all_windows_indices, "t, s"].values
    sort_indices = np.argsort(timestamps)

    timestamps_sorted = timestamps[sort_indices]
    predict_np_sorted = predict_np[sort_indices]
    output_true_ordered_sorted = output_true_ordered[sort_indices]

    plots_directory = os.path.join(directory, "plots")
    os.makedirs(plots_directory, exist_ok=True)

    for idx in range(12):
        plt.figure(figsize=(12, 5))
        plt.plot(timestamps_sorted, predict_np_sorted[:, idx], label='Предсказание модели', color='blue', alpha=0.9)
        plt.plot(timestamps_sorted, output_true_ordered_sorted[:, idx], label='Истина', color='orange', linestyle='--', alpha=0.8)
        plt.title(f"Траектория: {shape} | Параметр: {columns[idx+1]}")
        plt.xlabel("Время, с")
        plt.ylabel("Значение")
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_directory, f"{columns[idx+1].replace(' ', '_')}.png"), dpi=150)
        plt.show()
        plt.close()

    print(f"Все графики для эксперимента {shape} успешно сохранены в папку {plots_directory}\n")
    print("="*50 + "\n")

    








