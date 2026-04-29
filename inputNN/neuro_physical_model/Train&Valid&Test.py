import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import os
import optuna
import datetime
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Обучение на {device}")

df = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\УИРС\\SEM5\\filtered_robot_data.csv", encoding="cp1251", sep=";")

df = pd.get_dummies(df[["m1setvel", "m2setvel", "m3setvel", "m1cur", "m2cur", "m3cur", "surf", "m1vel", "m2vel", "m3vel"]], columns = ["surf"], prefix = "type_")

df.info()

df["m1setvel"] = pd.to_numeric(df["m1setvel"], errors="coerce")
df["m2setvel"] = pd.to_numeric(df["m2setvel"], errors="coerce")
df["m3setvel"] = pd.to_numeric(df["m3setvel"], errors="coerce")
df = df.dropna(subset=["m1setvel", "m2setvel", "m3setvel"])

train, temp = train_test_split(df, test_size = 0.2, random_state = 42, shuffle = True)
val, test = train_test_split(temp, test_size = 0.5, random_state = 42, shuffle = True)

features = ["m1setvel", "m2setvel", "m3setvel", "type__brown", "type__gray", "type__green", "type__table"]

features_extra = ["m1vel", "m2vel", "m3vel"]
targets = ["m1cur", "m2cur", "m3cur"]

df.info()

x_train = torch.tensor(train[features].values.astype(np.float32), dtype = torch.float32)
y_train = torch.tensor(train[targets].values.astype(np.float32), dtype = torch.float32)

x_val = torch.tensor(val[features].values.astype(np.float32), dtype = torch.float32)
y_val = torch.tensor(val[targets].values.astype(np.float32), dtype = torch.float32)

x_test = torch.tensor(test[features].values.astype(np.float32), dtype = torch.float32)
y_test = torch.tensor(test[targets].values.astype(np.float32), dtype = torch.float32)

train_dataset = TensorDataset(x_train, y_train)

val_dataset = TensorDataset(x_val, y_val)

test_dataset = TensorDataset(x_test, y_test)

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

    def __init__(self, seq_neur: list, device: str = "cuda") -> None:
        super().__init__()
        
        self.stage_1 = nn.ModuleList(seq_neur[0])
        self.stage_2 = seq_neur[1]

        self.to(device)
    
    def forward(self, vec: torch.Tensor):

        device = next(self.parameters()).device
        vec = vec.to(device).float()

        surfs = vec[:, 3:7]

        v1_s = self.stage_1[0](torch.cat((vec[:, 0:1], surfs), dim=1))
        v2_s = self.stage_1[1](torch.cat((vec[:, 1:2], surfs), dim=1))
        v3_s = self.stage_1[2](torch.cat((vec[:, 2:3], surfs), dim=1))

        v_st1 = torch.cat((v1_s, v2_s, v3_s), dim=1)

        in_st2 = torch.cat((v_st1, surfs), dim=1)

        return self.stage_2(in_st2)

    def evaluate(self, data_loader: DataLoader, name: str, save_path: str, device: str = "cpu") -> dict:

        
        all_pred = []
        all_true = []
        self.eval()
        for x, y in data_loader:

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
            mape = mean_absolute_percentage_error(all_pred, all_true, multioutput="raw_values")
            r2 = r2_score(all_true, all_pred, multioutput="raw_values")
            
            if name == "test":
                log_txt.write(20*"-"+"\n")
                log_txt.write("Результаты для тестовой выборки:\n")
                log_txt.write(f"Абсолютная ошибка (M1, M2, M3): {'  '.join(map(str, np.round(mae, 4)))}\n")
                log_txt.write(f"Относительная ошибка (M1, M2, M3): {'  '.join(map(str, np.round(mape * 100, 4)))}\n")

        return {"MSE": tuple(mse), "MAE" : tuple(mae), "MAPE" : tuple(mape), "R2" : tuple(r2)}



best_par_sv_v = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\SetVelocity_To_RealVelocity\MLP_study_20260413_204155\MLP_11_5-32-32-32-32-32-1_Adam_0.0001735565808786231_MSELoss_Batch_32\MLPconfig.pth")
best_par_v_c = torch.load(r"C:\Users\User\Documents\MyPythonProjects\inputNN\RealVelocity_To_Current\MLP_study_20260412_181402\MLP_34_7-64-64-64-64-3_Adam_0.0006238342122664613_MSELoss_Batch_64\MLPconfig.pth")
mlp_sv1_v1 = MLP(5, 32, 32, 32, 32, 32, 1)
mlp_sv2_v2 = MLP(5, 32, 32, 32, 32, 32, 1)
mlp_sv3_v3 = MLP(5, 32, 32, 32, 32, 32, 1)
mlp_vs_cs = MLP(7, 64, 64, 64, 64, 3)

mlp_sv1_v1.load_state_dict(best_par_sv_v["model"])
mlp_sv2_v2.load_state_dict(best_par_sv_v["model"])
mlp_sv3_v3.load_state_dict(best_par_sv_v["model"])
mlp_vs_cs.load_state_dict(best_par_v_c["model"])

npm = NPM([[mlp_sv1_v1, mlp_sv2_v2, mlp_sv3_v3],mlp_vs_cs], device="cuda")

npm.eval()

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
root_path = f".//neuro_physical_model//NPM_study_{timestamp}"
os.makedirs(root_path, exist_ok = True)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

data = {"train" : train_loader,
         "val" : val_loader,
           "test" : test_loader}

for key, value in data.items():

    res = npm.evaluate(value, name=key, save_path=root_path, device=device)

    metrics_df = pd.DataFrame(res, index=["Двигатель 1", "Двигатель 2", "Двигатель 3"]).T

    metrics_df.loc["MAPE, %"] = metrics_df.loc["MAPE"]*100
    metrics_df.drop("MAPE", inplace=True)

    table_path = os.path.join(root_path, f"FINAL_NPM_metrics_{key}.csv")
    metrics_df.to_csv(table_path, index_label="Метрики", encoding="utf-8-sig")