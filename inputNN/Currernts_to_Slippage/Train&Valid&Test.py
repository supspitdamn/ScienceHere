import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, weighted_absolute_percentage_error
import numpy as np
import os
import optuna
import datetime
import json
import tqdm

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

                for x, y in tqdm(train_loader, f"{_}/{len(train_loader)}"):

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

                for x, y in tqdm(val_loader,  f"{_}/{len(val_loader)}"):

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
            mape = mean_absolute_percentage_error(all_pred, all_true, multioutput="raw_values")
            r2 = r2_score(all_true, all_pred, multioutput="raw_values")
            
            if name == "test":
                log_txt.write(20*"-"+"\n")
                log_txt.write("Результаты для тестовой выборки:\n")
                log_txt.write(f"Абсолютная ошибка (M1, M2, M3): {'  '.join(map(str, np.round(mae, 4)))}\n")
                log_txt.write(f"Относительная ошибка (M1, M2, M3): {'  '.join(map(str, np.round(mape * 100, 4)))}\n")

        return {"MSE": tuple(mse), "MAE" : tuple(mae), "MAPE" : tuple(mape), "R2" : tuple(r2)}
    
    @staticmethod

    def objective(trial, input_size, train_dataset, val_dataset, root_path, device = "cpu")->float:

        lr = trial.suggest_float("lr", 1e-4, 1e-2)
        num_layers = trial.suggest_int("num_layers", 1, 5)
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        b_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 0, 1e-8)

        layers_struct = [input_size] + [hidden_size]*num_layers + [3]

        trial_model = MLP(*layers_struct).to(device=device)

        trial_op = torch.optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
        trial_loss = nn.MSELoss()
        trial_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=trial_op, factor = 0.5, patience = 10)

        v_load = DataLoader(val_dataset, 
                            batch_size=b_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)
        
        t_load = DataLoader(train_dataset, 
                            batch_size=b_size, 
                            shuffle=True,
                            num_workers=0,
                            pin_mode=True)

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
            patience=10
        )

        return best_val_loss

if __name__ == "__main__":

    timestamp_main = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    root_dir = r"./Current_to_Slippage"
    root_path = os.path.join(root_dir, f"Seeking_for_best_features_{timestamp_main}")
    os.makedirs(root_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение на {device}")

    df = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\УИРС\\SEM5\\filtered_robot_data.csv", encoding="cp1251", sep=";")
    
    cols_to_use = ["m1vel", "m2vel", "m3vel", "surf", "m1cur", "m2cur", "m3cur", "w1linslip", "w2linslip", "w3linslip"]
    df = pd.get_dummies(df[cols_to_use], columns=["surf"], prefix="type_").astype(float)

    train, temp = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    val, test = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)

    filter_cols = ["m1cur", "m2cur", "m3cur", "w1linslip", "w2linslip", "w3linslip"]
    train = train[(train[filter_cols] > 1e-2).all(axis=1)]
    val = val[(val[filter_cols] > 1e-2).all(axis=1)]
    test = test[(test[filter_cols] > 1e-2).all(axis=1)]

    print(f"После фильтрации: {len(train)}. До фильтрации: {df.shape[0]}. Потери: {(1-len(train)/df.shape[0])*100:.2f}%")

    features_variations = [
        ["m1vel", "m2vel", "m3vel"], 
        ["m2cur", "m1cur", "m3cur"], 
        ["m1cur", "m2cur", "m3cur", "type__brown", "type__gray", "type__green", "type__table"], 
        ["m1vel", "m2vel", "m3vel","type__brown", "type__gray", "type__green", "type__table"], 
        ["m1cur", "m2cur", "m3cur", "m1vel", "m2vel", "m3vel", "type__brown", "type__gray", "type__green", "type__table"]
    ] 

    for features in features_variations:

        targets = ["w1linslip", "w2linslip", "w3linslip"]
        
        x_train = torch.tensor(train[features].values, dtype=torch.float32)
        y_train = torch.tensor(train[targets].values, dtype=torch.float32)
        x_val = torch.tensor(val[features].values, dtype=torch.float32)
        y_val = torch.tensor(val[targets].values, dtype=torch.float32)
        x_test = torch.tensor(test[features].values, dtype=torch.float32)
        y_test = torch.tensor(test[targets].values, dtype=torch.float32)

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        test_dataset = TensorDataset(x_test, y_test)

        feat_folder_name = f"feat_count_{"_".join(features)}"
        current_feat_path = os.path.join(root_path, feat_folder_name)
        os.makedirs(current_feat_path, exist_ok=True)

        study = optuna.create_study(direction="minimize")

        input_dim = len(features)
        output_dim = len(targets)

        study.optimize(lambda trial: MLP.objective(trial, input_dim, train_dataset, val_dataset, current_feat_path, device), n_trials=40)

        best_trial = study.best_trial
        best_struct = [input_dim] + [best_trial.params["hidden_size"]] * best_trial.params["num_layers"] + [output_dim]
        
        folder_name = f"MLP_{best_trial.number}_{'-'.join(map(str, best_struct))}_Adam_{best_trial.params['lr']}_MSELoss_Batch_{best_trial.params['batch_size']}"
        best_weight_path = os.path.abspath(os.path.join(current_feat_path, folder_name, "MLPconfig.pth"))

        result = {
            "features": features,
            "best_trial": best_trial.number,
            "best_loss": best_trial.value,
            "params": best_trial.params,
            "best_model_path": best_weight_path
        }

        with open(os.path.join(current_feat_path, "optuna_results.json"), "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        model = MLP(*best_struct).to(device)
        checkpoint = torch.load(best_weight_path, map_location=device)
        model.load_state_dict(checkpoint["model"])

        loaders = {
            "train": DataLoader(train_dataset, batch_size=best_trial.params["batch_size"], shuffle=False),
            "val": DataLoader(val_dataset, batch_size=best_trial.params["batch_size"], shuffle=False),
            "test": DataLoader(test_dataset, batch_size=best_trial.params["batch_size"], shuffle=False)
        }

        for key, loader in loaders.items():
            res = model.evaluate(loader, name=key, save_path=current_feat_path, device=device)
            metrics_df = pd.DataFrame(res, index=["Двигатель 1", "Двигатель 2", "Двигатель 3"]).T
            
            if "MAPE" in metrics_df.index:
                metrics_df.loc["MAPE, %"] = metrics_df.loc["MAPE"] * 100
                metrics_df.drop("MAPE", inplace=True)

            metrics_df.to_csv(os.path.join(current_feat_path, f"FINAL_metrics_{key}.csv"), index_label="Метрики", encoding="utf-8-sig")
