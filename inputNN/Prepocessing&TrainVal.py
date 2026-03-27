import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\УИРС\\SEM5\\filtered_robot_data.csv")

df = pd.get_dummies(df[["m1vel", "m2vel", "m3vel", "surf", "m1cur", "m2cur", "m3cur"]], columns = ["surf"], prefix = "type_")

train, temp = train_test_split(df, test_size = 0.2, random_state=42, shuffle = True)
val, test = train_test_split(temp, test_size = 0.5, random_state = 42, shuffle = True)

features = ["m1vel", "m2vel", "m3vel", "type__brown", "type__gray", "type__green", "type__table"]
targets = ["m1cur", "m2cur", "m3cur"]

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train = torch.tensor(scaler_x.fit_transform(train[features]), dtype = torch.float32)
y_train = torch.tensor(scaler_y.fit_transform(train[targets]), dtype = torch.float32)

x_val = torch.tensor(scaler_x.transform(val[features]), dtype = torch.float32)
y_val = torch.tensor(scaler_y.transform(val[targets]), dtype = torch.float32)

x_test = torch.tensor(scaler_x.transform(test[features]), dtype = torch.float32)
y_test = torch.tensor(scaler_y.transform(test[targets]), dtype = torch.float32)

batch_size = 64

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

class MLP(nn.Module):

    def __init__(self, *args):

        super().__init__()
        self.struct = args
        self.l1 = nn.Linear(args[0], args[1])
        self.l2 = nn.Linear(args[1], args[2])
        self.l3 = nn.Linear(args[2], args[3])
    
    def forward(self, vec):

        x = self.l1(vec)
        x = F.relu(x)

        x = self.l2(x)
        x = F.relu(x)

        x = self.l3(x)

        return x

    def teaching(self, epochs, train_loader, val_loader, model_state_dict, verbose = True, patience = 15):

        trigger = 0

        epochs = range(epochs)

        train_loss_res = []
        val_loss_res = []

        for _ in epochs:

            train_epoch_loss = 0
            val_epoch_loss = 0

            self.train()

            for x, y in train_loader:

                res = self(x)

                loss = loss_func(res, y)

                op.zero_grad()
                loss.backward()
                op.step()

                train_epoch_loss += loss.item()
            
            avg_loss = train_epoch_loss/len(train_loader)

            train_loss_res.append(avg_loss)

            model.eval()

            for x, y in val_loader:

                with torch.no_grad():

                    res = self(x)
                    loss = loss_func(res, y)
                    val_epoch_loss += loss.item()
            
            avg_val_loss = val_epoch_loss/len(val_loader)
            val_loss_res.append(avg_val_loss)

            if len(val_loss_res) > 1:

                if abs(val_loss_res[-1] - val_loss_res[-2])  < 1e-3 :
                    trigger += 1
                else:
                    trigger = 0
            
            if trigger == 1:

                model_state_dict["model"] = self.state_dict()
                model_state_dict["optimizer"] = op.state_dict()
                torch.save(model_state_dict, f"MLPexp.pth")
                
            if _ % 100 == 0 and verbose:

                print(f"Эпоха: {_}, Лосс: {avg_loss}")
            
            if trigger == patience:

                print(f"Останов. Эпоха : {_}. Лосс: {avg_loss}")
                break
        
        plt.plot(range(len(train_loss_res)), train_loss_res, color="blue", label = "Обучение")
        plt.plot(range(len(val_loss_res)), val_loss_res, color = "red", label = "Валидация")
        plt.legend()
        plt.title(f"Функция потерь MLP{'-'.join(map(str, self.struct))}")
        plt.xlabel("Эпохи обучения")
        plt.ylabel("Лосс MSE")
        plt.grid(visible=True)

        plt.savefig(f"Loss_MLP_{'-'.join(map(str, self.struct))}.png", dpi = 300, bbox_inches = "tight")

        plt.show()
    
    def evaluate(self, data_loader, scaler_y):
        all_pred = []
        all_true = []
        self.eval()
        for x, y in data_loader:

            with torch.no_grad():

                predict = self.forward(x).detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                predict = (scaler_y.inverse_transform(predict))
                true_value = (scaler_y.inverse_transform(y))

                all_true.append(true_value)
                all_pred.append(predict)

        all_pred = np.vstack(all_pred)
        all_true = np.vstack(all_true)

        mse = mean_squared_error(all_pred, all_true, multioutput="raw_values")
        mae = mean_absolute_error(all_pred, all_true, multioutput="raw_values")
        mape = mean_absolute_percentage_error(all_pred, all_true, multioutput="raw_values")
        r2 = r2_score(all_true, all_pred, multioutput="raw_values")

        return {"MSE": tuple(mse), "MAE" : tuple(mae), "MAPE" : tuple(mape), "R2" : tuple(r2)}
                

model = MLP(7, 10, 15, 3)

op = optimizer.Adam(model.parameters(), 0.001)
loss_func = nn.MSELoss()
model.train()

model_state_dict = {
                    "testX": x_test,
                    "testY": y_test,
                    "scalerX": scaler_x, 
                    "ScalerY": scaler_y,
                    "Structure": model.struct, 
                    "optimizer": op.state_dict(), 
                    "Loss": loss_func, 
                    "model": model.state_dict(),
                    }

model_parameters = model.teaching(epochs=100,
                                   train_loader = train_loader,
                                     val_loader = val_loader,
                                       model_state_dict = model_state_dict)

data = {"train" : train_loader, "val" : val_loader, "test" : test_loader}

for key, value in data.items():

    res = model.evaluate(value, scaler_y=scaler_y)

    metrics_df = pd.DataFrame(res, index=["Двигатель 1", "Двигатель 2", "Двигатель 3"]).T

    metrics_df.loc["MAPE, %"] = metrics_df.loc["MAPE"]*100
    metrics_df.drop("MAPE", inplace=True)

    metrics_df.to_csv(f"MLP_{'-'.join(map(str, model.struct))}_metrics_{key}.csv", index_label="Метрики", encoding="utf-8-sig")