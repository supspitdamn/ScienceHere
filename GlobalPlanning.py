import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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

class MLP(nn.Module):

    def __init__(self, *args):

        super().__init__()

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

    def teaching(self, epochs, loader, verbose = True, patience = 15):

        trigger = 0
        epochs = range(epochs)
        loss_res = []

        for _ in epochs:

            epoch_loss = 0

            prev = 0

            for x, y in loader:

                res = self(x)

                loss = loss_func(res, y)

                op.zero_grad()
                loss.backward()
                op.step()

                epoch_loss += loss.item()
            
            avg_loss = epoch_loss/len(loader)
            loss_res.append(avg_loss)

            if len(loss_res) > 1:

                if abs(loss_res[-1] - loss_res[-2])  < 1e-3 :
                    trigger += 1
                else:
                    trigger = 0
                
            if _ % 100 == 0 and verbose:

                print(f"Эпоха: {_}, Лосс: {avg_loss}")
            
            if trigger == patience:

                print(f"Останов. Эпоха : {_}. Лосс: {avg_loss}")
                break
        
        plt.plot(range(len(loss_res)), loss_res)
        plt.title("Функция потерь MLP")
        plt.xlabel("Эпохи обучения")
        plt.ylabel("Лосс MSE")
        plt.grid(visible=True)
        plt.show()
        return model.get_parameter()


model = MLP(7, 10, 15, 3)

op = optimizer.Adam(model.parameters(), 0.01)
loss_func = nn.MSELoss()
model.train()

model_parameters = model.teaching(epochs=400, loader = train_loader)
print(model_parameters)
        


        
