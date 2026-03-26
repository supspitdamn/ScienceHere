# """19/03/2026"""

# """#3"""
import torch

# a = torch.Tensor(3, 5, 2) # Old
# a = torch.empty(3, 5, 2, dtype = torch.int32)
# print(a.dtype)

# print(torch.tensor([1]))
# print(torch.tensor([[1,2],[3,4],[5,6]]))

# d = [[1,2,3],[4,5,6]]
# t = torch.tensor(d, dtype = torch.torch.float32)
# print(t.type(), t.dtype, t.size(), t.shape, sep = '\n' )

# import numpy as np

# d_np = np.array([[1,2,3],[4,5,6]])
# t2 = torch.from_numpy(d_np) # ссылаются оба на одну область памяти!!!
# t2 = torch.tensor(d_np, dtype=torch.int32) # Независимы друг от дурга
# print(t2.type())

# d = t2.numpy()
# print(type(d))

# tf = t2.float()

# """#4"""

# tz = torch.zeros(2,3)
# # %%

# print(tz)

# to = torch.ones(2, 3)

# print(to)

# te = torch.eye(3)

# print(te)

# print(torch.full((2,4),5))

# print(torch.arange(7))
# print(torch.arange(-5, 0, 2)) # с границей

# print(torch.linspace(1, 5, 2))

# print(torch.rand(2,3))

# torch.manual_seed(12)

# # func_ - изменяет текущий тензор. func - формирует новый тензор

# x = torch.FloatTensor(2,4)

# x.fill_(1)

# x.uniform_(0, 1) # Рандом по диапазону

# x.normal_(0, 1)

# x = torch.arange(27)

# d = x.view(3, 9)
# print(d)

# x[0] = 100
# print(d)
# x.resize_(2,2)
# print(x)

# x.ravel() # тензор в вектор
# d.permute(1,0)

# """#5"""

# a = torch.arange(12)

# print(type(a[2]))

# print(type(a[0].item()))

# b = a[2:4]

# x = torch.zeros(3,3)

# print(x)

# a = torch.arange(1, 82).view(3,3,3,3)
# print(a)

# print(a[:, 1, :, :])

# """#6"""

# ## Просто математические операции +-*/

# """#7"""

# # a = torch.arange(10)

# # a.sum() # Тензор суммы элементов

# # a.mean()
# # a.max()
# # a.min()

# # a = a.view(5, 2)
# # print(a)

# # torch.matmul(a, b) # с возможностью транслирования (когда нельзя умножать матрично но есть возможность транслирования)

# bx = torch.randn(7,3,5)
# by = torch.randn(7, 5, 4)

# print(torch.bmm(bx, by))

# a = torch.arange(1, 10, dtype = torch.float32)

# c = torch.dot(a, b) # два вектора в один элемент
# c = torch.outer(a, b) # два вектора в матрицу n*n

# import matplotlib.pyplot as plt

# N = 5

# x1 = torch.rand(N)
# x2 = x1 + torch.randint(1, 10, [N])/10
# C1 = torch.vstack([x1, x2]).mT

# x1 = torch.rand(N)
# x2 = x1 - torch.randint(1, 10, [N])/10
# C2 = torch.vstack([x1, x2]).mT

# f = [0, 1]

# w = torch.FloatTensor([-0.3 , 0.3])

# for i in range(N):

#     x = C1[:][i]
#     y = torch.dot(w, x)

#     if y >= 0:

#         print("Class C1")
    
#     else:

#         print("Class C2")

# plt.scatter(C1[:, 0], C1[:, 1], s = 10, c = "red")
# plt.scatter(C2[:, 0], C2[:, 1], s = 10, c = "blue")
# plt.plot(f)
# plt.grid()
# plt.show()

# x = torch.tensor([2.0], requires_grad=True)
# y = torch.tensor([-4.0], requires_grad=True)

# f = (x+y)**2 + 2*x*y
# f.backward()

# print(f)
# print(x.data, x.grad)
# print(y.data, y.grad)



import torch
from random import randint
import matplotlib.pyplot as plt
import torch.optim as optim

# def model(X, w):

#     return X @ w

# N = 2
# w = torch.FloatTensor(N).uniform_(-1e-5, 1e-5)
# w.requires_grad_(True)

# x = torch.arange(0, 3, 0.1)

# y_train = 0.5*x + 0.2 * torch.sin(2*x) - 3.0
# x_train = torch.tensor([[_x ** _n for _n in range(N)] for _x in x])

# total = len(x)
# optimizer = optim.SGD(params = [w], lr = 0.01, momentum = 0.8, nesterov=True)
# loss_func = torch.nn.MSELoss()
# lr = torch.tensor([0.1, 0.01])

# for _ in range(1000):

#     k = randint(0, total - 1)
#     y = model(x_train[k], w)
#     loss = loss_func(y, y_train[k])

#     loss.backward()
#     # w.data = w.data - lr * w.grad
#     # w.grad.zero_()
#     optimizer.step()
#     optimizer.zero_grad()

# print(w)
# predict = model(x_train, w)

# plt.plot(x, y_train.numpy())
# plt.plot(x, predict.data.numpy())
# plt.grid()
# plt.show()

import torch.nn as nn
import torch.nn.functional as func

# def forward(inp, l1: nn.Linear, l2: nn.Linear):
#     u1 = l1.forward(inp)
#     s1 = func.tanh(u1)

#     u2 = l2.forward(s1)
#     s2 = func.tanh(u2)

#     return s2

# layer1 = nn.Linear(in_features=3, out_features=2)
# lasyer2 = nn.Linear(in_features=2, out_features=1)

# print(layer1.weight, layer1.bias, sep = "\n")

# class NN(nn.Module):

#     def __init__(self, input_dim, num_hidden, output_dim):

#         super().__init__()
#         self.Layer1 = nn.Linear(input_dim, num_hidden)
#         self.Layer2 = nn.Linear(num_hidden, output_dim)
    
#     def forward(self, x):

#         x = self.Layer1(x)
#         x = func.tanh(x)

#         x = self.Layer2(x)
#         x = func.tanh(x)

#         return x

# model = NN(3, 2, 1)
# optimizer = optim.RMSprop(params = model.parameters(), lr = 0.01)
# loss_func = nn.MSELoss()
# x_train = torch.FloatTensor([(-1,-1,-1), (-1,-1,1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1),
#                               (1, -1, 1), (1, 1, -1), (1, 1, 1)])
# y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])
# total = len(y_train)
# model.train()

# for _ in range(1000):
    
#     k = randint(0, total - 1)
#     y = model(x_train[k])
#     y = y.squeeze()

#     loss = loss_func(y, y_train[k])
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# model.eval()
# model.requires_grad_(False) # отключаем на уровне модели
# for x, d in zip(x_train, y_train):
#     with torch.no_grad(): # менеджер контекста для отключения градиентов
#         y = model(x)
#         print(f"Выходное значение НС: {y.data} -> {d}")
import os
import json
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs

class DigitDataset(data.Dataset):

    def __init__(self, path, train = True, transform = None):

        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform
        
        with open(os.path.join(path, "format.json"), "r") as fp:

            self.format = json.load(fp)
        
        self.length = 0
        self.files = []
        self.targets = torch.eye(10)

        for _dir, _target in self.format.items():

            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))
    
    def __getitem__(self, item):

        path_file, target = self.files[item]
        t = self.targets[target]
        img = Image.open(path_file)

        return img, t

    def __len__(self):

        return self.length

class NN(nn.Module):

    def __init__(self, input_dim, num_hidden, output_dim):

        super().__init__()
        self.Layer1 = nn.Linear(input_dim, num_hidden)
        self.Layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):

        x = self.Layer1(x)
        x = func.tanh(x)

        x = self.Layer2(x)
        x = func.tanh(x)

        return x
    
d_train = DigitDataset("mnist_dataset")
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)