import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.layers as layers
import torch as pt


# N = 5
# b = 3

# x1 = np.random.random(N)
# x2 = x1 + [np.random.randint(10)/10 for i in range(N)] + b
# C1 = [x1, x2]

# x1 = np.random.random(N)
# x2 = x1 - [np.random.randint(10)/10 for i in range(N)] - 0.1 - b
# C2 = [x1, x2]

# f = [0, 1]
# w2 = 0.5
# w3 = -b*w2

# w = np.array([-w2, w2, w3])

# for i in range(N):

#     x = np.array([C1[0][i],C1[1][i],1])
#     y = np.dot(w, x)
#     if y >= 0:
#         print("Class C1")
#     else:
#         print("Class C2")
# plt.scatter(C1[0][:], C1[1][:], s = 10, c = 'red')
# plt.scatter(C2[0][:], C2[1][:], s = 10, c = "blue")
# plt.plot(f)
# plt.grid(True)
# plt.show()

"""Here was demonstrated single perceptron func"""

c = np.array([-40, -10, 0, -8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

model = keras.Sequential()
model.add(layers.Dense(units = 1, input_shape = (1,), activation = "linear"))
model.compile(loss = "mean_squared_error", optimizer = keras.optimizers.Adam(0.1))

history = model.fit(c, f, epochs = 500, verbose = False)

plt.plot(history.history["loss"])
plt.grid(True)
plt.show()
