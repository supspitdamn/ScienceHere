import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from itertools import combinations

class SBS:

    def __init__(self, estimator, n_features,
                 scoring = accuracy_score,
                 test_size = 0.25, random_state=1):
        
        self.scoring = scoring
        self.estimator = estimator
        self.n_features = n_features
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     random_state=self.random_state, 
                                                     test_size=self.test_size, 
                                                     stratify = y)
        
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, X_test, y_train, y_test, self.indices_)
        
        self.scores_ = [score]

        while dim > self.n_features:

            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, X_test, y_train, y_test, p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            self.scores_.append(scores[best]) 
            dim -= 1
        
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, X_test, y_train, y_test, indices):

        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
    
data = load_wine()

X, y = data.data, data.target

print(data.target_names)
columns = data.feature_names + ["class_label"]
data = np.hstack((X, y.reshape(-1, 1)))

df = pd.DataFrame(data = data, columns=columns)
print(df)

X, y = df.iloc[:, 0:13], df.iloc[:, 13]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     random_state=1, 
                                                     test_size=0.3, 
                                                     stratify = y)

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
sc = StandardScaler()
X_train_std =sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, n_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
# plt.plot(k_feat, sbs.scores_, marker = "o")
# plt.ylim([0.7, 1.02])
# plt.ylabel("Accurancy")
# plt.xlabel("Num features")
# plt.grid()
# plt.tight_layout()
# plt.show()

k6 = sbs.subsets_[5]
print(df.columns[:13][list(k6)])

from sklearn.ensemble import RandomForestClassifier

features_labels = df.columns[:13]
forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(0, X_train.shape[1]):

     print(f"{i + 1:2}) {features_labels[indices[i]]:<30} {importances[indices[i]]:>0}")


# plt.bar(range(X_train.shape[1]), importances[indices], align = "center")

# plt.xticks(range(X_train.shape[1]))
# plt.xlim([-1, X_train.shape[1]])
# plt.tight_layout()
# # plt.show()


cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print(f"Eigenvalues: {eigen_vals}")

eigen_vals = np.real(eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse = True)]
cum_var_exp = np.cumsum(var_exp)
# plt.bar(range(1, 14), var_exp, align = "center", label = "Отдельные объясненные дисперсии")
# plt.step(range(1, 14), cum_var_exp, where = "mid", label = "Совокупная общая дисперсия")
# plt.ylabel("Доля объясненной дисперсии")
# plt.xlabel("Индекс главной компоненты")
# plt.legend(loc="best")
# plt.tight_layout()
# plt.show()

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

eigen_pairs.sort(key = lambda x: x[0], reverse = True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis])) # матрица проекции

print(w)

X_new_obrazec1 = X_train_std[0].dot(w)
X_train_new = X_train_std.dot(w)
X_test_new = X_test_std.dot(w)

colours = ["r", "b", "g"]
markers = ["o", "s", "^"]

for l, c, m in zip(np.unique(y_train), colours, markers):

    plt.scatter(X_train_new[y_train == l, 0], 
                X_train_new[y_train == l, 1], 
                c=c, 
                label=f"Class {int(l)}", 
                marker=m)
    
# plt.xlabel("PC 1")
# plt.ylabel("PC 2")
# plt.legend(loc = "lower left")
# plt.tight_layout()
# plt.show()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty = "l2",
                        C=100,
                        solver = "lbfgs")

lr.fit(X_train_new, y_train)
pred = lr.predict(X_test_new)

print(accuracy_score(y_test, pred))

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Настройка поверхности для рисования
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # Предсказание для каждой точки фона
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Отрисовка самих точек
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=f"Класс {int(cl)}")

# # Запуск
# plot_decision_regions(X_test_new, y_test, classifier=lr)
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc='lower left')
# plt.show()

# loadings = eigen_vecs * np.sqrt(eigen_vals)

# fig, ax = plt.subplots()
# ax.bar(range(13), loadings[:, 0], align = "center")
# ax.set_ylabel("Loadings for PC 1")
# ax.set_xticks(range(13))
# ax.set_xticklabels(df.columns[:13],  rotation = 90)
# plt.ylim([-1, 1])
# plt.tight_layout()
# plt.show()

np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis = 0))