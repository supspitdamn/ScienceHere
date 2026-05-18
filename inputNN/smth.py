from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
sc = StandardScaler()

sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# plt.figure(figsize = [10, 5])
# plt.hist(X[:, 1])
# plt.show()

# print(f"Признаков: {len(X)}")

# plt.figure(figsize = [10, 5])
# plt.hist(X[:, 0])
# plt.show()

# scatter = plt.scatter(X[:, 0], X[:, 1], c = y, cmap = "Pastel1")

# plt.xlabel("Petal length")
# plt.ylabel("Petal width")
# plt.title("Distribution")

# plt.legend(handles = scatter.legend_elements()[0], labels=list(iris.target_names))

# plt.show()

# ppn = Perceptron(eta0 = 0.1, random_state=1)
# ppn.fit(X_train_std, y_train)
# pred = ppn.predict(X_test_std)

# print(accuracy_score(y_test, pred))

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    
    # Настройка маркеров и цветов
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Определяем границы плоскости
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Генерируем сетку точек (исправлено arrange -> arange)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # Предсказываем класс для каждой точки сетки
    lab = classifier.predict(X=np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    # Рисуем закрашенные области
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max()) # исправлено xlim -> ylim

    # Рисуем обучающие примеры
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    label=f"Class {cl}", # исправлено (cl) -> {cl}
                    edgecolors="black")
    
    # Выделяем тестовые примеры, если они переданы
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx] # исправлено y[test_idx, :] -> y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c="none", edgecolors="black", alpha=1.0,
                    linewidth=1, marker="o",
                    s=100, label="test set")

# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X = X_combined_std,
#                        y = y_combined,
#                        classifier=ppn,
#                        test_idx = range(105, 150))

# plt.xlabel("Длина лепестка (стд)")
# plt.ylabel("Ширина лепестка (стд)")
# plt.legend(loc = "upper left")
# plt.tight_layout()
# plt.show()

# lgr = LogisticRegression(C=100, solver = "lbfgs")
# lgr.fit(X_train_std, y_train)
# plot_decision_regions(X = X_combined_std,
#                        y = y_combined,
#                        classifier=lgr,
#                        test_idx = range(105, 150))

# plt.xlabel("Длина лепестка (стд)")
# plt.ylabel("Ширина лепестка (стд)")
# plt.legend(loc = "upper left")
# plt.tight_layout()
# plt.show()

# print(lgr.predict_proba(X_test_std[:1, :]))
# np.random.seed(1)
# X_xor = np.random.randn(200, 2)
# y_xor = np.logical_xor(X_xor[:, 0] > 0,
#                        X_xor[:, 1] > 0)

# y_xor = np.where(y_xor, 1, 0)
# plt.scatter(X_xor[y_xor == 1, 0],
#             X_xor[y_xor == 1, 1],
#             c="royalblue",
#             marker = "s",
#             label = "Class 1"
#             )

# plt.scatter(X_xor[y_xor == 0, 0],
#             X_xor[y_xor == 0, 1],
#             c="tomato",
#             marker = "o",
#             label = "Class 0")

# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# plt.xlabel("Признак 1")
# plt.ylabel("Признак 2")
# plt.legend(loc = "best")
# plt.tight_layout()
# plt.show()

# svm = SVC(kernel="rbf", random_state=1, gamma = 0.1, C= 10)
# svm.fit(X_xor, y_xor)
# plot_decision_regions(X_xor, y_xor, classifier=svm)
# plt.legend(loc = "upper left")
# plt.tight_layout()
# plt.show()

# from sklearn.tree import DecisionTreeClassifier

# tr = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)
# tr.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X=X_combined,
#                       y = y_combined,
#                       classifier=tr,
#                       test_idx = range(105, 150))
# plt.xlabel("Длина лепестка (см)")
# plt.ylabel("Ширина лепестка (см)")
# plt.legend(loc = "upper left")
# plt.tight_layout()
# plt.show()

# feature_names = iris.feature_names
# print(feature_names)

# from sklearn import tree

# tree.plot_tree(tr, feature_names=feature_names, filled = True)
# plt.show()

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=25,
                                max_depth=2,
                                random_state = 1,
                                n_jobs = 2)

forest.fit(X_train, y_train)
plot_decision_regions(X=X_combined, y=y_combined, classifier=forest, test_idx = range(105, 150))
plt.xlabel("Длина лепестка (см)")
plt.ylabel("Ширина лепестка (см)")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()
