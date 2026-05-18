from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import validation_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts

data = load_breast_cancer()

print(f"Фичи {data.feature_names}")
print(f"Цель {data.target_names}")

column_names = data.feature_names

df = pd.DataFrame(data.data[:, ::-1], columns=column_names[::-1])
df["class"] = data.target

print(df)

sc = StandardScaler()

X = df.iloc[:, ::-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

plda = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state = 1)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1, shuffle=True)
np.random.seed(1)
param_range = sts.loguniform(0.0001, 1000.0)
param_grid = [{"classifier__C" : param_range,
                "classifier__kernel" : ["linear"]},
                
                {"classifier__C" : param_range,
                 "classifier__gamma" : param_range,
                 "classifier__kernel" : ["rbf"]}
              ]

hs = HalvingRandomSearchCV(estimator=plda,
                        param_distributions=param_grid,
                        n_candidates = "exhaust",
                        resource = "n_samples",
                        factor = 1.5,
                        n_jobs = -1)

hs = hs.fit(X_train, y_train)
print(f"Лучшая точность: {hs.best_score_}\nЛучшие параметры {hs.best_params_}")
clf = hs.best_estimator_

train_sizes, train_scores, val_scores = learning_curve(
    estimator = clf,
    X = X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv = 5,
    n_jobs = 1
)

train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
val_mean = np.mean(val_scores, axis = 1)
val_std = np.std(val_scores, axis = 1)

plt.plot(train_sizes, train_mean, label = "Обучение", color = "blue", marker = "o")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = "blue")
plt.plot(train_sizes, val_mean, label = "Валидация", color = "green", marker = "s")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha = 0.15, color = "green")

plt.title("Кривая обучения")
plt.xlabel("Размер обучающей выборки")
plt.ylabel("Accuracy")
plt.legend(loc = "best")
plt.grid(True)
plt.show()

param_range = np.logspace(-4, 3, 10)
train_scores, val_scores = validation_curve(estimator=clf,
                                            X = X_train,
                                            y = y_train,
                                            param_name="classifier__C",
                                            param_range = param_range,
                                            cv = 5)

train_mean = np.mean(train_scores, axis = 1)
train_std =np.std(train_scores, axis = 1)
val_mean = np.mean(val_scores, axis = 1)
val_std = np.std(val_scores, axis = 1)

plt.plot(param_range, train_mean, color = "b", marker = "o", label = "Обучение")
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color = "b", alpha = 0.15)
plt.plot(param_range, val_mean, color = "green", label = "Валидация", marker = "s")
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, color = "green", alpha = 0.15)

plt.title("Зависимость точности от степени регуляризации")
plt.ylabel("Accuracy")
plt.xlabel("Обратный коэффициент регуляризации")
plt.grid(True)
plt.legend(loc = "best")
plt.show()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(accuracy_score(pred, y_test))









