from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

data = load_iris()
columns = data.feature_names
X = data.data

df = pd.DataFrame(X, columns = columns)
df["class"] = data.target
print(df)

X = df.iloc[:, 0:4]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y, shuffle = True, test_size = 0.2)

ppt = Pipeline(
  [ ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=1))]
)

ppt.fit(X_train, y_train)

importances = ppt["classifier"].feature_importances_

feature_pairs = list(zip(columns, importances))
sorted_pairs = sorted(feature_pairs, key = lambda x : x[1], reverse = True)

plt.plot()

for name, val in sorted_pairs:
    print(f"{name}: {val}")

names = [pair[0] for pair in sorted_pairs]
values = [pair[1] for pair in sorted_pairs]

plt.bar(names,values)
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()

X_train_pca = X_train
X_test_pca = X_test
y_train_pca = y_train
y_test_pca = y_test

X = df.iloc[:, 2:4]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y, shuffle = True, test_size = 0.2)

ppt = Pipeline(
  [ ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=1,
                                      max_iter=10000))]
)

ppt_pca = Pipeline(
  [ ("scaler", StandardScaler()),
    ("PCA", PCA()),
    ("classifier", LogisticRegression(random_state=1,
                                      max_iter = 10000))]
)

piplines = zip([ppt, ppt_pca], [[X_train, y_train, X_test, y_test], [X_train_pca, y_train_pca, X_test_pca, y_test_pca]])
param_range = sts.loguniform(0.001, 10)
params_distribution = {"classifier__C" : param_range,
                       "classifier__solver" : ["lbfgs",
                                               "saga", "newton-cg"]
                        }

for pipline, data in piplines:

    rs = RandomizedSearchCV(estimator = pipline,
                            param_distributions=params_distribution,
                            n_jobs = -1,
                            cv = 5,
                            random_state=1,
                            n_iter = 30,
                            refit = True)

    rs.fit(data[0], data[1])

    print(f"Лучшая точность: {rs.best_score_}\nЛучшие параметры: {rs.best_params_}")
    clf = rs.best_estimator_

    train_sizes, train_scores, val_scores = learning_curve(
        estimator = clf,
        X = data[0],
        y = data[1],
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv = 5,
        n_jobs=-1,
        random_state=1,
        scoring = "accuracy"
    )

    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)
    val_mean = np.mean(val_scores, axis = 1)
    val_std = np.std(val_scores, axis = 1)

    plt.plot(train_sizes, train_mean, color = "blue", label = "Обучение", marker = "o")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color = "blue", alpha = 0.15)
    plt.plot(train_sizes, val_mean, color = "green", label = "Валидация", marker = "s")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color = "green", alpha = 0.15)

    plt.xlabel("Количество примеров")
    plt.ylabel("Accuracy")
    plt.title(f"Влияние количества обучающих примеров на точность для {type(clf).__name__}")
    plt.legend(loc = "best")
    plt.grid(True)
    plt.show()

    y_pred = clf.predict(data[2])

    confmat = confusion_matrix(y_true = data[3],
                               y_pred = y_pred)
    
    print(confmat)



