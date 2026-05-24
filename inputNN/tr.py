import torch
import pandas as pd
import numpy as np
from numpy import interp
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, matthews_corrcoef, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.utils import resample

# data = load_iris()

# print(data.feature_names)
# print(data.data.shape)

# features = np.hstack((data.data, data.target.reshape(-1, 1)))
# target = data.target

# columns = data.feature_names + ["class"]

# df = pd.DataFrame(features, columns=columns)

# print(df["class"].unique())

# print(df)

logr = Pipeline(
    [("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("model", LogisticRegression(penalty = "l2",
                       random_state = 1,
                       solver = "lbfgs",
                       C = 100.0))]
)

X, y = make_classification(
    n_samples=1000, 
    n_features=5, 
    n_classes=2, 
    weights=[0.95, 0.05], 
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify = y, test_size = 0.2)
cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
fig = plt.figure(figsize = (7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train_idx, test_idx) in enumerate(cv):

    X_tr, y_tr = X[train_idx], y[train_idx]

    X_tr_class_0 = X_tr[y_tr == 0]
    X_tr_class_1 = X_tr[y_tr == 1]

    X_tr_class_1_resampled = resample(
        X_tr_class_1,
        replace = True,
        n_samples = len(X_tr_class_0),
        random_state = 1
    )
    
    X_tr_resampled = np.vstack((X_tr_class_0, X_tr_class_1_resampled))
    y_tr_resampled = np.hstack((np.zeros(len(X_tr_class_0)), np.ones(len(X_tr_class_1_resampled)))).astype(int)

    probas = logr.fit(X_tr_resampled, y_tr_resampled).predict_proba(X_train[test_idx])
    fpr, tpr, threshholds = roc_curve(y_train[test_idx],
                                      probas[:, 1],
                                      pos_label = 1)
    interp_tpr = interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    mean_tpr += interp_tpr
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             label = f"Выборка ROC {i+1} (area = {roc_auc:.2f})")
    
plt.plot([0, 1], [0, 1],
            linestyle = "--",
            color = "red",
            label = "Случайный выбор (area = 0.5)")

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, "k--",
            label = f"Средн. ROC (area = {mean_auc:.2f})", lw = 2)

plt.plot([0, 0, 1],
            [0, 1, 1],
            linestyle = ":",
            color = "black",
            label = "Идеальная производительность (area = 1.0)")

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("Доля ложноположительных прогнозов")
plt.ylabel("Доля истинно положительных прогнозов")
plt.legend(loc = "lower right")

### 



