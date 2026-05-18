from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from io import StringIO

# csv_data = """А,В,С,О
# 1.0,2.0,3.0,4.0
# 5.0,6.0,,8.0
# 10.0, 11.0, 12.0,"""

# df = pd.read_csv(StringIO(csv_data))
# imr = SimpleImputer(missing_values=np.nan, strategy="mean")
# imr = imr.fit(df.values)
# imputed_data = imr.transform(df.values)
# print(imputed_data)

# csv_data = """А,В,С,О
# 1.0,2.0,3.0,4.0
# 5.0,6.0,,8.0
# 10.0, 11.0, 12.0,"""

# df = pd.read_csv(StringIO(csv_data))
# sc = StandardScaler()
# data_scaled = sc.fit_transform(df)
# knnim = KNNImputer(missing_values=np.nan, n_neighbors=2, weights="uniform")
# data_imputed = knnim.fit_transform(data_scaled)

# data_final = sc.inverse_transform(data_imputed)

# print(data_final)


df = pd.DataFrame([["green", "M", 10.1, "class2"],
                  ["red", "L", 13.5, "class1"],
                  ["blue", "XL", 15.3, "class2"]]
                  )

df.columns = ["colour", "size", "price", "classlabel"]
print(df)

size_mapping = {"XL" : 3,
                "L" : 2,
                "M" : 1}

inv_size_mapping = {value : key for key, value in size_mapping.items()}
df["size"] = df["size"].map(size_mapping)
print(df)
df["size"] = df["size"].map(inv_size_mapping)
print(df)

class_mapping = {label : idx for idx, label in enumerate(np.unique(df["classlabel"]))}
inv_class_mapping = {value : key for key, value in class_mapping.items()}
df["classlabel"] = df["classlabel"].map(class_mapping)
print(df)
df["classlabel"] = df["classlabel"].map(inv_class_mapping)
print(df)

class_le = LabelEncoder()
y = class_le.fit_transform(df["classlabel"].values)
print(y)

# можем также закодировать цвета. Но проблема - теперь алгоритмы могут сравнивать признаки (зеленый и синий)

from sklearn.preprocessing import OneHotEncoder

color_ohe = OneHotEncoder()
X = df[["colour", "size", "price"]].values

color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

print(pd.get_dummies(df[["price", "colour", "size"]], drop_first=True))

df = pd.DataFrame([["green", "M", 10.1, "class2"],
                  ["red", "L", 13.5, "class1"],
                  ["blue", "XL", 15.3, "class2"]]
                  )

df.columns = ["colour", "size", "price", "classlabel"]

df["size > M"] = df["size"].apply(lambda x: x in ("L", "XL"))
df["size > L"] = df["size"].apply(lambda x: x == "XL")
df.drop(columns=['size'], inplace = True)
print(df)