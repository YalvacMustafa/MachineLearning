import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("positions.csv")

level = data.iloc[:, 1].values.reshape(-1, 1)
salary = data.iloc[:, 2].values

regression = RandomForestRegressor(n_estimators = 10, random_state=0) #estimators = kaç tane decision tree oluşturcak ona karar verir.
regression.fit(level, salary) #random_state sürekli aynı algoritma çalıştırır. Değer değişmez.

print(regression.predict([[8.3]]))