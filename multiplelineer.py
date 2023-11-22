import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("insurance.csv")

# y ekseni
expenses = data.expenses.values.reshape(-1, 1)

# x ekseni
ageBmis = data.iloc[:, [0,2]].values
regression = LinearRegression()
regression.fit(ageBmis, expenses)

print(regression.predict([[20,20]]))
print(regression.predict([[20,21]]))
print(regression.predict([[20,22]]))
print(regression.predict([[30,20]]))