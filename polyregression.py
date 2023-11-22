import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("positions.csv")

# Seviye göre maaş bulma seviye artarsa maaş ne olur?

# x ekseni
level = data.iloc[:,1].values.reshape(-1, 1)

# y ekseni
salary = data.iloc[:,2].values.reshape(-1, 1)

regression = LinearRegression()
regression.fit(level, salary)

tahmin = regression.predict([[8.3]])
print(tahmin)

plt.scatter(level, salary, color = "red")
plt.plot(level, regression.predict(level), color = "blue")


regressionPoly = PolynomialFeatures(degree = 4)

levelPoly = regressionPoly.fit_transform(level) # Level değerlerini polinom haline getirir.

_regression = LinearRegression()
_regression.fit(levelPoly, salary)

_tahmin = _regression.predict(regressionPoly.transform(np.array([8.3]).reshape(-1, 1)))
print(_tahmin)

plt.scatter(level, salary, color = "red")
plt.plot(level, regression.predict(level), color = "blue")
plt.plot(level, _regression.predict(levelPoly), color = "purple")
plt.xlabel('Level')
plt.ylabel("Salary")
plt.show()
