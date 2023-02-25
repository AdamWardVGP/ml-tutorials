import pandas
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot
import numpy

dataset = pandas.read_csv("Position_Salaries.csv")

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

# Simple Linear Regressor
regressor = LinearRegression()
regressor.fit(x, y)

pyplot.scatter(x, y, color="red")
pyplot.plot(x, regressor.predict(x), color="blue")
pyplot.title('Truth or Bluff (Linear Regression)')
pyplot.xlabel('Position Level')
pyplot.ylabel('Salary')
pyplot.show()

# Called linear regression because of the linear combination of features,
# the coefficients are linear just not the operation performed on the terms

poly_features = PolynomialFeatures(degree=4)
x_poly = poly_features.fit_transform(x)

poly_regressor = LinearRegression()
poly_regressor.fit(x_poly, y)

pyplot.scatter(x, y, color="red")
pyplot.plot(x, poly_regressor.predict(x_poly), color="blue")
pyplot.title('Truth or Bluff (Polynomial Regression)')
pyplot.xlabel('Position Level')
pyplot.ylabel('Salary')
pyplot.show()

# higher resolution
X_grid = numpy.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
pyplot.scatter(x, y, color='red')
pyplot.plot(X_grid, poly_regressor.predict(poly_features.fit_transform(X_grid)), color='blue')
pyplot.title('Truth or Bluff (Polynomial Regression)')
pyplot.xlabel('Position level')
pyplot.ylabel('Salary')
pyplot.show()

# bad prediction with linear
print(f'linear prediction {regressor.predict([[6.5]])}')

# good prediction with poly
print(f'poly prediction {poly_regressor.predict(poly_features.transform([[6.5]]))}')
