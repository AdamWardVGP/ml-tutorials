import pandas
from sklearn import ensemble
import numpy
from matplotlib import pyplot

dataset = pandas.read_csv("Position_Salaries.csv")

# retrieve needed columns
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

regressor = ensemble.RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)

predict_y = regressor.predict([[6.5]])
print(f'predicted salary {predict_y}')

x_grid = numpy.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
pyplot.scatter(x, y, color='red')
pyplot.plot(x_grid, regressor.predict(x_grid), color='blue')
pyplot.scatter([[6.5]], [[predict_y]], color='green')
pyplot.title('Truth or Bluff (Random Forest Regression)')
pyplot.xlabel('Position level')
pyplot.ylabel('Salary')
pyplot.show()
