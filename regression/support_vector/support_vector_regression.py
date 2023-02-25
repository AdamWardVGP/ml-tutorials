import pandas
import sklearn.svm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from matplotlib import pyplot
import numpy

# Open data file
dataset = pandas.read_csv("Position_Salaries.csv")

# retrieve needed columns
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# reshape y because standard scaler needs a 2d array
y = y.reshape(len(y), 1)

print(f'x:\n{x}\n\ny:\n{y}')

# Scaling is required when they have an implicit relationship between the dependent variable (y) and the features (x)
x_scaler = StandardScaler()
x_scaled = x_scaler.fit_transform(x)
# Because the model is implicit, feature scaling is needed for Y also, this is because we need values closer to one
# another so that a feature is not ignored. Large differences between dependant variable and features should scale
# the dependant variable
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

print(f'After Scaling----\n\nx:\n{x_scaled}\n\ny:\n{y_scaled}')

regressor = SVR(kernel='rbf')
regressor.fit(x_scaled, y_scaled)

predict_x_transformed = regressor.predict(x_scaler.transform([[6.5]]))
predict_y = y_scaler.inverse_transform(predict_x_transformed.reshape(-1, 1))

print(f'y predicted {predict_y}')

X_grid = numpy.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
pyplot.scatter(x, y, color='red')
y_grid_predictions = regressor.predict(x_scaler.fit_transform(X_grid)).reshape(-1, 1)
pyplot.plot(X_grid, y_scaler.inverse_transform(y_grid_predictions), color='blue')
pyplot.scatter([[6.5]], [[predict_y]], color='green')
pyplot.title('Truth or Bluff (Support Vector Regression)')
pyplot.xlabel('Position level')
pyplot.ylabel('Salary')
pyplot.show()
