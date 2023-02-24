import matplotlib.pyplot
import numpy
from matplotlib.pyplot import plot
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import pandas

dataset = pandas.read_csv("50_Startups.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)

# Encode the categorical data
encoder = ('encoder', OneHotEncoder(), [3])
column_transform = ColumnTransformer(
    transformers=[encoder], remainder='passthrough'
)
x = numpy.array(column_transform.fit_transform(x))

# columns are added at the beginning
print(x)

# No need to Remove 1 of the one hot encoded columns, sklearn will do it automatically
# No feature scaling needed
# No P calculation needed
# No backward elimination not needed, sklearn will figure out the best features to predict the dependant variable

# Split training set and test set
x_training_set, x_test_set, y_training_set, y_test_set = \
    train_test_split(x, y, test_size=0.2, random_state=0)

# Train Multiple Linear Regression model on the data set
linear_regressor = LinearRegression()
linear_regressor.fit(x_training_set, y_training_set)

# Run test sets
y_predicted = linear_regressor.predict(x_test_set)

numpy.set_printoptions(precision=2)

# reshape to be vertical for concatenation
y_predicted_vertical = y_predicted.reshape(len(y_predicted), 1)
y_test_vertical = y_test_set.reshape(len(y_test_set), 1)

# Horizontal concatenation of vertical data
results_compared = numpy.concatenate((y_predicted_vertical, y_test_vertical), axis=1)

print(f"Results compared\n{results_compared}")

print(f"predicted {linear_regressor.predict([[1, 0, 0, 160000, 130000, 300000]])}")
print(linear_regressor.coef_)
print(linear_regressor.intercept_)
