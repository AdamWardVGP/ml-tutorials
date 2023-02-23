import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# File import and column location
dataset = pandas.read_csv('Salary_Data.csv')

# separate columns to feature vector and dependant vector
feature_vector = dataset.iloc[:, :-1].values
dependant_vector = dataset.iloc[:, -1]

# separate training data from test data
train_feature, test_feature, train_target, test_target = \
    train_test_split(feature_vector, dependant_vector, test_size=0.2, random_state=1)

# standardize data
# std_sc = StandardScaler()
# train_feature = std_sc.fit_transform(train_feature)
# test_feature = std_sc.transform(test_feature)

print(train_feature)
print(train_target)

print(test_feature)
print(test_target)

# train the simple linear regression model
lr = LinearRegression()
lr.fit(train_feature, train_target)

# Visualize Training
plt.scatter(train_feature, train_target, color='red')
# plot draws a series of line segments connecting the points where the train_feature are the x values
# and predict returns the y values. Although they are separate collections each index in the two arrays
# are treated as pairs i.e. [1,2,3] [5,20,13] becomes (1,5), (2,20), (3,13)
plt.plot(train_feature, lr.predict(train_feature), color='blue')
plt.title = 'Salary vs Experience (Training Set)'
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize Test
plt.scatter(test_feature, test_target, color='red')
plt.plot(train_feature, lr.predict(train_feature), color='blue')
plt.title = 'Salary vs Experience (Test Set)'
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Linear regressions are good only if there is a linear correlation between the features and targets

# Once we've plotted our data we can make predictions with it
# Note the predict takes a 2d array, hence wrapping it twice [[x]]
print(f'Prediction for 12 {lr.predict([[12]])}')

# Or we can find the coefficients
print(f'Coefficient {lr.coef_}')
print(f'Intercept {lr.intercept_}')
