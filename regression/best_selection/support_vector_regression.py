# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
x_train = sc_X.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)

# Training the SVR model on the Training set
regressor = SVR(kernel='rbf')
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_predicted = sc_y.inverse_transform(regressor.predict(sc_X.transform(x_test)).reshape(-1, 1))
np.set_printoptions(precision=2)
print(np.concatenate((y_predicted.reshape(len(y_predicted), 1), y_test.reshape(len(y_test), 1)), 1))

# Evaluating the Model Performance
print(f'r2={r2_score(y_test, y_predicted)}')
