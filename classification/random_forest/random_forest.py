# Multiple Linear Regression

# Importing the libraries
import numpy as np
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(f'x:\n{x}\n\ny:\n{y}')

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# normalize the data set:
training_scaler = StandardScaler()
x_train = training_scaler.fit_transform(x_train)
# same scaler is used for the test set to ensure its values are the same range
x_test = training_scaler.transform(x_test)

print(f'x:\n{x_train}\n\ny:\n{y_train}')

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

# Running a sample test will also need to be transformed
x_sample = training_scaler.transform([[30, 87000]])
sample_prediction = classifier.predict(x_sample)
print(f'sample prediction: {sample_prediction}')

y_predicted = classifier.predict(x_test)

print(y_predicted)

# Evaluating the Model Performance
confusion_m = confusion_matrix(y_test, y_predicted)

print(confusion_m)
print(f'accuracy {accuracy_score(y_test, y_predicted)}')

ConfusionMatrixDisplay(confusion_m).plot()
pyplot.show()

# visualize training data
x_set, y_set = training_scaler.inverse_transform(x_train), y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=.5),
                     np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=5))

pyplot.contourf(x1, x2,
                classifier.predict(training_scaler.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
                alpha=0.75, cmap=ListedColormap(('red', 'green')))

pyplot.xlim(x1.min(), x1.max())
pyplot.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    pyplot.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)

pyplot.title('Random Forest (Training set)')
pyplot.xlabel('Age')
pyplot.ylabel('Estimated Salary')
pyplot.legend()
pyplot.show()

# visualize test data
x_set, y_set = training_scaler.inverse_transform(x_test), y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=.5),
                     np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=5))
pyplot.contourf(x1, x2, classifier.predict(training_scaler.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
                alpha=0.75, cmap=ListedColormap(('red', 'green')))
pyplot.xlim(x1.min(), x1.max())
pyplot.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    pyplot.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
pyplot.title('Random Forest (Test set)')
pyplot.xlabel('Age')
pyplot.ylabel('Estimated Salary')
pyplot.legend()
pyplot.show()
