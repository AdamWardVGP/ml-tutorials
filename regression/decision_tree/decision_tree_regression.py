import pandas
from sklearn import tree
import numpy
from matplotlib import pyplot

dataset = pandas.read_csv("Position_Salaries.csv")

# retrieve needed columns
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

classifier = tree.DecisionTreeClassifier(random_state=1)
classifier.fit(x, y)

predict_y = classifier.predict([[6.5]])
print(f'predicted salary {predict_y}')

# tree.plot_tree(decision_tree=classifier)
#
x_grid = numpy.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
pyplot.scatter(x, y, color='red')
pyplot.plot(x_grid, classifier.predict(x_grid), color='blue')
pyplot.scatter([[6.5]], [[predict_y]], color='green')
pyplot.title('Truth or Bluff (Support Vector Regression)')
pyplot.xlabel('Position level')
pyplot.ylabel('Salary')
pyplot.show()
