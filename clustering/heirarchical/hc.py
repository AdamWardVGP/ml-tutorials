import pandas
from matplotlib import pyplot
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

input_data = pandas.read_csv("Mall_Customers.csv")

# exclude "customer id" column 0, as it's not a real data point
# reduce to 2 dimensions just so we can visualize clustering
x_values = input_data.iloc[:, [3, 4]].values

# use dendrogram to determine optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(x_values, method='ward'))
pyplot.title("Dendrogram")
pyplot.xlabel("Observation points (Customers)")
pyplot.ylabel('Variance (Euclidean Distance)')
pyplot.show()

# 5 looks good, 3 is also close

# train k means on the dataset
hierarchical_cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_cluster = hierarchical_cluster.fit_predict(x_values)

# visualize
pyplot.scatter(x_values[y_cluster == 0, 0], x_values[y_cluster == 0, 1], s=50, c='red', label='cluster 1')
pyplot.scatter(x_values[y_cluster == 1, 0], x_values[y_cluster == 1, 1], s=50, c='blue', label='cluster 2')
pyplot.scatter(x_values[y_cluster == 2, 0], x_values[y_cluster == 2, 1], s=50, c='green', label='cluster 3')
pyplot.scatter(x_values[y_cluster == 3, 0], x_values[y_cluster == 3, 1], s=50, c='orange', label='cluster 4')
pyplot.scatter(x_values[y_cluster == 4, 0], x_values[y_cluster == 4, 1], s=50, c='cyan', label='cluster 5')
pyplot.title("Hierarchical Clusters")
pyplot.xlabel("Annual Income (in k$)")
pyplot.ylabel('Spending Score (1-100)')
pyplot.legend()
pyplot.show()
