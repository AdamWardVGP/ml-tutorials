import pandas
from matplotlib import pyplot
from sklearn.cluster import KMeans

input_data = pandas.read_csv("Mall_Customers.csv")

# exclude "customer id" column 0, as it's not a real data point
# reduce to 2 dimensions just so we can visualize clustering
x_values = input_data.iloc[:, [3, 4]].values

# use "elbow" method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x_values)
    wcss.append(kmeans.inertia_)

pyplot.plot(range(1, 11), wcss)
pyplot.title("Elbow Method Evaluation")
pyplot.xlabel("# of clusters")
pyplot.ylabel('wcss')
pyplot.show()

# 5 looks good

# train k means on the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_cluster = kmeans.fit_predict(x_values)

# visualize
pyplot.scatter(x_values[y_cluster == 0, 0], x_values[y_cluster == 0, 1], s=50, c='red', label='cluster 1')
pyplot.scatter(x_values[y_cluster == 1, 0], x_values[y_cluster == 1, 1], s=50, c='blue', label='cluster 2')
pyplot.scatter(x_values[y_cluster == 2, 0], x_values[y_cluster == 2, 1], s=50, c='green', label='cluster 3')
pyplot.scatter(x_values[y_cluster == 3, 0], x_values[y_cluster == 3, 1], s=50, c='orange', label='cluster 4')
pyplot.scatter(x_values[y_cluster == 4, 0], x_values[y_cluster == 4, 1], s=50, c='cyan', label='cluster 5')
pyplot.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='centroid')
pyplot.title("KMeans Clusters")
pyplot.xlabel("Annual Income (in k$)")
pyplot.ylabel('Spending Score (1-100)')
pyplot.legend()
pyplot.show()
