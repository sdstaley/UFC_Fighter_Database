import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

# sean staley

df = pd.read_csv(r'C:\Users\notse\OneDrive\GSU\processed_data.csv')
print(df)

########################################################################################################################

# Viewing a scatter plot of the first 2 principal components blue fighter
ax = df.plot.scatter(x='principal component 1B', y='principal component 2B')
plt.show(ax)

# Viewing a scatter plot of the first 2 principal components red fighter
ax = df.plot.scatter(x='principal component 1R', y='principal component 2R')
plt.show(ax)

########################################################################################################################
df = df.dropna()
ks = range(1, 10)
inertias = []
points = df[['principal component 1B', 'principal component 2B']]
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(df[['principal component 1B', 'principal component 2B']])

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)



plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

########################################################################################################################
X = df[['principal component 1B', 'principal component 2B']].values
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()

y_km = kmeans.fit_predict(X)
plt.scatter(X[y_km==0, 0], X[y_km==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_km==1, 0], X[y_km==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_km==2, 0], X[y_km==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_km==3, 0], X[y_km==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.show()

########################################################################################################################
# performing k-means on red fighter data
ks = range(1, 10)
inertias = []
points = df[['principal component 1R', 'principal component 2R']]
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(df[['principal component 1R', 'principal component 2R']])

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)



plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

########################################################################################################################
# plotting red fighter data for k-means
X = df[['principal component 1R', 'principal component 2R']].values
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()

y_km = kmeans.fit_predict(X)
plt.scatter(X[y_km==0, 0], X[y_km==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_km==1, 0], X[y_km==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_km==2, 0], X[y_km==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_km==3, 0], X[y_km==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.show()

########################################################################################################################

# randomly sampling 100 rows from the dataframe to reduce the size of dendrogram on both fighters
newDF = df.sample(n=100)

# hierarchical clustering on blue fighter data
X = newDF[['principal component 1B', 'principal component 2B']].values

dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('pca')
plt.ylabel('Euclidean distances')
plt.show()

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='ward')
y_hc=hc.fit_predict(X)

plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', label ='Cluster 5')
plt.title('(Hierarchical Clustering Model)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

########################################################################################################################

# hierarchical clustering on red fighter data
X = newDF[['principal component 1R', 'principal component 2R']].values

dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('pca')
plt.ylabel('Euclidean distances')
plt.show()

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='ward')
y_hc=hc.fit_predict(X)

plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', label ='Cluster 5')
plt.title('(Hierarchical Clustering Model)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

########################################################################################################################