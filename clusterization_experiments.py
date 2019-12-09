from sklearn import metrics
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import seaborn as sns
from decompositions import doTsne, doLle, doPcaN
from sklearn.neighbors import NearestNeighbors

sns.set()

"""
Решение проблемы немаркированных данных через построение кластеризации (для анализа) и определения 
ближайшего соседа к неразмеченной точке

https://habr.com/ru/company/ods/blog/325654/
https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
"""
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 50)

curr_dir = os.path.abspath(os.curdir)
df_csv = pd.read_csv(curr_dir + '/datasets/grouped_columns.csv')

columns = list(df_csv.columns)
columns.remove('Cocktail Name')
columns.remove('rating')

def draw2D_points(data, labels, title):
    plt.figure(figsize=(12,10))
    plt.scatter(data[:, 0], data[:, 1], c=labels,
                edgecolor='none', alpha=0.7, s=40,
                cmap=plt.cm.get_cmap('nipy_spectral', 10))
    plt.colorbar()
    plt.title(title)
    plt.show()

df = df_csv.loc[df_csv['rating'] != 0]

df_test = df_csv.loc[df_csv['rating'] == 0]
df_test = df_test[columns]

X = df[columns].to_numpy()
y = df['rating'].to_numpy()
print(X.shape, y.shape)

def draw_inertia_for_clusterization_method(method, title):
    inertia = []
    for k in range(1, 10):
        m = method(n_clusters=k, random_state=42).fit(X)
        inertia.append(np.sqrt(m.inertia_))

    plt.plot(range(1, 10), inertia, marker='s')
    plt.xlabel('$k$')
    plt.ylabel('$J(C_k)$')
    plt.title(title)
    plt.show()

draw_inertia_for_clusterization_method(KMeans, 'KMeans')
draw_inertia_for_clusterization_method(MiniBatchKMeans, 'MiniBatchKMeans')

draw2D_points(doPcaN(X, 2), y, 't-SNE')
draw2D_points(doPcaN(df_csv[columns].to_numpy(), 2), df_csv['rating'].to_numpy(), 'pca 2 components')

draw2D_points(doTsne(X, 2), y, 't-SNE')
draw2D_points(doTsne(df_csv[columns].to_numpy(), 2), df_csv['rating'].to_numpy(), 't-SNE')

draw2D_points(doLle(X, 2), y, 'lle')
draw2D_points(doLle(df_csv[columns].to_numpy(), 2), df_csv['rating'].to_numpy(), 'lle')

algorithms = []
algorithms.append(KMeans(n_clusters=7, random_state=42))
algorithms.append(MiniBatchKMeans(n_clusters=5))

data = []
for algo in algorithms:
    algo.fit(X)
    data.append(({
        'ARI': metrics.adjusted_rand_score(y, algo.labels_),
        'AMI': metrics.adjusted_mutual_info_score(y, algo.labels_),
        'Homogenity': metrics.homogeneity_score(y, algo.labels_),
        'Completeness': metrics.completeness_score(y, algo.labels_),
        'V-measure': metrics.v_measure_score(y, algo.labels_),
        'Silhouette': metrics.silhouette_score(X, algo.labels_)}))

results = pd.DataFrame(data=data,
                       columns=['ARI', 'AMI', 'Homogenity', 'Completeness', 'V-measure', 'Silhouette'],
                       index=['K-means', 'MiniBatchKMeans'])

print(results)

decomposition_methods = [doPcaN, doTsne, doLle]
data = []
for algo in algorithms:
    for decomp in decomposition_methods:
        X_decomp = decomp(X, 2)
        algo.fit(X_decomp)
        draw2D_points(X, algo.labels_, str(type(algo)) + ' ' + str(decomp))


# Для каждой неразмеченной точки найдем N ближайших соседа и примем за метку среднее значение их рейтингов
N_NEIGHBORS = 3
nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='ball_tree').fit(X)
X_unmarked = df_test.to_numpy()
graph = nbrs.kneighbors_graph(X_unmarked).toarray()
print(len(graph), X_unmarked.shape)

def calibrate_rating(val):
    if val > 5:
        return 5
    if 4.5 <= val < 5:
        return 4.5
    if 4 <= val < 4.5:
        return 4
    if 3.5 <= val < 4:
        return 3.5
    if 3 <= val < 3.5:
        return 3
    if 2.5 <= val < 3:
        return 2.5
    return 2

marked_rating = df['rating'].to_numpy()
ratings = []
for i,gr in enumerate(graph):
    rating = 0
    for j, val in enumerate(gr):
        if val!=0:
            # print('for point with index',i, 'closest neighbour from marked data is', j, 'with rating', marked_rating[j])
            rating = rating + marked_rating[j]
    ratings.append(calibrate_rating(rating / N_NEIGHBORS))

df_unmarked = df_csv.loc[df_csv['rating'] == 0]
df_unmarked['rating'] = ratings
df_marked = df_csv.loc[df_csv['rating'] != 0]

result = pd.concat([df_marked, df_unmarked])
print(result.shape)
print(result.sample(10))
print(result.loc[result['rating'] == 0])

result.to_csv(curr_dir +'/datasets/grouped_columns_filled_with_clusterization.csv', index=False)
print('done')