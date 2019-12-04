from sklearn import metrics
import pandas as pd
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, \
    FeatureAgglomeration
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

"""
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

df = df_csv.loc[df_csv['rating'] != 0]

df_test = df_csv.loc[df_csv['rating'] == 0]
df_test = df_test[columns]

X = df[columns].to_numpy()
y = df['rating'].to_numpy()
print(X.shape, y.shape)

algorithms = []
algorithms.append(KMeans(n_clusters=6, random_state=42))
algorithms.append(AffinityPropagation())
algorithms.append(SpectralClustering(n_clusters=4, random_state=42,
                                     affinity='nearest_neighbors'))
algorithms.append(AgglomerativeClustering(n_clusters=4))
#algorithms.append(FeatureAgglomeration(n_clusters=4))

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

results = pd.DataFrame(data=data, columns=['ARI', 'AMI', 'Homogenity',
                                           'Completeness', 'V-measure',
                                           'Silhouette'],
                       index=['K-means', 'Affinity',
                              'Spectral', 'Agglomerative'])

print(results)

row_test = df_test.iloc[:100,]
test_val = row_test.to_numpy()
#ind = algorithms[0].predict(test_val)
res = algorithms[0].fit_predict(test_val)
print('predicted ratings >>>', res)
#print(algorithms[0].labels_)

"""
distance_mat = pdist(X) # pdist посчитает нам верхний треугольник матрицы попарных расстояний

Z = hierarchy.linkage(distance_mat, 'single') # linkage — реализация агломеративного алгоритма
plt.figure(figsize=(10, 5))
dn = hierarchy.dendrogram(Z, color_threshold=0.5)
plt.show()
"""