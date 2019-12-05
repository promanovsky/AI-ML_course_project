from sklearn import metrics
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, MiniBatchKMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

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

"""
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.fit_transform(X)
y = LabelEncoder().fit_transform(y)
print('Y',list(y))
"""

def draw_inertia_for_clusterization_method(method, label):
    inertia = []
    for k in range(1, 10):
        m = method(n_clusters=k, random_state=42).fit(X)
        inertia.append(np.sqrt(m.inertia_))

    plt.plot(range(1, 10), inertia, marker='s')
    plt.xlabel('$k$')
    plt.ylabel('$J(C_k)$')
    plt.title(label)
    plt.show()

draw_inertia_for_clusterization_method(KMeans, 'KMeans')
draw_inertia_for_clusterization_method(MiniBatchKMeans, 'MiniBatchKMeans')

algorithms = []
algorithms.append(KMeans(n_clusters=7, random_state=42))
algorithms.append(SpectralClustering(n_clusters=7, random_state=42, affinity='nearest_neighbors'))
algorithms.append(AgglomerativeClustering(n_clusters=7))
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

results = pd.DataFrame(data=data, columns=['ARI', 'AMI', 'Homogenity', 'Completeness', 'V-measure', 'Silhouette'],
                       index=['K-means', 'Spectral', 'Agglomerative', 'MiniBatchKMeans'])

print(results)

#row_test = df_test.iloc[:100,]
test_val = df_test.to_numpy()

def draw_confustion_matrix(pred, n_clusters):
    mat = confusion_matrix(y.round(), pred.round())
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels = range(n_clusters),
                yticklabels = range(n_clusters))
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

for alg, n_clusters in zip(algorithms, [7,7,7,5]):
    pred = alg.fit_predict(X)
    print(alg.labels_[pred])
    #draw_confustion_matrix(y[pred], n_clusters)