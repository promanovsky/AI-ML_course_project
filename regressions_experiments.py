from sklearn import metrics
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

from common.tools import forest_regression_test, gradient_boosting_regression_test
from decompositions import doTsne, doLle, doPcaN

"""
Решение проблемы немаркированных данных через построение регрессионной модели определения 
предсказания рейтинга для неразмеченной точки
"""

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 50)

curr_dir = os.path.abspath(os.curdir)
df_csv = pd.read_csv(curr_dir + '/datasets/grouped_columns.csv')
print('Total dataset shape', df_csv.shape)

columns = list(df_csv.columns)
columns.remove('Cocktail Name')
columns.remove('rating')

df = df_csv.loc[df_csv['rating'] != 0]

df_test = df_csv.loc[df_csv['rating'] == 0]
df_test = df_test[columns]
print('Unmarked dataset shape',df_test.shape)

#df = df_csv

X = df[columns].to_numpy()
y = df['rating'].to_numpy()
print('Marked dataset shape',X.shape, y.shape)

scaler = StandardScaler()
scaler.fit(X)
standart_scaled = scaler.fit_transform(X)
Y = LabelEncoder().fit_transform(y)

forest_regression_test(standart_scaled, Y)
gradient_boosting_regression_test(standart_scaled, Y)

X = standart_scaled

def draw_accuracy(data, step, title):
    plt.plot(data[0], data[1], marker='s')
    plt.xlabel('step = '+ str(step))
    plt.ylabel('accuracy')
    plt.title(title)
    plt.show()

data = [[],[]]
for i in range(1, 11):
    print('>>>>> Neighbors count={}'.format(i))
    neigh = KNeighborsRegressor(n_neighbors=i, metric='euclidean')
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.1, random_state = 101)
    neigh.fit(X_Train, Y_Train)
    prediction = neigh.predict(X_Test)
    #print('KNeighborsRegressor mean_squared_error', mean_squared_error(Y_Test, prediction))
    #print("KNeighborsRegressor precision = {}".format(precision_score(Y_Test, prediction.round(), average='macro')))
    #print("KNeighborsRegressor recall = {}".format(recall_score(Y_Test, prediction.round(), average='macro')))
    acc = accuracy_score(Y_Test, prediction.round())
    print("KNeighborsRegressor accuracy = {}".format(acc))
    data[0].append(i)
    data[1].append(acc)

draw_accuracy(data, 1, 'KNeighborsRegressor')
# 1 или 10 - оптимальные параметры исходя из графика
NEIGHBORS_COUNT = 9
decomposition_methods = [doPcaN, doTsne, doLle]
data = [[],[]]
for ind, decomp in enumerate(decomposition_methods):
    print('decomposition with method', decomp)
    X_decomp = decomp(X, 2)
    neigh = KNeighborsRegressor(n_neighbors=NEIGHBORS_COUNT, metric='euclidean')
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_decomp, Y, test_size = 0.1, random_state = 101)
    neigh.fit(X_Train, Y_Train)
    prediction = neigh.predict(X_Test)
    #print('KNeighborsRegressor-1 mean_squared_error', mean_squared_error(Y_Test, prediction))
    #print("KNeighborsRegressor-1 precision = {}".format(precision_score(Y_Test, prediction.round(), average='macro')))
    #print("KNeighborsRegressor-1 recall = {}".format(recall_score(Y_Test, prediction.round(), average='macro')))
    acc = accuracy_score(Y_Test, prediction.round())
    print("KNeighborsRegressor accuracy = {}".format(acc))
    data[0].append(ind)
    data[1].append(acc)

fig, ax = plt.subplots()
ax.scatter(data[0], data[1])

for i, m in enumerate(['PCA', 'tSNE', 'LLE']):
    ax.annotate(m, (data[0][i], data[1][i]))
plt.show()

# Выберем подходящий метод декомпозиции
decomp = doTsne

neigh = KNeighborsRegressor(n_neighbors=NEIGHBORS_COUNT, metric='euclidean')
X_decomp = decomp(X, 2)
neigh.fit(X_decomp, Y)

X_unlabeled = df_test.to_numpy()
scaler.fit(X_unlabeled)
standart_scaled = scaler.fit_transform(X_unlabeled)
decomp_data = decomp(standart_scaled, 2)
pred = neigh.predict(decomp_data)

ratings = [5 if r > 5 else r for r in pred]
ratings = [4.5 if 4.5 <= r < 5 else r for r in ratings]
ratings = [4 if 4 <= r < 4.5 else r for r in ratings]
ratings = [3.5 if 3.5 <= r < 4 else r for r in ratings]
ratings = [3 if 3 <= r < 3.5 else r for r in ratings]
ratings = [2.5 if 2.5 <= r < 3 else r for r in ratings]

df_unmarked = df_csv.loc[df_csv['rating'] == 0]
df_unmarked['rating'] = ratings
df_marked = df_csv.loc[df_csv['rating'] != 0]

result = pd.concat([df_marked, df_unmarked])
# print(result.shape)
# print(result.sample(10))
# print(result.loc[result['rating'] == 0])

result.to_csv(curr_dir +'/datasets/grouped_columns_filled_with_regression.csv', index=False)
print('done')