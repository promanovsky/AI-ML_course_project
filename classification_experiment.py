from sklearn import metrics
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

from common.tools import forest_classification_test, \
    tree_classification_test, gradient_boosting_classification_test
from decompositions import doTsne, doLle, doPcaN

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 50)

curr_dir = os.path.abspath(os.curdir)
df_csv = pd.read_csv(curr_dir + '/datasets/grouped_columns.csv')

columns = list(df_csv.columns)
columns.remove('Cocktail Name')
columns.remove('rating')

"""
df = df_csv.loc[df_csv['rating'] != 0]

df_test = df_csv.loc[df_csv['rating'] == 0]
df_test = df_test[columns]
"""
df = df_csv

X = df[columns].to_numpy()
y = df['rating'].to_numpy()
print(X.shape, y.shape)

scaler = StandardScaler()
scaler.fit(X)
standart_scaled = scaler.fit_transform(X)
Y = LabelEncoder().fit_transform(y)

forest_classification_test(standart_scaled, Y)
tree_classification_test(standart_scaled, Y)
gradient_boosting_classification_test(standart_scaled, Y)

def draw_accuracy(data, step, title):
    plt.plot(data[0], data[1], marker='s')
    plt.xlabel('step = '+ str(step))
    plt.ylabel('accuracy')
    plt.title(title)
    plt.show()

data = [[],[]]
