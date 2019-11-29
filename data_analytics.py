import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from common.tools import forest_regression_test, forest_classification_test, gradient_boosting_classification_test, \
    gradient_boosting_regression_test

curr_dir = os.path.abspath(os.curdir)
df = pd.read_csv(curr_dir + '/datasets/engineering_in_progress_out.csv')
print(df.shape)

columns_to_scale = list(df.columns)
columns_to_scale.remove('Cocktail Name')
columns_to_scale.remove('rating')
print(len(columns_to_scale), columns_to_scale)

min_max_scaled = minmax_scale(df[columns_to_scale])
min_max_scaled_columns = pd.DataFrame(min_max_scaled, columns=columns_to_scale)
table = pd.DataFrame(pd.concat((df['Cocktail Name'], min_max_scaled_columns, df['rating']), axis = 1))
table.to_csv(curr_dir +'/datasets/processed_minmax_scaler_dataset.csv', index=False)

scaler = StandardScaler()
scaler.fit(df[columns_to_scale])
standart_scaled = scaler.fit_transform(df[columns_to_scale])
standart_scaled_columns = pd.DataFrame(standart_scaled, columns=columns_to_scale)
table = pd.DataFrame(pd.concat((df['Cocktail Name'], standart_scaled_columns, df['rating']), axis = 1))
table.to_csv(curr_dir +'/datasets/processed_standart_scaler_dataset.csv', index=False)

def doPca3(pca_data):
    pca = PCA(n_components=3)
    pca.fit(pca_data)
    X = pca.transform(pca_data)
    print(X.shape)
    fig = plt.figure(1, figsize=(40, 30))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=10, azim=45)
    plt.cla()
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm, edgecolor='k')
    plt.show()

doPca3(standart_scaled_columns)
doPca3(min_max_scaled_columns)

def doPca2(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    X = pca.transform(data)
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], cmap=plt.cm, edgecolor='k')
    plt.show()

doPca2(standart_scaled_columns)
doPca2(min_max_scaled_columns)

def showXGBTrainImportance(data, targetColumn, feature_columns, needSave=False):
    xgbTrainData = xgb.DMatrix(data, targetColumn, feature_names=feature_columns)
    param = {'max_depth':7, 'objective':'reg:linear', 'eta':0.2}
    model = xgb.train(param, xgbTrainData, num_boost_round=300)
    xgb.plot_importance(model, grid ="false", max_num_features=30, height=0.5)
    if needSave:
        plt.savefig('feature importance param'+str(np.random.randint(0, 100))+'.pdf',size=1024, format='pdf',bbox_inches="tight")
    plt.show()

showXGBTrainImportance(standart_scaled_columns, df['rating'], columns_to_scale)
showXGBTrainImportance(min_max_scaled_columns, df['rating'], columns_to_scale)

# классификация по рейтингу предварительно его округлив
Y = LabelEncoder().fit_transform(df['rating'].round())

forest_classification_test(standart_scaled, Y)
#forest_classification_test(min_max_scaled, Y)

gradient_boosting_classification_test(standart_scaled, Y)
#gradient_boosting_classification_test(min_max_scaled, Y)


# регрессия (предсказание рейтинга)
Y = LabelEncoder().fit_transform(df['rating'])

forest_regression_test(standart_scaled, Y)
#forest_regression_test(min_max_scaled, Y)

gradient_boosting_regression_test(standart_scaled, Y)
#gradient_boosting_regression_test(min_max_scaled, Y)

print('done')