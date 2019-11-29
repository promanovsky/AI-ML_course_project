import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from common.tools import forest_regression_test, forest_classification_test, gradient_boosting_classification_test, \
    gradient_boosting_regression_test, showXGBTrainImportance
from decompositions import doPca2, doPca3, doPcaN, doTsne, doLda, doLle, doAE

curr_dir = os.path.abspath(os.curdir)
df = pd.read_csv(curr_dir + '/datasets/engineering_in_progress_out.csv')
print(df.shape)

columns_to_scale = list(df.columns)
columns_to_scale.remove('Cocktail Name')
columns_to_scale.remove('rating')
#print(len(columns_to_scale), columns_to_scale)

min_max_scaled = minmax_scale(df[columns_to_scale])
#min_max_scaled_columns = pd.DataFrame(min_max_scaled, columns=columns_to_scale)
#table = pd.DataFrame(pd.concat((df['Cocktail Name'], min_max_scaled_columns, df['rating']), axis = 1))
#table.to_csv(curr_dir +'/datasets/processed_minmax_scaler_dataset.csv', index=False)

scaler = StandardScaler()
scaler.fit(df[columns_to_scale])
standart_scaled = scaler.fit_transform(df[columns_to_scale])
#standart_scaled_columns = pd.DataFrame(standart_scaled, columns=columns_to_scale)
#table = pd.DataFrame(pd.concat((df['Cocktail Name'], standart_scaled_columns, df['rating']), axis = 1))
#table.to_csv(curr_dir +'/datasets/processed_standart_scaler_dataset.csv', index=False)

#showXGBTrainImportance(standart_scaled_columns, df['rating'], columns_to_scale)
#showXGBTrainImportance(min_max_scaled_columns, df['rating'], columns_to_scale)

# классификация по рейтингу предварительно его округлив
#Y = LabelEncoder().fit_transform(df['rating'].round())

#forest_classification_test(standart_scaled, Y)
#forest_classification_test(min_max_scaled, Y)

#gradient_boosting_classification_test(standart_scaled, Y)
#gradient_boosting_classification_test(min_max_scaled, Y)


# регрессия (предсказание рейтинга)
#Y = LabelEncoder().fit_transform(df['rating'])
#Y = LabelEncoder().fit_transform(df['rating'].round()) # повышает точность

#forest_regression_test(standart_scaled, Y)
#forest_regression_test(min_max_scaled, Y)

#gradient_boosting_regression_test(standart_scaled, Y)
#gradient_boosting_regression_test(min_max_scaled, Y)

#================================================================================================

scaled_data = standart_scaled
#scaled_data = min_max_scaled

def doPca_decomposition_demonstration():
    pca2, pca3, pca30 = doPca2(scaled_data), doPca3(scaled_data), doPcaN(scaled_data, 30)
    print('PCA CLASSIFICATION TESTS >>>>>>>>>>>>>>>>>>>>>>')
    Y = LabelEncoder().fit_transform(df['rating'].round())
    print('     PCA 2 components >>>>>>>>>>>')
    forest_classification_test(pca2, Y)
    gradient_boosting_classification_test(pca2, Y)

    print('     PCA 3 components >>>>>>>>>>>')
    forest_classification_test(pca3, Y)
    gradient_boosting_classification_test(pca3, Y)

    print('     PCA N=30 components >>>>>>>>>>>')
    forest_classification_test(pca30, Y)
    gradient_boosting_classification_test(pca30, Y)

    print('PCA REGRESSION TESTS >>>>>>>>>>>>>>>>>>>>>>')
    #Y = LabelEncoder().fit_transform(df['rating'])
    print('     PCA 2 components >>>>>>>>>>>')
    forest_regression_test(pca2, Y)
    gradient_boosting_regression_test(pca2, Y)

    print('     PCA 3 components >>>>>>>>>>>')
    forest_regression_test(pca3, Y)
    gradient_boosting_regression_test(pca3, Y)

    print('     PCA 30 components >>>>>>>>>>>')
    forest_regression_test(pca30, Y)
    gradient_boosting_regression_test(pca30, Y)

#doPca_decomposition_demonstration()

#================================================================================================
def do_tsne_decomposition_demonstration():
    tSNE = doTsne(scaled_data, 3)
    print('TSNE CLASSIFICATION TESTS >>>>>>>>>>>>>>>>>>>>>>')
    Y = LabelEncoder().fit_transform(df['rating'].round())
    forest_classification_test(tSNE, Y)
    gradient_boosting_classification_test(tSNE, Y)

    print('TSNE REGRESSION TESTS >>>>>>>>>>>>>>>>>>>>>>')
    #Y = LabelEncoder().fit_transform(df['rating'])
    forest_regression_test(tSNE, Y)
    gradient_boosting_regression_test(tSNE, Y)

#do_tsne_decomposition_demonstration()

def do_lda_decomposition_demonstration(n_components=3):
    Y = LabelEncoder().fit_transform(df['rating'].round())
    lda = doLda(scaled_data,Y, n_components)
    print('LDA with {} components CLASSIFICATION TESTS >>>>>>>>>>>>>>>>>>>>>>'.format(n_components))
    forest_classification_test(lda, Y)
    gradient_boosting_classification_test(lda, Y)

    print('LDA with {} components REGRESSION TESTS >>>>>>>>>>>>>>>>>>>>>>'.format(n_components))
    #Y = LabelEncoder().fit_transform(df['rating'])
    forest_regression_test(lda, Y)
    gradient_boosting_regression_test(lda, Y)

#do_lda_decomposition_demonstration(3)
#do_lda_decomposition_demonstration(30)

def do_lle_decomposition_demonstration(n_components=3):
    lle = doLle(scaled_data,n_components)
    print('LLE with {} components CLASSIFICATION TESTS >>>>>>>>>>>>>>>>>>>>>>'.format(n_components))
    Y = LabelEncoder().fit_transform(df['rating'].round())
    forest_classification_test(lle, Y)
    gradient_boosting_classification_test(lle, Y)

    print('LLE with {} components REGRESSION TESTS >>>>>>>>>>>>>>>>>>>>>>'.format(n_components))
    #Y = LabelEncoder().fit_transform(df['rating'])
    forest_regression_test(lle, Y)
    gradient_boosting_regression_test(lle, Y)

#do_lle_decomposition_demonstration(3)
#do_lle_decomposition_demonstration(30)

def do_AE_decomposition_demonstration():
    ae_data = doAE(scaled_data)
    print('AE DECOMPOSITION CLASSIFICATION TESTS >>>>>>>>>>>>>>>>>>>>>>')
    Y = LabelEncoder().fit_transform(df['rating'].round())
    forest_classification_test(ae_data, Y)
    gradient_boosting_classification_test(ae_data, Y)

    print('AE DECOMPOSITION REGRESSION TESTS >>>>>>>>>>>>>>>>>>>>>>')
    #Y = LabelEncoder().fit_transform(df['rating'])
    forest_regression_test(ae_data, Y)
    gradient_boosting_regression_test(ae_data, Y)

do_AE_decomposition_demonstration()

print('done')