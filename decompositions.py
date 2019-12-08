import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from common.tools import forest_classification_test, gradient_boosting_classification_test, forest_regression_test, \
    gradient_boosting_regression_test, tree_classification_test


def doPca2(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    X = pca.transform(data)
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], cmap=plt.cm, edgecolor='k')
    plt.show()
    return X

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
    return X

def doPcaN(data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    X = pca.transform(data)
    return X

def doTsne(data, n_components):
    tsne = TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=300)
    X_tsne = tsne.fit_transform(data)
    return X_tsne

def doLda(data, Y, n_components):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit(data, Y).transform(data)
    return X_lda

def doLle(data, n_components):
    embedding = LocallyLinearEmbedding(n_components=n_components, eigen_solver='dense')
    X_lle = embedding.fit_transform(data)
    return X_lle

def doAE(data):
    input_layer = Input(shape=(data.shape[1],))
    encoded = Dense(3, activation='relu')(input_layer)
    decoded = Dense(data.shape[1], activation='softmax')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    X1, X2, Y1, Y2 = train_test_split(data, data, test_size=0.3, random_state=101)
    autoencoder.fit(X1, Y1,
                    epochs=100,
                    batch_size=300,
                    shuffle=True,
                    verbose = 30,
                    validation_data=(X2, Y2))

    encoder = Model(input_layer, encoded)
    X_ae = encoder.predict(data)
    return X_ae

def doPca_decomposition_demonstration(scaled_data, labels_df):
    pca2, pca3, pca30 = doPca2(scaled_data), doPca3(scaled_data), doPcaN(scaled_data, 30)
    print('PCA CLASSIFICATION TESTS >>>>>>>>>>>>>>>>>>>>>>')
    Y = LabelEncoder().fit_transform(labels_df)
    print('     PCA 2 components >>>>>>>>>>>')
    forest_classification_test(pca2, Y)
    tree_classification_test(pca2, Y)
    gradient_boosting_classification_test(pca2, Y)

    print('     PCA 3 components >>>>>>>>>>>')
    forest_classification_test(pca3, Y)
    tree_classification_test(pca3, Y)
    gradient_boosting_classification_test(pca3, Y)

    print('     PCA N=30 components >>>>>>>>>>>')
    forest_classification_test(pca30, Y)
    tree_classification_test(pca30, Y)
    gradient_boosting_classification_test(pca30, Y)

    print('PCA REGRESSION TESTS >>>>>>>>>>>>>>>>>>>>>>')
    print('     PCA 2 components >>>>>>>>>>>')
    forest_regression_test(pca2, Y)
    gradient_boosting_regression_test(pca2, Y)

    print('     PCA 3 components >>>>>>>>>>>')
    forest_regression_test(pca3, Y)
    gradient_boosting_regression_test(pca3, Y)

    print('     PCA 30 components >>>>>>>>>>>')
    forest_regression_test(pca30, Y)
    gradient_boosting_regression_test(pca30, Y)

def do_tsne_decomposition_demonstration(scaled_data, labels_df):
    tSNE = doTsne(scaled_data, 3)
    print('TSNE CLASSIFICATION TESTS >>>>>>>>>>>>>>>>>>>>>>')
    Y = LabelEncoder().fit_transform(labels_df)
    forest_classification_test(tSNE, Y)
    tree_classification_test(tSNE, Y)
    gradient_boosting_classification_test(tSNE, Y)

    print('TSNE REGRESSION TESTS >>>>>>>>>>>>>>>>>>>>>>')
    forest_regression_test(tSNE, Y)
    gradient_boosting_regression_test(tSNE, Y)

def do_lda_decomposition_demonstration(scaled_data, labels_df, n_components=3):
    Y = LabelEncoder().fit_transform(labels_df)
    lda = doLda(scaled_data,Y, n_components)
    print('LDA with {} components CLASSIFICATION TESTS >>>>>>>>>>>>>>>>>>>>>>'.format(n_components))
    forest_classification_test(lda, Y)
    tree_classification_test(lda, Y)
    gradient_boosting_classification_test(lda, Y)

    print('LDA with {} components REGRESSION TESTS >>>>>>>>>>>>>>>>>>>>>>'.format(n_components))
    forest_regression_test(lda, Y)
    gradient_boosting_regression_test(lda, Y)

def do_lle_decomposition_demonstration(scaled_data, labels_df, n_components=3):
    lle = doLle(scaled_data,n_components)
    print('LLE with {} components CLASSIFICATION TESTS >>>>>>>>>>>>>>>>>>>>>>'.format(n_components))
    Y = LabelEncoder().fit_transform(labels_df)
    forest_classification_test(lle, Y)
    tree_classification_test(lle, Y)
    gradient_boosting_classification_test(lle, Y)

    print('LLE with {} components REGRESSION TESTS >>>>>>>>>>>>>>>>>>>>>>'.format(n_components))
    forest_regression_test(lle, Y)
    gradient_boosting_regression_test(lle, Y)

def do_AE_decomposition_demonstration(scaled_data, labels_df):
    ae_data = doAE(scaled_data)
    print('AE DECOMPOSITION CLASSIFICATION TESTS >>>>>>>>>>>>>>>>>>>>>>')
    Y = LabelEncoder().fit_transform(labels_df)
    forest_classification_test(ae_data, Y)
    tree_classification_test(ae_data, Y)
    gradient_boosting_classification_test(ae_data, Y)

    print('AE DECOMPOSITION REGRESSION TESTS >>>>>>>>>>>>>>>>>>>>>>')
    forest_regression_test(ae_data, Y)
    gradient_boosting_regression_test(ae_data, Y)