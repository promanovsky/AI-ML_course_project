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