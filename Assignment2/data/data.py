import pandas as pd
import urllib.request
from sklearn.model_selection import train_test_split
import os
import numpy as np
from operator import itemgetter
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def loadData():
    if not os.path.exists('./Skin_NonSkin.txt'):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'
        urllib.request.urlretrieve(url, './Skin_NonSkin.txt')

    df = pd.read_csv('Skin_NonSkin.txt', sep='\t', names=['B', 'G', 'R', 'skin'])
    df.head()
    # NO MISSING VALUES
    df.isna().sum()

#   Standardize

    feature = df[df.columns[~df.columns.isin(['skin'])]]  # Except Label
    label = (df[['skin']] == 1) * 1  # Converting to 0 and 1 (this col has values 1 and 2)
    feature = feature / 255.  # Pixel values range from 0-255 converting between 0-1

    feature.head()
    label.head()

    x = feature.values
    y = label.values

    # We will keep fix test and take 5 cross validation set
    # so we will have five different data set
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=1)

    # Lets see the size of xtrain, xtest
    len(xtrain), len(xtest)

    # 5 Fold Split
    # First merge xtrain and ytrain so that we can easily divide into 5 chunks

    data = np.concatenate([xtrain, ytrain], axis=1)
    # Observe the shape of array
    xtrain.shape, ytrain.shape, data.shape

    # Divide our data to 5 chunks
    chunks = np.split(data, 5)

    datadict = {
        'fold1': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}},
        'fold2': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}},
        'fold3': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}},
        'fold4': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}},
        'fold5': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}}, }

    for i in range(5):
        datadict['fold' + str(i + 1)]['val']['x'] = chunks[i][:, 0:3]
        datadict['fold' + str(i + 1)]['val']['y'] = chunks[i][:, 3:4]

        idx = list(set(range(5)) - set([i]))
        X = np.concatenate(itemgetter(*idx)(chunks), 0)
        datadict['fold' + str(i + 1)]['train']['x'] = X[:, 0:3]
        datadict['fold' + str(i + 1)]['train']['y'] = X[:, 3:4]

    folder = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.join(folder,'data.pkl')

    writepickle(datadict, filePath)

def writepickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def readpickle(filename):
    folder = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.join(folder,filename)
    with open(filePath, 'rb') as f:
        data = pickle.load(f)
    return data