import pandas as pd
import urllib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
import numpy as np
from operator import itemgetter
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


def linearReg(data):
    logreg = LogisticRegression()

    testScores = []
    valScores = []

    for i in range(1,6):
        foldnum = ('fold' + str(i))
        fold = data[foldnum]
        fold_train = fold['train']
        fold_val = fold['val']
        fold_test = fold['test']

        xtrain, ytrain = fold_train['x'], fold_train['y']
        xval, yval = fold_val['x'], fold_val['y']
        xtest, ytest = fold_test['x'], fold_test['y']

        logreg.fit(xtrain, ytrain.ravel())

        testScores.append(logreg.score(xtest, ytest))
        valScores.append(logreg.score(xval, yval))

    print('---------------------')
    print('     | Accuracy')
    print('Fold |  Val  |  Test ')
    print('---------------------')
    print('     |       |')
    for i in range(5):
        print(str(i+1) + '    |{:.5f}'.format(valScores[i]) + '|{:.5f}'.format(testScores[i]))
    print('---------------------')
    valAVG = (sum(valScores)/len(valScores))
    testAVG = (sum(testScores)/len(testScores))
    print('AVG  |{:.5f}'.format(valAVG) + '|{:.5f}'.format(testAVG))

    n_groups = 5
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = .8

    rects1 = plt.bar(index, valScores, bar_width, alpha=opacity, color = 'r', label = 'Val')
    rects2 = plt.bar(index + bar_width, testScores, bar_width, alpha=opacity, color='b', label='Test')

    plt.ylim(.91,.92)
    plt.xlabel("Fold")
    plt.ylabel('Score')
    plt.title('Linear Regression Accuracy Scores')
    plt.xticks(index + bar_width, ('1','2','3','4','5'))
    plt.legend()
    plt.tight_layout()
    plt.show()





