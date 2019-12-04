#!/usr/bin/python3.7
import tensorflow as tf
from utils.feature import *
from models.CNN import *
from models.neighbor import *
from utils.specGraph import specDisplay


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
          'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

trainPath = 'Data/TrainingData/genres'
testPath = 'Data/TestingData/genres'

if __name__ == '__main__':

    # Loading audio samples to list
    trainData = loadData(70, trainPath)
    testData = loadData(30, testPath)
    specDisplay('Data/TrainingData/genres/blues/blues.00000.wav')

    print("Loaded raw data")

    # Running feature extraction to convert samples to melspectrograms
    x_train, genres_array_train, train2D = featureExtract(trainData, 70)
    x_test, genres_array_test, test2D = featureExtract(testData, 30)

    print("Converted data to melspectrograms")

    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(genres_array_train)
    y_test = tf.keras.utils.to_categorical(genres_array_test)

    n_features = x_train.shape[2]
    input_shape = (None, n_features)

    buildCNN(input_shape, x_train, x_test, y_train, y_test)

    print("Running kNN")

    knn(train2D,y_train)





