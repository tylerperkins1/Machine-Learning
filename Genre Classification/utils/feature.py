import os
import librosa
import numpy as np

genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
          'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}


# Loads song samples from data folders in to an array
def loadData(numColumns, path):
    samples = []
    for i in range(10):
        column = []
        for j in range(numColumns):
            column.append('empty')
        samples.append(column)

    i = 0
    j = 0

    for file in os.listdir(path):
        x = os.listdir(path + '/' + file)
        j = 0
        for entry in x:
            samples[i][j] = (path + '/' + file + '/' + entry)
            j = j + 1
        i = i + 1

    return samples


def featureExtract(samples, size):

    genres_array = []
    features = []
    features2d = []

    for i in range(10):
        for j in range(size):
            genres_array.append(i)
            song = samples[i][j]

            y,sr = librosa.load(song)

            hop_length = 512

            # Convert song to melspectrogram, then to dB, then take covariance
            spectro = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop_length, n_mels=32)
            spectro = librosa.power_to_db(spectro, ref=np.max)
            covariance = np.cov(np.array(spectro))

            feature = np.stack((covariance),axis=1)
            features.append(feature)

            # Creating 2D version for kNN use
            cov1d = covariance.flatten()
            features2d.append(cov1d)

    specs = np.array(features)
    specs2d = np.array(features2d)

    return specs, genres_array, specs2d
