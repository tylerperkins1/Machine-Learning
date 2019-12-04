from keras.models import Model
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from utils.graphs import *
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.optimizers import Adam
from keras import regularizers


genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
          'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
N_LAYERS = 3
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 56
BATCH_SIZE = 32
LSTM_COUNT = 96
EPOCH_COUNT = 200
NUM_HIDDEN = 64
L2_regularization = 0.001


def buildCNN(input_shape, x_train, x_test, y_train, y_test):
    #   Build CNN model
    model_input = Input(input_shape, name='input')
    layer = model_input
    for i in range(N_LAYERS):
        # give name to the layers
        layer = Conv1D(
            filters=CONV_FILTER_COUNT,
            kernel_size=FILTER_LENGTH,
            kernel_regularizer=regularizers.l2(L2_regularization),
            name='convolution_' + str(i + 1)
        )(layer)
        layer = BatchNormalization(momentum=0.9)(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling1D(2, padding="same")(layer)
        layer = Dropout(0.4)(layer)

    ## LSTM Layer
    layer = LSTM(LSTM_COUNT, return_sequences=False)(layer)
    layer = Dropout(0.4)(layer)

    ## Dense Layer
    layer = Dense(NUM_HIDDEN, kernel_regularizer=regularizers.l2(L2_regularization), name='dense1')(layer)
    layer = Dropout(0.4)(layer)

    ## Softmax Output
    layer = Dense(10)(layer)
    layer = Activation('softmax', name='output_realtime')(layer)
    model_output = layer
    model = Model(model_input, model_output)
    opt = Adam(lr=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    cnn = model

    print(model.summary())

    hist = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
                        validation_data=(x_test, y_test), verbose=1)

    score = cnn.evaluate(x_test, y_test, verbose=0)
    print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))

    save_history(hist, 'logs/evaluate.png')

    preds = np.argmax(cnn.predict(x_test), axis=1)
    y_orig = np.argmax(y_test, axis=1)
    cm = confusion_matrix(preds, y_orig)

    keys = OrderedDict(sorted(genres.items(), key=lambda t: t[1])).keys()
    plot_confusion_matrix('logs/cm.png', cm, keys, normalize=True)

    cnn.save('models/.h5')