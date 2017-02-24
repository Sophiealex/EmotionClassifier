import numpy as np

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers import LocallyConnected2D


class NewEmotionClassifier:

    def __init__(self, num_classes, save_path=''):
        """ Constructor for EmotionClassifier that builds placeholders and the learning model.
        :param num_classes: The number of different classifications.
        :type num_classes: int
        :param save_path: A file path for the session variables to be saved. If not set the session will not be saved.
        :type save_path: A file path.
        """
        self.keep_prob = 0.5
        self.model = self.build_model(num_classes)
        self.save_path = save_path

    def build_model(self, num_classes, activation=False):
        model = Sequential()

        model.add(Convolution2D(64, 5, 5, 'normal', border_mode='valid', subsample=(1,1), input_shape=(88, 88, 1)))
        if activation: model.add(Activation('relu'))
        model.add(MaxPooling2D((3,3),(2,2)))

        model.add(Convolution2D(64, 5, 5, 'normal', border_mode='valid', subsample=(1, 1)))
        if activation: model.add(Activation('relu'))
        model.add(MaxPooling2D((3, 3), (2, 2)))

        model.add(LocallyConnected2D(32, 3, 3, 'normal', border_mode='valid', subsample=(1, 1)))
        if activation: model.add(Activation('relu'))
        model.add(Dropout(self.keep_prob))

        model.add(LocallyConnected2D(32, 3, 3, 'normal', border_mode='valid', subsample=(1, 1)))
        if activation: model.add(Activation('relu'))
        model.add(Dropout(self.keep_prob))

        model.add(Flatten())
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        return model

    def train(self, data, epochs=100, batch_size=128):
        sgd = SGD(lr=0.0001, momentum=0.9, decay=0.0005, nesterov=False)
        optimizer = Adam()
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        x, y = [m[0] for m in data], [n[1] for n in data]

        self.model.fit(np.asarray(x), np.asarray(y), batch_size, epochs, 1, validation_split=0.2, shuffle=True)
        history = self.model.save(self.save_path)
        print history
