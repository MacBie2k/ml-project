import time

import numpy as np
import os
import cv2
import random
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from tensorflow._api.v2.v2 import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from sklearn.preprocessing import LabelEncoder


def create_training_data(categories, data_directory, training_data, img_size):
    for category in categories:
        path = os.path.join(data_directory, category)
        class_number = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                if img_array is not None:
                    scaled_array = cv2.resize(img_array, (img_size, img_size))
                    training_data.append([scaled_array, class_number])
                else:
                    print(f'Image {img} in path {path} could not be read')
            except Exception as e:
                print('Exception occurred:', e)


def shuffle_training_data(training_data):
    random.shuffle(training_data)


def pickle_data(training_data, img_size):
    X = []
    y = []
    label_encoder = LabelEncoder()
    for features, label in training_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, img_size, img_size, 3)
    y = label_encoder.fit_transform(y)
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def prepare_data():
    training_data = []
    data_directory = ".../GarbageTypes"
    categories = ["cardboard", "glass", "metal", "paper", "plastic"]
    img_size = 224
    create_training_data(categories, data_directory, training_data, img_size)
    shuffle_training_data(training_data)
    pickle_data(training_data, img_size)


def train_model():
    img_size = 224
    name = "garbarge-types-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(name))
    X = pickle.load(open("X.pickle", "rb"))
    y = pickle.load(open("y.pickle", "rb"))
    print('Number of samples in X:', len(X))
    print('Number of labels in y:', len(y))
    y = tf.keras.utils.to_categorical(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    model = Sequential()
    base_model = tf.keras.applications.VGG19(input_shape=(img_size, img_size, 3),
                                             include_top=False,
                                             weights='imagenet')
    base_model.trainable = False
    model.add(base_model)
    model.add(keras.layers.GlobalAveragePooling2D())

    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(5, activation='softmax'))

    opt = RMSprop(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=5,
              callbacks=[early_stopping, tensorboard])

    model.save('{}.model'.format(name),  save_traces=True)


if __name__ == '__main__':
    #prepare_data()
    train_model()
    print('program end')
