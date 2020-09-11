# Change Log:
# modified NN structure
# increased Dropout
# Added new Dense Layer
#
#
# Average accuracy: 0.9998 (99.98% accuracy)
#

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import load
from os import fsencode
from os import fsdecode

def createDataArrays(photos, labels):
    photos, labels = loadTrainingData(photos, labels)

    photos = asarray(photos)
    labels = asarray(labels)
    save('PNEUMONIA_or_NOT_photos.npy', photos)
    save('PNEUMONIA_or_NOT_labels.npy', labels)
    print(photos.shape, labels.shape)
    createModel(photos, labels)
def __main__():
    photos, labels = list(), list()
    createDataArrays(photos, labels)
    photos, labels = load_nparrays()

def load_test_data():
    test_photos, test_labels = list(), list()
    path = 'chest_xray/test/NORMAL/'
    folder = fsencode(path)
    for file in listdir(fsdecode(folder)):
        filename=fsdecode(file)
        photo = load_img(path+filename, grayscale=True, color_mode='grayscale', target_size=(200,200))
        photo= img_to_array(photo, None, None)
        test_photos.append(photo)
        output = 0.0
        test_labels.append(output)
    path = 'chest_xray/test/PNEUMONIA'
    folder = fsencode(path)
    for file in listdir(fsdecode(folder)):
        filename = fsdecode(file)
        photo = load_img(path+filename, grayscale=True, color_mode='grayscale', target_size=(200,200))
        photo = img_to_array(photo, None, None)
        test_photos.append(photo)
        output = 1.0
        test_labels.append(output)
    return test_photos, test_labels

def loadTrainingData(photos, labels):
    path = 'chest_xray/train/NORMAL/'
    folder = fsencode(path)
    for file in listdir(fsdecode(folder)):
        filename = fsdecode(file)
        photo = load_img(path+filename, grayscale=True, color_mode='grayscale', target_size=(200,200))
        photo = img_to_array(photo, None, None)
        photos.append(photo)
        output = 0.0
        labels.append(output)
    path = 'chest_xray/train/PNEUMONIA/'
    folder = fsencode(path)
    for file in listdir(fsdecode(folder)):
        filename = fsdecode(file)
        photo = load_img(path+filename, grayscale=True, color_mode='grayscale', target_size=(200,200))
        output = 1.0
        photo = img_to_array(photo, None, None)
        labels.append(output)
        photos.append(photo)

    return photos, labels




def load_nparrays():
    photos = load('PNEUMONIA_or_NOT_photos.npy')
    labels = load('PNEUMONIA_or_NOT_labels.npy')
    return photos, labels



def createModel(photos, labels):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation=tf.nn.relu, input_shape=(200,200,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],)
    model.fit(x = photos, y = labels, epochs = 250)
    test_photos, test_labels = load_test_data()
    model.evaluate(test_photos, test_labels)

    model.save('model.h5', model)

__main__()
