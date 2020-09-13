import random
from tensorflow.keras.models import load_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, Embedding
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import load


def loadData():
    dog_breeds = list()
    photos, labels = list(), list()
    output = 0.0
    path = 'DogBreedDataset/images/Images/'
    for dir in listdir(path):
        dog_breeds.append([dir[10::], output])
        for file in listdir(path + dir):
            photo = load_img(path + dir + '/' + file, grayscale=True, color_mode='grayscale', target_size=(150, 150))
            photo = img_to_array(photo, None, None)
            if len(photos) != 0:
                x = random.randint(0, len(photos) - 1)
                photos.insert(x, photo)
                labels.insert(x, output)
            else:
                photos.append(photo)
                labels.append(output)
        output += 1.0
    print(dog_breeds)
    with open('DogBreedOutputsKey.txt', 'w+') as file:
        for i in dog_breeds:
            file.write(str(i[0]))
            file.write(' ')
            file.write(str(i[1]))
            file.write('\n')
    for i in range(20):
        print(labels[i])

    photos = asarray(photos)
    labels = asarray(labels)

    return photos, labels


def createModel(photos, labels):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(150, 150, 1)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.softmax))
    model.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
    model.fit(x=photos, y=labels, epochs=50)
    model.save('model_save.h5', model)
    model.save_weights('model_weights.h5', model)


def __main__():
    photos, labels = loadData()
    print(photos.shape)
    print(labels.shape)

    save('DogBreedPhotos.npy', photos)
    save('DogBreedLabels.npy', labels)

    photos = load('DogBreedPhotos.npy')
    labels = load('DogBreedLabels.npy')
    createModel(photos, labels)

    model = load_model('model.h5')
    pred = model.predict(photos[0].reshape(1, 150, 150, 1))
    print(pred.argmax())


__main__()
