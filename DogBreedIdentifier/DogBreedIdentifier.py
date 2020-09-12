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
    photos, labels = list(), list()
    output = 0.0
    path = 'DogBreedDataset/images/Images/'
    for dir in listdir(path):
        for file in listdir(path+dir):
            photo = load_img(path+dir+'/'+file, grayscale = True, color_mode = 'grayscale', target_size = (150, 150))
            photo = img_to_array(photo, None, None)
            photos.append(photo)
            labels.append(output)
        output += 1.0
    photos = asarray(photos)
    labels = asarray(labels)
    return photos, labels


def createModel(photos, labels):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation=tf.nn.relu, input_shape = (150,150,1)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.softmax))
    model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'spars e_categorical_crossentropy')
    model.fit(x = photos, y = labels, epochs = 50)

    model.save_weights('model.h5', model)

def __main__():
    photos, labels = loadData()
    print(photos.shape)
    print(labels.shape)

    save('DogBreedPhotos.npy', photos)
    save('DogBreedLabels.npy', labels)

    photos = load('DogBreedPhotos.npy')
    labels = load('DogBreedLabels.npy')
    createModel(photos, labels)

    model = load('model.h5')
    pred = model.predict(photos[0].reshape(1,150,150,1))
    print(pred.argmax())


__main__()


