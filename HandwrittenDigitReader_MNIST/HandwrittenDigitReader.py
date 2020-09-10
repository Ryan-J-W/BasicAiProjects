#import necessary libraries
import tensorflow as tf #used to download the dataset 
import matplotlib.pyplot as plt #used to display the images in a GUI window
from tensorflow.keras.models import Sequential #the model used in this script is a Sequential() type model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D #these are the layers that we'll be using to construct our CNN(convolutional neural network)


#download the dataset and save it to the described variables
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#this block picks a random image from the data set and displays the image selected
image_index = 7777 #random number
print(y_train[image_index]) #print the value of this image
plt.imshow(x_train[image_index], cmap='Greys') #plot image onto a graph
plt.show() #show image
#make sure to close this window so the rest of the script can continue
#note: the CMD line printed value should match the number shown on the graph

#Resahpe the dependent data into a 4 dimensional array to be used later
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#the shape of each input image
input_shape = (28,28,1)

#converting to type float32 for rescaling purposes
x_train = x_train.astype('float32')
x_test=x_test.astype('float32')

#rescaling data into either a 1 or a 0 instead of any value between 0 and 255
x_train /=255
x_test /=255

#print the shape of the training data for visual validation
print('x_train shape: ', x_train.shape)
print('Number of images in x_train: ', x_train.shape[0]) #number of training images
print('Number of images in x_test: ', x_test.shape[0]) #number of testing images


#create a new model of type Sequential
model = Sequential()

#add a new layer to the NN
#this layer is a Convolutional layer that will breakdown the image into many 3x3 pixel parts for later processing
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))

#Once the convolutional layer breaks down the image into features, the max pooling will select the Maximum value of each part to be passed on
# this creates a new 2x2 dimensional matrix
model.add(MaxPooling2D(pool_size=(2,2)))

#Flatten the matrix into a 1 dimensional array
model.add(Flatten())

#add a new Dense layer to the model
model.add(Dense(128, activation=tf.nn.relu))

#drop some data to prevent over fitting
model.add(Dropout(0.2))

#add a new Dense layer
model.add(Dense(10, activation=tf.nn.softmax))

#compile the model with the 'adam' optimizer and 'sparse categorical crossentropy' to compute the loss in accuracy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

#fit the model to the data and train for 100 passes over the data
#this is more than the necessary number of passes
#10 epochs would be plenty
model.fit(x=x_train, y=y_train, epochs=100)

#evaluates the model based on the testing data
model.evaluate(x_test, y_test)

#save the model for later use so we don't have to retrain the model
model.save('model.h5')


#the following code segment is used soley for testing purposes

#pick a random image
image_index = 4444

#plot image on graph
plt.imshow(x_test[image_index], cmap='Greys')

#show graph
plt.show()

#use the model to predict which digit the image is
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))

#prints what the model predicts the shown image to be
print(pred.argmax())

if pred.argmax() == y_test[image_index]:
  print("The model correctly predicted the value of the shown digit")
  
else:
  print("The model did not correctly predict the value of the shown digit")
