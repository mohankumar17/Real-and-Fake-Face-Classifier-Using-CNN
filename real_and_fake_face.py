from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
cnn_model = Sequential()

#input layers
cnn_model.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
#cnn_model.add(Dropout(0.25))
cnn_model.add(Flatten())

#hidden layers
cnn_model.add(Dense(128,activation='relu'))

#output layer
cnn_model.add(Dense(1,activation='sigmoid'))

#configuring the learning process
cnn_model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

'''test_set = test_datagen.flow_from_directory('dataset',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

cnn_model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''


#training the model
cnn_model.fit_generator(training_set,
                        samples_per_epoch = 8000,
                        nb_epoch = 15)


cnn_model.save('real_fake_face_model_2.h5')
