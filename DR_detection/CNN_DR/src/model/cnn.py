import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.callbacks import History

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation,\
    GlobalAveragePooling2D, Add, Input, Lambda, multiply,ZeroPadding2D,Convolution2D

from keras.applications.inception_v3 import preprocess_input   # change this for different base models
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model


from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.applications.inception_v3 import InceptionV3
from keras.layers.pooling import AveragePooling2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

POOL_SIZE = (2, 2)
KERNEL_SIZE = (3, 3)
INPUT_SHAPE = (299, 299,3)
TARGET_SIZE = (299, 299)
TRAIN = '..\\data\\train'
VAL = '..\\data\\test'
JSON_FILE = '..\\model\\model.json'
HDF5_FILE = '..\\model\\model.h5'
EPOCHS = 60
BATCH_SIZE = 30
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 22
weights_path='..\\data\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def create_model():

    '''
    base_model=InceptionV3(include_top=False,weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)


    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(2,activation='softmax')(x)

    #for layer in base_model.layers[:200]:
    #b    layer.trainable = False
    '''
    #model= Model(input=base_model.input, output=x)
    base_model = InceptionV3(include_top=False, weights='imagenet')
    model= Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2,activation='softmax'))

    return model


def init_model(model):
    # Adam optimizer
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_model_history(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def train_model(model):

    trainDataGen = ImageDataGenerator(#rescale=1./255,
        #rotation_range=180,
        shear_range=0.1, zoom_range=0.1,
        #samplewise_std_normalization=True,
        preprocessing_function=preprocess_input,
        vertical_flip=True)

    testDataGen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_set = trainDataGen.flow_from_directory(
        TRAIN, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
    test_set = testDataGen.flow_from_directory(
        VAL, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)

    history = model.fit_generator(train_set, steps_per_epoch=STEPS_PER_EPOCH,
                                  epochs=EPOCHS, validation_steps=VALIDATION_STEPS, validation_data=test_set, verbose=2)

    # Save model

    '''
    model_json = model.to_json()
    with open(JSON_FILE, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(HDF5_FILE)
    '''
    model.save('myMLmodel.h5')
    plot_model_history(history)

    return history
