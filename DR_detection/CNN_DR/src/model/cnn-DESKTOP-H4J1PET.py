import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.callbacks import History

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation,\
    GlobalAveragePooling2D, Add, Input, Lambda, multiply

# import keras.applications.densenet.preprocess_input as preprocess_input  # change this for different base models
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input

from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.applications.inception_v3 import InceptionV3
from keras.layers.pooling import AveragePooling2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

POOL_SIZE = (2, 2)
KERNEL_SIZE = (3, 3)
INPUT_SHAPE = (299, 299, 3)
TARGET_SIZE = (299, 299)
TRAIN = '..\\data\\train'
VAL = '..\\data\\val'
JSON_FILE = '..\\model\\model.json'
HDF5_FILE = '..\\model\\model.h5'
EPOCHS = 60
BATCH_SIZE = 15
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 33


def create_model():



    ''

    base_model = InceptionResNetV2(input_shape=INPUT_SHAPE, include_top=False,
                             weights='imagenet')    # change this for different base models
    #for layer in base_model.layers:
     #   layer.trainable = False
    x = base_model.output
    # Global Pooling

    x = GlobalAveragePooling2D()(x)


    #x = Dense(1024, activation='relu')(x)
    #x = BatchNormalization()(x)

    #x = Dropout(0.25)(x)
    # FC layers
    # model.add(Dropout(0.25))
    ''


    x = Dense(3, activation='softmax')(x)

    model = Model(input=base_model.input, output=x)

    '''

    base_model = InceptionV3(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential()
    for k in range(len(base_model.layers)):
        
        model.add(base_model.get_layer(index=k))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(3, activation='softmax'))
    
    '''



    #x = GlobalAveragePooling2D()(model.output)


    #x = Dense(3, activation='softmax')(model.output)
    #model2=Model(inputs=model.input,outputs=[x])
    return model


def init_model(model):
    # Adam optimizer
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    opt = keras.optimizers.Adam(lr=0.00005, decay=1e-6)
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
    history = History()

    TrainDataGen = ImageDataGenerator(  # rescale=1./255,
        rotation_range=180,
        shear_range=0.2, zoom_range=0.1,
        samplewise_std_normalization=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input)
    ValDataGen = ImageDataGenerator(  # rescale=1./255,
        rotation_range=180,
        shear_range=0.2, zoom_range=0.1,
        samplewise_std_normalization=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input)
    train_set = TrainDataGen.flow_from_directory(
        TRAIN,color_mode='rgb', target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
    test_set = ValDataGen.flow_from_directory(
        VAL,color_mode='rgb', target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)

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
