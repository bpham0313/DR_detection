import keras
import numpy as np
import sys
np.set_printoptions(threshold= sys.maxsize )
from keras.models import load_model,model_from_json
import cv2


with open('..\\model\\model.json', 'r') as f:
    mod= model_from_json(f.read())
mod.load_weights('..\\model\\model.h5')

opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
mod.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
img=cv2.imread('290_right.jpeg')
img = cv2.resize(img, (212, 212))
img = np.reshape(img, [1, 212, 212, 3])
prediction = mod.predict_classes(img)
print(prediction)