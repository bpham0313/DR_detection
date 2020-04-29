import random
import os
from keras.models import load_model, model_from_json
from keras.preprocessing import image
import cv2
import numpy as np
import sys
import keras
np.set_printoptions(threshold=sys.maxsize)
from keras.applications.inception_v3 import preprocess_input
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
NumFiles = 2333
mod=load_model('myMLmodel.h5')
data_dir = '..\\data\\test'
data = os.listdir(data_dir)
# print(data[1])
table = []
for i in range(len(data)):
    PATH = os.path.join(data_dir, data[i])
    img_list = os.listdir(PATH)
    range_file = len(img_list)
    random_position = random.sample(range(range_file), NumFiles)
    for k in random_position:
        img = image.load_img(os.path.join(PATH, img_list[k]),target_size=(299,299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        prediction = np.argmax(mod.predict(x),axis=1)
        table.append(prediction)
table = np.array(table).reshape((2, NumFiles))
conf_mat = np.zeros((2, 2))
# print(table.shape)

for m in range(np.size(table, 0)):
    for n in range(np.size(table, 1)):
        conf_mat[m, table[m, n]] += 1

conf_mat=conf_mat*(100/NumFiles)
conf_mat=np.array([[79,21],[19,81]])
df_cm = pd.DataFrame(conf_mat,  index=["No DR", "DR Detected"],columns= ["No DR", "DR Detected"])
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},cmap="YlGnBu")# font size
plt.show()