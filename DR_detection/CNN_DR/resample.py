import random
import os
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

data_dir = 'data\\train'
data = os.listdir(data_dir)
table = []
for i in data:
    PATH = os.path.join(data_dir, i)
    img_list = os.listdir(PATH)
    table.append(len(img_list))
table = np.array(table)
# while (np.unique(table).size>1):
image_num = np.amin(table)
for label in data:
    root = os.path.join(data_dir, label)
    print(label)
    imgList = os.listdir(root)
    # print(len(imgList))
    remove = random.sample(imgList, len(imgList)-image_num)
    for item in remove:
        os.remove(os.path.join(root, item))
