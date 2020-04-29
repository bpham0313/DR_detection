import random
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

data_dir = "..\\data\\train"
data = os.listdir(data_dir)
print(data)
table = []

for i in data:
    PATH = os.path.join(data_dir, i)
    img_list = os.listdir(PATH)
    range_file = len(img_list)
    table.append(range_file)
#table = np.array(table)

print(table)
plt.bar(data, table)
plt.title("UNBALANCED TRAINING DATA")
plt.show()
