import os
import numpy as np

files = os.listdir('.')
for each in files:
    if each.endswith('.npy'):
        data_json = np.load(os.path.join(os.getcwd(),each), allow_pickle=True)
        data = data_json.item()
        print(1)

