"""
Revision history
01 - 1st version
"""
__CreatedBy__ = "Patrick Wan"
__Date__ = "16-Sep-2022"
__Revision__ = "01"

import numpy as np
import os

import pandas as pd

import Config
import cv2
from PIL import Image
import json

from sklearn.model_selection import train_test_split


#

"""
[Data Structure]

 Level 1: ID  
 Level 2: ID + Video No + Frame No
 Level 3: Index 0 (Original): Image + Mask + Coordinates
          Index 1-N (Augmented) : Image + Mask + Coordinates
"""


def load_npz(file_name):
    dict_all = np.load(file_name, allow_pickle=True)
    loop = 0
    # for key, val in sorted(dict_test.items()):
    for key, val in dict_all.items():
        # ID
        loop += 1
        print('#{}: ID:{}, total:{}'.format(loop, key, val.shape[0]))
    """
        for i in range(val.shape[0]):
            # file-name that belongs to the same ID
            print(val[i, 0])
            for idx in val[i, 1].keys():
                # 0- original, augmented from 1 to augment_rounds
                print(idx)
                # image
                print(val[0, 1][idx][0].shape)
                # mask
                print(val[0, 1][idx][1].shape)
                # points
                print(val[0, 1][idx][2].shape)
    """
    return dict(dict_all)


def train_valid_test():
    files = config["data_file"]["files"]
    print(files)

    list_of_dicts = []
    for file in files:
        print(file)
        data_dict = load_npz(file)
        list_of_dicts.append(data_dict)

    dataset_dicts = {k: v for list_item in list_of_dicts for (k, v) in list_item.items()}

    X = list(dataset_dicts.keys())
    y = list(dataset_dicts.values())

    # In the first step we will split the data in training and remaining dataset
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)

    # Now since we want the valid and test size to be equal (10% each of overall data).
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    test_size = 0.5
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


with open("Config.json", "r") as f:
    config = json.load(f)

if __name__ == "__main__":
    train_valid_test()

