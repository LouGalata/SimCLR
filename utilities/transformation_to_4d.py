import os

import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
import json


ROOT_FOLDER = '/home/lougalata/Desktop/SingleSide'
label_path = os.path.join(os.pardir,'datasets', 'lab_data', 'Singleside_classnames.txt')
DEST_PATH = 'SingleSide_4D-2'

FINAL_HEIGHT = 480
FINAL_WIDTH = 480


if __name__ == "__main__":
    filenames = [os.path.join(ROOT_FOLDER, x) for x in os.listdir(ROOT_FOLDER)]
    ccnt = len(filenames) - 1  # Minus the new folder with the new 4d data
    if not os.path.exists(os.path.join(ROOT_FOLDER, DEST_PATH)):
        os.makedirs(os.path.join(ROOT_FOLDER, DEST_PATH))

    height, width, _ = cv2.imread(filenames[10]).shape

    tuples = []
    for x in filenames:
        if 'rgb' in x:
            for y in filenames:
                if os.path.isfile(x) and os.path.isfile(y):
                    if x != y:
                        if x.split('-')[0] == y.split('-')[0]:
                            tuples.append([x, y])

    arrays = []
    tuples = sorted(tuples, key=lambda tup: int(os.path.basename(tup[0]).split('-')[0]))

    for idx, (x, y) in enumerate(tuples):
        img_rgb = cv2.imread(x)
        try:
            img_d = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
            height, width, _ = img_rgb.shape
            img_rgb = cv2.resize(img_rgb, (FINAL_HEIGHT, FINAL_WIDTH))
            img_d = cv2.resize(img_d, (FINAL_HEIGHT, FINAL_WIDTH))
            img_d = np.expand_dims(img_d, axis=-1)
        except Exception as e:
            print(str(e))

        arrays.append((idx, np.concatenate((img_rgb, img_d), axis=-1)))

    with open(label_path) as lp:
        meta = json.load(lp)

    for cnt, fp in arrays:
        label = [item for item in meta if (item['img_id'] == str(cnt) + '-rgb')][0]

        arraynp = np.array(fp)

        # fig = plt.figure(figsize=(10, 10))
        # plt.axis('off')
        # plt.title(label['category'])
        # plt.imshow(fp[..., 0:3])
        # plt.show()

        filepath = os.path.join(ROOT_FOLDER, DEST_PATH, str(cnt) + '.npy')
        np.save(filepath, arraynp)
