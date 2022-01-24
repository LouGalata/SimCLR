import logging
import os
import os.path
import numpy as np
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import VisionDataset
from PIL import Image
import cv2.cv2 as cv2
import pandas as pd
from sklearn import preprocessing
import csv
import json
from copy import deepcopy
import matplotlib.pyplot as plt

GRAYSCALE = False


class LaboratoryDataset(VisionDataset):
    base_folder = 'lab_data'
    url = None  # maybe the .tar.gz
    filename = 'lab_data.gz.tar'  # "stl10_binary.tar.gz"

    class_names_file = 'class_names.txt'
    folds_list_file = 'fold_indices.txt'
    label_list = ["Flat", "Crumbled", "Folded", "Half-folded"]
    texture_list = ["TableclothTwoVerticalStripes", "RectangleTowel", "TvTablecloth", "TowelHorizontalStripes", "SmallPlaid", "SquareWhiteTowel", "BeigeTablecloth"]

    splits = ('train', 'train+unlabeled', 'unlabeled', 'test')

    def __init__(
            self,
            root: str,
            split: str = "unlabeled",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        super(LaboratoryDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        # now load the picked numpy arrays
        self.labels: Optional[np.ndarray]
        self.textures: Optional[np.ndarray]
        if split == 'unlabeled':
            # self.data, self.labels = self.__loadfile(self.label_list)
            self.data, self.labels, self.textures = self.__loadaugmentedfile()
            # print(self.data.dtype)

        class_file = os.path.join(
            self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        target: Optional[int]
        if self.textures is not None:
            img, target, texture = self.data[index], self.labels[index], self.textures[index]
        elif self.labels is not None:
            img, target, texture = self.data[index], self.labels[index], None
        else:
            img, target, texture = self.data[index], None, None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        # print("1 - {}".format(len(img.getbands())))
        if self.transform is not None:
            img = self.transform(img)
        # print("2 - {}".format(img[0].shape[0]))
        # print("3 - {}".format(img[1].shape[0]))
        if self.target_transform is not None:
            img = self.target_transform(target)

        return img, {'label': target, 'texture': texture}

    def __loadfile(self, data_files: list) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        root_path = os.path.join(self.root, self.base_folder)
        filenames = [(os.listdir(os.path.join(root_path, fold)), fold) for fold in data_files]
        dfs = [(pd.read_csv(os.path.join(root_path, fold, 'class_labels.csv'), index_col=False), fold) for fold in
               data_files]
        labels = []
        ccnt = int(sum([len(f) - 1 for f, _ in filenames]) / 2)
        # ccnt = 623
        training_data = np.empty((ccnt, 3, 300, 290), dtype=np.uint8)
        idx = 0
        for files, fold in filenames:
            for fsname in files:
                fs = os.path.join(root_path, fold, fsname)
                if fs[-10:] == 'depth.jpeg':
                    continue
                if fs is not None:
                    img = cv2.imread(fs)
                    if img is None:
                        continue
                    # img = cv2.resize(img, (300, 290))

                    img = np.expand_dims(img, axis=0)
                    img = np.transpose(img, (0, 3, 1, 2))
                    training_data[idx] = img

                    df = [df for df, df_fold in dfs if df_fold == fold][0]
                    row = df[df['name'] == fsname[:-5]]
                    if row.any(axis=None):
                        label = row['class'].values[0]
                    else:
                        logging.ERROR("Image {} does not appear on the class_labels.csv labels file".format(fsname))
                    labels.append(label)
                idx += 1


        le = preprocessing.LabelEncoder()
        le.fit(labels)
        le.classes_ = np.array(self.label_list)
        labels = le.transform(labels)

        fp = os.path.join(root_path, 'class_names.txt')
        with open(fp, 'w') as f:
            cw = csv.writer(f, delimiter='\n')
            cw.writerow(list(le.classes_))
        return training_data, labels

    def __loadaugmentedfile(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        root_path = os.path.join(self.root, self.base_folder, "SingleSide")
        filenames = [os.path.join(root_path, x) for x in os.listdir(root_path)]
        filenames = sorted(filenames, key=lambda x: int(os.path.basename(x).split('.')[0]))

        ccnt = len(filenames)

        height, width, channels = np.load(filenames[0]).shape
        training_data = np.empty((ccnt, 4, height, width), dtype=np.uint8)

        label_path = os.path.join(self.root, self.base_folder, 'Singleside_classnames.txt')
        with open(label_path) as lp:
            meta = json.load(lp)

        labels = np.empty(ccnt, dtype=np.object)
        textures = np.empty(ccnt, dtype=np.object)

        for idx, fsname in enumerate(filenames):
            if fsname is not None:

                img = np.load(fsname)
                if img is None:
                    continue

                im_h, im_w, _ = img.shape
                if im_h != height or im_w != width:
                    img = cv2.resize(img, (width, height))

                label = [item for item in meta if (item['img_id'] == str(idx) + '-rgb')][0]

                labels[idx] = deepcopy(label['category'])
                textures[idx] = deepcopy(label['texture'])

                # fig = plt.figure(figsize=(10, 10))
                # plt.axis('off')
                # plt.title(label['category'])
                # plt.imshow(img[..., 0:3])
                # plt.show()
                # Convert from 4d --> 3d
                # img = img[..., 0:3]

                if GRAYSCALE:
                    img = img[..., 0:3]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img = np.expand_dims(img, axis=0)
                img = np.transpose(img, (0, 3, 1, 2))
                training_data[idx] = img

        le = preprocessing.LabelEncoder()
        le.fit(labels)
        le.classes_ = np.array(self.label_list)
        labels = le.transform(labels)

        le = preprocessing.LabelEncoder()
        le.fit(textures)
        le._classes = np.array(self.texture_list)
        textures = le.transform(textures)

        return training_data, labels, textures
