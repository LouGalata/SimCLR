from torch.utils.data import Dataset
import os
import os.path
import numpy as np
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import VisionDataset
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg
from . import binary_transformation


class SimulationClothDataset(VisionDataset):
    base_folder = 'small_dataset'
    url = None  # maybe the .tar.gz
    filename = 'small_dataset.zip'  # "stl10_binary.tar.gz"
    # tgz_md5 = '91f7769df0f17e558f3565bffb0c7dfb'  NO idea what is it

    class_names_file = 'class_names.txt'
    folds_list_file = 'fold_indices.txt'
    label_list = [
        ['train_X.bin', '918c2871b30a85fa023e0c44e0bee87f'],
        ['train_y.bin', '5a34089d4802c674881badbb80307741'],
        ['unlabeled_X.bin', '5242ba1fed5e4be9e1e742405eb56ca4']
    ]

    test_list = [
        ['test_X.bin', '7f263ba9f9e0b06b93213547f721ac82'],
        ['test_y.bin', '36f9794fa4beb8a2c72628de14fa638e']
    ]
    splits = ('train', 'train+unlabeled', 'unlabeled', 'test')

    def __init__(
            self,
            root: str,
            split: str = "unlabeled",
            folds: Optional[int] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(SimulationClothDataset, self).__init__(root, transform=transform,
                                                     target_transform=target_transform)
        self.split = verify_str_arg(split, "split", self.splits)
        # self.folds = self._verify_folds(folds)

        if download:
            self.download()

        # now load the picked numpy arrays
        self.labels: Optional[np.ndarray]
        if self.split == 'train':
            self.data, self.labels = self.__loadfile(
                self.label_list[0][0], self.label_list[1][0])
            self.__load_folds(folds)

        elif self.split == 'train+unlabeled':
            self.data, self.labels = self.__loadfile(
                self.label_list[0][0], self.label_list[1][0])
            self.__load_folds(folds)
            unlabeled_data, _ = self.__loadfile(self.label_list[2][0])
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate(
                (self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == 'unlabeled':
            self.data, _ = self.__loadfile(self.label_list[2][0])
            self.labels = np.asarray([-1] * self.data.shape[0])
        else:  # self.split == 'test':
            self.data, self.labels = self.__loadfile(
                self.test_list[0][0], self.test_list[1][0])

        class_file = os.path.join(
            self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()


    def _verify_folds(self, folds: Optional[int]) -> Optional[int]:
        if folds is None:
            return folds
        elif isinstance(folds, int):
            if folds in range(10):
                return folds
            msg = ("Value for argument folds should be in the range [0, 10), "
                   "but got {}.")
            raise ValueError(msg.format(folds))
        else:
            msg = "Expected type None or int for argument folds, but got type {}."
            raise ValueError(msg.format(type(folds)))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        target: Optional[int]
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __loadfile(self, data_file: str, labels_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        images, labels = binary_transformation.custom_training_dataset_retrieval(self.root, self.base_folder)

        width, height, channels, _ = images.shape
        images = np.reshape(images, (-1, 3, width, height))
        return images, labels

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.label_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
        self._check_integrity()

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def __load_folds(self, folds: Optional[int]) -> None:
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(
            self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds, 'r') as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.int64, sep=' ')
            self.data = self.data[list_idx, :, :, :]
            if self.labels is not None:
                self.labels = self.labels[list_idx]
