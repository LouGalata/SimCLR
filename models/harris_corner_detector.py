import numpy as np
from cv2 import cv2
import argparse
import os
import random
from data_aug.load_laboratory_dataset import LaboratoryDataset
from torchvision.transforms import transforms


parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='laboratory',
                    help='dataset name', choices=['stl10', 'cifar10', 'simulation_towels', 'laboratory'])
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')

def transformations():
    return transforms.Compose([transforms.ToTensor()])


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    is_rgb = True
    data = LaboratoryDataset(os.path.join(os.pardir, args.data), split='unlabeled', transform=transformations())
    size = len(data)
    for i in range(3):
        n = random.randint(0, size-1)
        img = data[n][0][0:3]
        img = np.transpose(img.numpy(), (1, 2, 0))
        cv2.imshow('Original image', img)
        cv2.waitKey(1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_flt = np.float32(gray)

        dst = cv2.cornerHarris(image_flt, 2, 3, 0.04)

        dst = cv2.dilate(dst, None)

        img[dst > 0.01 * dst.max()] = [0, 0, 255]

        frame = cv2.imshow('Detected corners', img)
        filepath = os.path.join(os.pardir, "images", f"HarrisCorner-{i}.jpg")
        cv2.imwrite(filepath, 255*img)
