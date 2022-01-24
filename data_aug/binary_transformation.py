import argparse
import os
import pandas as pd
from zipfile import ZipFile
import datetime
import base64
import cv2
import numpy as np
import glob
from PIL import Image


CHOICES = ['one_hand_diagonal', 'one_hand_lifting', 'one_hand_sideways']
# specifying the zip file name


def extract_files(args):
    fp = os.path.join(args.data, args.file_name)
    if not os.path.isdir(fp):
        # opening the zip file in READ mode
        fp += '.zip'
        with ZipFile(fp, 'r') as zip:
            for info in zip.infolist()[:10]:
                print(info.filename)
                print('\tModified:\t' + str(datetime.datetime(*info.date_time)))
                print('\tSystem:\t\t' + str(info.create_system) + '(0 = Windows, 3 = Unix)')
                print('\tZIP version:\t' + str(info.create_version))
                print('\tCompressed:\t' + str(info.compress_size) + ' bytes')
                print('\tUncompressed:\t' + str(info.file_size) + ' bytes')

            print('Extracting all the files now...')
            zip.extractall()
            print('Done!')
            return
    print('Files are already extracted!')


def get_filenames(args):
    fp = os.path.join(args.data, args.file_name)
    files_to_training_set = []
    fname = list(map(lambda x: os.path.join(fp, x), CHOICES))
    for i in fname:
        file_categories = os.listdir(i)
        for j in file_categories:
            path = os.path.join(i, j)
            path += '/data.csv'
            df = pd.read_csv(path)
            df_flat = df[(df.cloth_state.values == 'flat') & (df.action.values == 'approaching')]
            df_crumble = df[(df.cloth_state.values == 'diagonallyFolded') & (df.action.values == 'waiting')]
            images = df_flat.filename.values.tolist() + df_crumble.filename.values.tolist()
            for im in images:
                image_path = os.path.join(i, j)
                image_path += '/RGB/' + im + '.png'
                files_to_training_set.append(image_path)

    print('Files paths are ready!')
    return files_to_training_set


def concatanate_images(filenames):
    for file in filenames:
        if cv2.imread(file) is not None:

            images = [cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (216, 216)) for file in filenames if cv2.imread(file) is not None]
            folder = [(os.path.abspath(os.path.join(file, os.pardir, os.pardir)), os.path.basename(file)) for file in filenames
                      if cv2.imread(file) is not None]
            labels = []
            for fold, idx in folder:
                path = os.path.join(fold, 'data.csv')
                data = pd.read_csv(path)
                label = data[data['filename'] == idx[:-4]]['cloth_state'].values[0]
                labels.append(label)

            conc_img = np.stack(images, axis=-1)
    return conc_img, labels



def binary_transform(args, images):
    binary_file = args.file_name + '_binary'
    fp = os.path.join(args.data, binary_file)
    if not os.path.isdir(fp):
        os.makedirs(fp)

    for im in images:
        if os.path.isfile(im):
            with open(im, "rb") as f:
                png_encoded = base64.b16encode(f.read())

            encoded_b2 = "".join([format(n, '08b') for n in png_encoded])
            im = os.path.basename(im)
            filename, _ = os.path.splitext(im)
            filename += '.bin'
            binary = os.path.join(fp, filename)
            if not os.path.isfile(binary):
                with open(binary, 'wb') as f:
                    try:
                        f.write(encoded_b2)
                    except:
                        print("File cannot be saved!")
                        continue


def custom_training_dataset_retrieval(*args):
    parser = argparse.ArgumentParser(description='Dataset Transformation in Torchdataset')
    if not args:
        parser.add_argument('-data', metavar='DIR', default='.',
                            help='path to dataset')
        parser.add_argument('--file-name', default='small_dataset', help='Zip file to be transformed')
        parser.add_argument('--size-training-set', default=2000, type=int, help='size of Training set')
    else:
        parser.add_argument('-data', metavar='DIR', default=args[0],
                            help='path to dataset')
        parser.add_argument('--file-name', default=args[1], help='Zip file to be transformed')

    args = parser.parse_args()
    extract_files(args)
    filenames = get_filenames(args)
    images, labels = concatanate_images(filenames)

    return images, labels

    # binary_transform(args, filenames)
