import os
from cv2 import cv2
import matplotlib.pyplot as plt
import json

ROOT_FOLDER = '/home/lougalata/Desktop/SingleSideAugm'
CLASSES = ["Flat", "Crumbled", "Folded", "Half-folded"]

TEXTILES = ['TableclothTwoVerticalStripes', 'SquareWhiteTowel', 'GreenTowel', 'RectangleTowel',
            'TowelHorizontalStripes','TvTablecloth', 'BigPlaid', 'SmallPlaid', 'WCtowel', 'BeigeTablecloth']


def get_info():
    while True:
        try:
            int_info = int(input("INFOS\n0: TableclothTwoVerticalStripes \n1: SquareWhiteTowel \n2: GreenTowel \n3: "
                                 "RectangleTowel \n4: TowelHorizontalStripes \n 5: TvTablecloth \n6: BigPlaid \n7: "
                                 "SmallPlaid \n8: WCtowel \n9: BeigeTablecloth\n"))
            break
        except Exception as e:
            print("please enter number")

    info = None
    if int_info == 0:
        info = TEXTILES[0]
    elif int_info == 1:
        info = TEXTILES[1]
    elif int_info == 2:
        info = TEXTILES[2]
    elif int_info == 3:
        info = TEXTILES[3]
    elif int_info == 4:
        info = TEXTILES[4]
    elif int_info == 5:
        info = TEXTILES[5]
    elif int_info == 6:
        info = TEXTILES[6]
    elif int_info == 7:
        info = TEXTILES[7]
    elif int_info == 8:
        info = TEXTILES[8]
    elif int_info == 9:
        info = TEXTILES[9]

    while info not in TEXTILES:
        print("Input should be one of the following infos: {}".format(TEXTILES))
        info = get_info()
    return info


def get_label():
    while True:
        try:
            int_label = int(input("LABELS \n 0: Flat \n1: Crumbled \n2: Folded \n3: Half-folded \n"))
            break
        except Exception as e:
            print("please enter number")

    label = None
    if int_label == 0:
        label = CLASSES[0]
    elif int_label == 1:
        label = CLASSES[1]
    elif int_label == 2:
        label = CLASSES[2]
    elif int_label == 3:
        label = CLASSES[3]

    while label not in CLASSES:
        print("Input should be one of the following labels: {}".format(CLASSES))
        label = get_label()
    return label


def load_files(fname):
    im = cv2.imread(fname)
    fig = plt.figure()
    plt.imshow(im)
    plt.show()

    label = get_label()
    cloth_info = get_info()
    img_id = os.path.basename(fname).split('.')[0]

    data_info['img_id'].append({
        'img_id': img_id,
        'category': label,
        'texture': cloth_info
    })



if __name__ == "__main__":
    filenames = [os.path.join(ROOT_FOLDER, p) for p in os.listdir(ROOT_FOLDER) if os.path.isfile(os.path.join(ROOT_FOLDER, p)) and 'rgb' in p]

    data_info = {}
    data_info['img_id'] = []
    ccnt = len(filenames)
    filenames = sorted(filenames)
    filenames = sorted(filenames, key=lambda x: int(os.path.basename(x).split('-')[0]))
    for fname in filenames:
        load_files(fname)

    fp = 'Singleside_classnames_augm.txt'
    with open(fp, 'w') as f:
        json.dump(data_info, f)
