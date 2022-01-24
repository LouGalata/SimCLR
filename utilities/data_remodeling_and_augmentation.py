import os
from shutil import copyfile
from cv2 import cv2
import random
import numpy as np
import json
from copy import deepcopy


def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img


def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img


def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img


def filters(img, f_type="blur"):
    '''
    ### Filtering ###
    img: image
    f_type: {blur: blur, gaussian: gaussian, median: median}
    '''
    if f_type == "blur":
        image = img.copy()
        fsize = 9
        return cv2.blur(image, (fsize, fsize))

    elif f_type == "gaussian":
        image = img.copy()
        fsize = 9
        return cv2.GaussianBlur(image, (fsize, fsize), 0)

    elif f_type == "median":
        image = img.copy()
        fsize = 9
        return cv2.medianBlur(image, fsize)


def noisy(img, noise_type="gauss"):
    '''
    ### Adding Noise ###
    img: image
    cj_type: {gauss: gaussian, sp: salt & pepper}
    '''
    if noise_type == "gauss":
        image = img.copy()
        mean = 0
        st = 0.7
        gauss = np.random.normal(mean, st, image.shape)
        gauss = gauss.astype('uint8')
        image = cv2.add(image, gauss)
        return image

    elif noise_type == "sp":
        image = img.copy()
        prob = 0.05
        if len(image.shape) == 2:
            black = 0
            white = 255
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        return image



def do_transforms(four_d_image, path, idx, label_path, label):

    transforms = ['brightness', 'noisy', 'vertical_flip', 'rotation', 'filters']
    labels = [{} for _ in range(5)]

    for cnt in range(5):
        rgb_image = four_d_image[..., 0:3]
        depth_image = four_d_image[..., -1]
        count = CCNT + 5*idx + cnt % 5

        d = random.choice(transforms)
        if d == "channel_shift":
            temperatures = np.arange(40, 60)
            temperature = random.choice(temperatures)
            rgb_image = channel_shift(rgb_image, temperature)
            depth_image = np.expand_dims(depth_image, axis=-1)
            img = np.concatenate((rgb_image, depth_image), axis=-1)
        elif d == "brightness":
            low = random.choice([0.5, 0.6, 0.7])
            high = random.choice([2.8, 3.0, 3.2])
            rgb_image = brightness(rgb_image, low, high)
            depth_image = np.expand_dims(depth_image, axis=-1)
            img = np.concatenate((rgb_image, depth_image), axis=-1)
        elif d == "noisy":
            rgb_image = noisy(rgb_image)
            depth_image = np.expand_dims(depth_image, axis=-1)
            img = np.concatenate((rgb_image, depth_image), axis=-1)

            # img = horizontal_flip(img, True)
        elif d == "vertical_flip":
            rgb_image = vertical_flip(rgb_image, True)
            depth_image = vertical_flip(depth_image, True)
            depth_image = np.expand_dims(depth_image, axis=-1)
            img = np.concatenate((rgb_image, depth_image), axis=-1)

        elif d == "rotation":
            angle = random.choice(np.arange(10, 75))
            rgb_image = rotation(rgb_image, angle)
            depth_image = rotation(depth_image, angle)
            depth_image = np.expand_dims(depth_image, axis=-1)
            img = np.concatenate((rgb_image, depth_image), axis=-1)

        # filters
        else:
            filter_id = random.choice(['blur', 'gaussian', 'median'])
            rgb_image = filters(rgb_image, filter_id)
            depth_image = np.expand_dims(depth_image, axis=-1)
            img = np.concatenate((rgb_image, depth_image), axis=-1)

        filename = os.path.join(augmdata, str(count) + '.npy')
        np.save(filename, img)
        label['img_id'] = str(count) + '-rgb'
        labels[cnt] = label.copy()
    return labels

if __name__ == "__main__":
    ROOT = '/home/lougalata/Desktop'
    four_d_data = "SingleSideAugm/SingleSide_4D"
    AUGM = "SingleSideAUGMDATA"
    paths = [os.path.join(ROOT, four_d_data, x) for x in os.listdir(os.path.join(ROOT, four_d_data))]
    paths = sorted(paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
    CCNT = len(paths)
    augmdata = os.path.join(ROOT, AUGM)
    label_path = os.path.join(os.getcwd(), 'Singleside_classnames_augm.txt')
    if not os.path.exists(augmdata):
        os.makedirs(augmdata)

    with open(label_path) as lp:
        meta = json.load(lp)
        meta = meta['img_id']
        meta = sorted(meta, key=lambda x: int(x['img_id'].split('-')[0]))
    augm_meta = deepcopy(meta)

    for idx, d in enumerate(paths):
        basename = os.path.basename(d)
        four_d_image = np.load(d)
        temp = os.path.join(augmdata, basename)
        copyfile(d, temp)

        try:
            label = [item for item in meta if (item['img_id'] == str(idx) + '-rgb')][0]
        except Exception:
            pass
        labels = do_transforms(four_d_image, d, idx, label_path, label)
        [augm_meta.append(k) for k in labels]


    with open(label_path, 'w') as lp:
        json.dump(augm_meta, lp)
