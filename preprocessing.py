import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator

from data_aug.load_laboratory_dataset import LaboratoryDataset
output_classes = ["Flat", "Crumbled", "Folded", "Half-folded"]



sty = 'seaborn'

def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([color_jitter], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        GaussianBlur(kernel_size=int(0.1 * size)),
                                        transforms.ToTensor()])
    return data_transforms


def plot_transformed_dataset(root_folder, distr_path):
    train_dataset = LaboratoryDataset(root_folder, split='unlabeled', transform=(
        ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(32),
                                         n_views=2)))

    # Make labels readable: 0 --> Cumble, 1 --> Flat
    strlabels = [str(int(t)) for t in list(train_dataset.labels)]
    dic = {'0': 'Crumble', '1': 'Flat'}
    strlabels = [dic.get(n, n) for n in strlabels]

    # make an indexed DataFrame
    indexing = np.arange(train_dataset.labels.shape[0])
    input = np.vstack((indexing, strlabels)).T
    df = pd.DataFrame({'id': input[..., 0], 'label': input[..., 1]})

    # Plot a DataFrame head
    fig = plt.figure(figsize=(16, 16))
    fig.tight_layout()
    columns = 3
    rows = 3
    imgs = df.sample(10, random_state=1)

    for i in range(1, columns * rows + 1):
        img, label = [imgs.iloc[[i]]['id'].values[0], imgs.iloc[[i]]['label'].values[0]]
        img = train_dataset.data[int(img)]
        channel, height, width = img.shape
        img = np.reshape(img, (height, width, channel))
        ax = fig.add_subplot(rows, columns, i)
        ax.title.set_text(label)
        plt.imshow(img.astype(np.int32))
        plt.axis('off')
    plt.suptitle("Transformed Laboratory Dataset", fontsize=16, fontweight="bold")
    fp = os.path.join(distr_path, 'transformed_lab_dataset.png')
    plt.savefig(fp)
    plt.close(fig)

    # Plot a DataFrame head
    fig2 = plt.figure(figsize=(16, 16))
    fig2.tight_layout()
    columns = 2
    rows = 4
    transform = ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(64), n_views=2)

    for i in range(1, columns * rows + 1, 2):
        img, label = [imgs.iloc[[i]]['id'].values[0], imgs.iloc[[i]]['label'].values[0]]
        img = train_dataset.data[int(img)]
        img_vis = Image.fromarray((np.transpose(img, (1, 2, 0)).astype(np.uint8)))
        img_vis = transform(img_vis)
        channel, height, width = img_vis[0].shape
        img_vis[0] = torch.reshape(img_vis[0], (height, width, 3))
        img_vis[1] = torch.reshape(img_vis[1], (height, width, 3))
        ax1 = fig2.add_subplot(rows, columns, i)
        ax1.title.set_text("First View - " + label)
        plt.imshow(img_vis[0].numpy())
        plt.axis('off')

        ax2 = fig2.add_subplot(rows, columns, i+1)
        ax2.title.set_text("Second View - " + label)
        plt.imshow(img_vis[1].numpy())
        plt.axis('off')
    plt.suptitle("Augmented Views", fontsize=16, fontweight="bold")
    fp = os.path.join(distr_path, 'augmented_dataset.png')
    plt.savefig(fp)
    plt.close(fig2)


def plot_original_dataset(root_folder, distr_path):
    train_dataset = LaboratoryDataset(root_folder, split='unlabeled', transform=None)

    # Make labels readable: 0 --> Cumble, 1 --> Flat
    strlabels = [str(int(t)) for t in list(train_dataset.labels)]
    dic = {'0': 'Crumble', '1': 'Flat'}
    strlabels = [dic.get(n, n) for n in strlabels]


    # make an indexed DataFrame
    indexing = np.arange(train_dataset.labels.shape[0])
    input = np.vstack((indexing, strlabels)).T
    df = pd.DataFrame({'id': input[..., 0], 'label': input[..., 1]})

    # Plot a DataFrame head
    fig = plt.figure(figsize=(16, 16))
    fig.tight_layout()
    columns = 3
    rows = 3
    imgs = df.sample(10, random_state=1)
    for i in range(1, columns * rows + 1):
        img, label = [imgs.iloc[[i]]['id'].values[0], imgs.iloc[[i]]['label'].values[0]]
        img = train_dataset.data[int(img)]
        channel, height, width = img.shape
        img = np.reshape(img, (height, width, channel))
        ax = fig.add_subplot(rows, columns, i)
        ax.title.set_text(label)
        plt.imshow(img.astype(np.int32))
        plt.axis('off')
    plt.suptitle("Original Laboratory Dataset", fontsize=16, fontweight="bold")
    fp = os.path.join(distr_path, 'original_lab_dataset.png')
    plt.savefig(fp)
    plt.close(fig)


if __name__ == "__main__":
    # mpl.style.use(sty)
    root_folder = './datasets'
    distr_path = './images'
    if not os.path.exists(distr_path):
        os.makedirs(distr_path)

    # plot_transformed_dataset(root_folder, distr_path)
    # plot_original_dataset(root_folder, distr_path)

    # Start the plotting
    batch_size = 16
    train_dataset = LaboratoryDataset(root_folder, split='unlabeled', transform=None)
    # Make labels readable: 0 --> Cumble, 1 --> Flat
    strlabels = [output_classes[i] for i in list(train_dataset.labels)]

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.75,0.75]) # axis starts at 0.1, 0.1

    y = [strlabels.count(i) for i in output_classes]

    ax.bar(train_dataset.label_list, y, width=0.3, color=['aquamarine', 'mediumseagreen'])
    # Annotating the bar plot with the values (total death count)

    for i in ax.patches:
        ax.annotate(format(i.get_height(), 'd'),
                       (i.get_x() + i.get_width() / 2,
                        i.get_height()), ha='center', va='center',
                       size=12, xytext=(0, 8),
                       textcoords='offset points')

    ax.set_title("Image distribution", fontsize=16, fontweight="bold")

    ax.axes.yaxis.set_ticks([])
    ax.set_xlabel("States", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    fp = os.path.join(distr_path, 'label_distribution.png')
    plt.savefig(fp)


