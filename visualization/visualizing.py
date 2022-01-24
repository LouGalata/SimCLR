import argparse
import pathlib
import torch
from torchvision import models
from models.resnet_simclr import ResNetSimCLR
from data_aug.load_laboratory_dataset import LaboratoryDataset
import torch.utils.data as data
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from torchvision import transforms
from matplotlib import cm
from models.deep_features import DeepFeatures
from tqdm import tqdm


tsne = TSNE()
device = 'cpu'
VIS_HEIGHT = 250
VIS_WIDTH = 250
output_classes = ["Flat", "Crumbled", "Folded", "Half-folded"]
texture_classes = ["VerticalStripes", "RectangleTowel", "TvCloth", "HorizontalStripes", "Plaid", "SquareTowel", "BeigeTowel"]


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR Visualization')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-checkpoint', default='Dec23_12-07-15_msi-iri', help='checkpoint model')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--out_dim', default=32, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')


def plot_vecs_n_labels(v, labels, fname):
    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    sns.set_style("darkgrid")
    sns.scatterplot(v[:, 0], v[:, 1], hue=labels, legend='full')
    plt.legend(output_classes)
    plt.savefig(fname)
    plt.close()


def display_pca_scatterplot_2D(model, test_dataset):
    print("Forward Action!")

    te = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    y = torch.autograd.Variable(torch.FloatTensor())
    y_label = torch.autograd.Variable(torch.FloatTensor())

    for k in range(5):
        x, labels = iter(te).next()
        labels = labels['label']
        x = x.to(device)
        y = torch.cat([y, model.forward(x)], dim=0)
        y_label = torch.cat([y_label, torch.flatten(labels)], dim=0)

    embedding_space_instances = y.cpu().detach().numpy()
    embedding_space_labels = y_label.cpu().detach().numpy()
    tsne_proj = tsne.fit_transform(embedding_space_instances)
    ######
    cmap = cm.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = 4
    for lab, j in zip(range(num_categories), output_classes):
        indices = embedding_space_labels == lab
        ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap(lab)).reshape(1, 4), label=j,
                   alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.title("Linear Regression - Vertical Camera")
    plt.savefig('../images/tsne2d_LR_vertical_oneside_4d.png')

    # fig, ax = plt.subplots(figsize=(8, 8))
    # tsne3d = TSNE(3, verbose=1)
    # tsne_proj = tsne3d.fit_transform(embedding_space_instances)
    # cmap = cm.get_cmap('tab20')
    # num_categories = 4
    # for lab, j in zip(range(num_categories), output_classes):
    #     indices = embedding_space_labels == lab
    #     ax.scatter(tsne_proj[indices, 0],
    #                tsne_proj[indices, 1],
    #                tsne_proj[indices, 2],
    #                c=np.array(cmap(lab)).reshape(1, 4),
    #                label=j,
    #                alpha=0.5)
    # ax.legend(fontsize='large', markerscale=2)
    # plt.title("Linear Regression - Vertical Camera")
    # plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    checkpoint_path = os.path.join('..', 'runs', args.checkpoint)
    latest_file = max(os.listdir(checkpoint_path), key=lambda x: pathlib.Path(x).suffix == '.tar')

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.Resize((VIS_HEIGHT, VIS_WIDTH)),
        transforms.ToTensor()
    ])
    test_dataset = LaboratoryDataset('../datasets/', split='unlabeled', transform=transform)

    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    batch_imgs, batch_labels = next(iter(data_loader))

    batch_size, num_channels, _, _ = batch_imgs.shape

    model_state = checkpoint_path + '/' + latest_file

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, num_input_channel=num_channels)
    x = torch.load(model_state, map_location='cpu')
    model.load_state_dict(x['state_dict'])
    model.eval()

    display_pca_scatterplot_2D(model, test_dataset)
    root = "../Outputs"
    files = os.listdir(root)
    last_run = 0
    for fp in files:
        if '_' in fp:
            run = int(fp.split('_')[-1])
            if run >= last_run:
                last_run = run + 1
    run = "run_" + str(last_run).zfill(3)

    IMGS_FOLDER = '../Outputs/' + run + '/VerticalImages/Images'
    EMBS_FOLDER = '../Outputs/' + run + '/VerticalImages/Embeddings'
    TB_FOLDER = '../Outputs/' + run + '/Tensorboard'
    EXPERIMENT_NAME = 'VERTICAL-IMAGES'
    DEVICE = 'cpu'

    if not os.path.exists(IMGS_FOLDER):
        os.makedirs(IMGS_FOLDER)
    if not os.path.exists(EMBS_FOLDER):
        os.makedirs(EMBS_FOLDER)
    if not os.path.exists(TB_FOLDER):
        os.makedirs(TB_FOLDER)

    DF = DeepFeatures(model=model,
                      run=run,
                      imgs_folder=IMGS_FOLDER,
                      embs_folder=EMBS_FOLDER,
                      tensorboard_folder=TB_FOLDER,
                      experiment_name=EXPERIMENT_NAME)

    embs = model(batch_imgs)
    image_outsizes = [VIS_HEIGHT, VIS_WIDTH]
    print("Input images: " + str(batch_imgs.shape))
    print("Embeddings: " + str(embs.shape))

# For only one BATCH
    # Save Labels separately on a line-by-line manner.
    # with open(os.path.join('../Outputs', 'metadata.tsv'), "w") as f:
    #     batch_labels = list(map(lambda i: "Crumble" if (i == 0) else "Flat", batch_labels.numpy()))
    #     for batch_label in batch_labels:
    #         f.write("{}\n".format(batch_label))
    # DF.write_batch_embedding(x=batch_imgs.to(DEVICE))

# For 7 Batches
    DF.write_data_loader_embeddings(args, data_loader, output_classes, texture_classes, image_outsizes)
    DF.create_tensorboard_log()