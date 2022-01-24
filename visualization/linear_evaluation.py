import argparse
import torch
import os
import matplotlib.pyplot as plt
import pathlib
from torchvision import models
import numpy as np
from torchvision import transforms
from data_aug.load_laboratory_dataset import LaboratoryDataset
from linear_classifier import LogisticRegression
from models.resnet_simclr import ResNetSimCLR
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


device = 'cpu'
output_classes = ["Flat", "Crumbled", "Folded", "Half-folded"]

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# making a command line interface
parser = argparse.ArgumentParser(description="This is the command line interface for the linear evaluation model")

parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-checkpoint', default='Dec23_12-45-30_msi-iri', help='checkpoint model')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--out_dim', default=64, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-num-workers', default=2, type=int, help="")
parser.add_argument('-device', default='cpu', help="CPU or GPU device")
parser.add_argument('-logistic-batch-size', default=32, type=int, help='mini batch size for the Logistic Regression model')
parser.add_argument('-logistic-epochs', default=100, type=int, help="Epochs for the logistic Regression model")


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h = simclr_model(x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y['label'].numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def train(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )
    return loss_epoch, accuracy_epoch


def flatten(t):
    return [item for sublist in t for item in sublist]


def test(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    y_true = []
    y_pred = []
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)

        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc
        y_pred.append(predicted.numpy())
        y_true.append(y.numpy())

        loss_epoch += loss.item()
    y_pred = flatten(y_pred)
    y_true = flatten(y_true)
    y_pred = list(map(lambda i: output_classes[i], y_pred))
    y_true = list(map(lambda i: output_classes[i], y_true))

    cm = confusion_matrix(y_true, y_pred, labels=output_classes)
    print(classification_report(y_true, y_pred, labels=output_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,  display_labels=output_classes)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig('confusion-matrix-vertical-camera.png')
    return loss_epoch, accuracy_epoch


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint_path = os.path.join(os.pardir, 'runs', args.checkpoint)
    latest_file = max(os.listdir(checkpoint_path), key=lambda x: pathlib.Path(x).suffix == '.tar')

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    data_path = os.path.join(os.pardir, 'datasets')
    full_dataset = LaboratoryDataset(data_path, split='unlabeled', transform=transform)

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    dataloaders['test'] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    batch_imgs, batch_labels = next(iter(dataloaders['train']))

    batch_size, num_channels, _, _ = batch_imgs.shape


    model_state = checkpoint_path + '/' + latest_file

    simclr_model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, num_input_channel=num_channels)
    x = torch.load(model_state, map_location='cpu')
    simclr_model.load_state_dict(x['state_dict'])
    simclr_model.eval()



    ## Logistic Regression
    n_classes = 4
    # simclr_model features of embedding space. TODO:change the explicit assignment
    model = LogisticRegression(args.out_dim, n_classes)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        simclr_model, dataloaders['train'], dataloaders['test'], args.device
    )

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logistic_batch_size
    )

    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(
            args, arr_train_loader, simclr_model, model, criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
        )

        # final testing
    loss_epoch, accuracy_epoch = test(
        args, arr_test_loader, simclr_model, model, criterion, optimizer
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
    )
