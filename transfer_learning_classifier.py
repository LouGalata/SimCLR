import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from data_aug.load_laboratory_dataset import LaboratoryDataset
from torch.utils.tensorboard import SummaryWriter
import time
import os
import copy
import pandas as pd
import logging
from utils import save_config_file, save_checkpoint
import argparse
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from itertools import product


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ["Flat", "Crumbled", "Folded", "Half-folded"]

parser = argparse.ArgumentParser(description="This is the command line interface for the linear evaluation model")

parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('--out_dim', default=32, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-num-workers', default=2, type=int, help="")
parser.add_argument('-epochs', default=50)
parser.add_argument('-logistic-batch-size', default=16, type=int, help='mini batch size for the Logistic Regression model')
parser.add_argument('-logistic-epochs', default=100, type=int, help="Epochs for the logistic Regression model")
parser.add_argument('--disable-cuda', action='store_false',
                    help='Disable CUDA')
parser.add_argument('--rgb-input', action='store_false', help='Use RGB or Grayscale images')


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    _, _, channels = inp.shape
    if channels == 4:
        inp = inp[..., 0:3]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels['label'].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                # ax = plt.subplot(num_images//2, 2, images_so_far)
                # ax.axis('off')
                # ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                # imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def train_model(args, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    logging.info(f"Start SimCLR training for {num_epochs} epochs.")
    logging.info(f"Training with gpu: {args.disable_cuda}.")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            logging.debug(f"Epoch: {epoch}\t {phase} Loss: {epoch_loss}\t {phase} accuracy: {epoch_acc}")


            writer.add_scalar('loss', epoch_loss, global_step=epoch)
            writer.add_scalar('acc', epoch_acc, global_step=epoch)

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, epoch_loss


def flatten(t):
    return [item for sublist in t for item in sublist]


if __name__ == "__main__":
    args = parser.parse_args()

    # TRAIN THE ONE THE LAST LAYER
    writer = SummaryWriter()
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)
    save_config_file(writer.log_dir, args)

    parameters = dict(
        lr=[0.001],
        batch_size=[8],
        rgb=[True, False]
    )
    param_values = [v for v in parameters.values()]

    for lr, bs, rgb in product(*param_values):
        if rgb:
            transform = transforms.Compose([
                # you can add other transformations in this list
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                # you can add other transformations in this list
                transforms.ToTensor()
            ])
        data_path = os.path.join(os.pardir, 'datasets')
        full_dataset = LaboratoryDataset(data_path, split='unlabeled', transform=transform)

        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        dataset_sizes = {'train': train_size, 'test': test_size}
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers
        )

        dataloaders['test'] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers
        )

        batch_imgs, batch_labels = next(iter(dataloaders['train']))

        batch_size, num_channels, _, _ = batch_imgs.shape

        model_conv = torchvision.models.resnet18(pretrained=True)
        for param in model_conv.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, len(class_names))

        model_conv = model_conv.to(device)

        criterion = nn.CrossEntropyLoss().to(device)

        if num_channels != 3:
            model_conv.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
            model_conv.conv1.requires_grad_(True)

        # Observe that only parameters of final layer are being optimized as
        # opposed to before.
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
        model_conv, best_acc, last_epoch_total_loss = train_model(args, model_conv, criterion, optimizer_conv,
                                 exp_lr_scheduler, num_epochs=args.epochs)

        logging.info("Training has finished.")
        # save model checkpoints
        file_name = "epochs=%d-bs=%d-lr=%g-rgb=%s" % (args.epochs, bs, lr, str(rgb))
        checkpoint_name = 'checkpoint_' + file_name + '.pth.tar'
        save_checkpoint({
            'epoch': args.epochs,
            'lr': lr,
            'batch_size': bs,
            'rgb': rgb,
            'arch': 'resnet18',
            'state_dict': model_conv.state_dict(),
            'optimizer': optimizer_conv.state_dict(),
        }, is_best=False, filename=os.path.join(writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {writer.log_dir}.")

        writer.add_hparams(
            {"lr": lr, "bsize": bs, "rgb": rgb},
            {
                "acc-top1": best_acc,
                "loss": last_epoch_total_loss,
            },
        )
        # visualize_model(model_conv)

        loss_epoch = 0
        accuracy_epoch = 0
        model_conv.eval()
        y_true = []
        y_pred = []
        for step, (x, y) in enumerate(dataloaders['test']):
            model_conv.zero_grad()

            x = x.to(device)
            y = y['label'].to(device)

            output = model_conv(x)
            loss = criterion(output, y)

            predicted = output.argmax(1)

            acc = (predicted == y).sum().item() / y.size(0)
            accuracy_epoch += acc
            y_pred.append(predicted.cpu().detach().numpy())
            y_true.append(y.cpu().detach().numpy())

            loss_epoch += loss.item()


        y_pred = flatten(y_pred)
        y_true = flatten(y_true)
        y_pred = list(map(lambda i: class_names[i], y_pred))
        y_true = list(map(lambda i: class_names[i], y_true))
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        cr = classification_report(y_true, y_pred, output_dict=True, labels=class_names)

        pd.DataFrame(cr).to_csv('report-' + file_name + '.csv', sep=',')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.savefig('cm-' + file_name + '.png')

        # TRAIN THE WHOLE MODEL
        # model_ft = models.resnet18(pretrained=True)
        # num_ftrs = model_ft.fc.in_features
        # # Here the size of each output sample is set to 2.
        # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        # model_ft.fc = nn.Linear(num_ftrs, len(class_names))
        #
        # model_ft = model_ft.to(device)
        #
        # criterion = nn.CrossEntropyLoss()
        #
        # # Observe that all parameters are being optimized
        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        #
        # # Decay LR by a factor of 0.1 every 7 epochs
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        #
        # model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
        #                        num_epochs=25)
        #
        # visualize_model(model_ft)