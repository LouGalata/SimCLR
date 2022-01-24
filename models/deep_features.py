import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import cv2
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pandas as pd
import torchvision


class DeepFeatures(torch.nn.Module):
    '''
    This class extracts, reads, and writes data embeddings using a pretrained deep neural network. Meant to work with
    Tensorboard's Embedding Viewer (https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin).
    When using with a 3 channel image input and a pretrained model from torchvision.models please use the
    following pre-processing pipeline:

    transforms.Compose([transforms.Resize(imsize),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) ## As per torchvision docs

    Args:
        model (nn.Module): A Pytorch model that returns an (B,1) embedding for a length B batched input
        imgs_folder (str): The folder path where the input data elements should be written to
        embs_folder (str): The folder path where the output embeddings should be written to
        tensorboard_folder (str): The folder path where the resulting Tensorboard log should be written to
        experiment_name (str): The name of the experiment to use as the log name


    '''

    def __init__(self, model,
                 run,
                 imgs_folder,
                 embs_folder,
                 tensorboard_folder,
                 experiment_name=None):

        super(DeepFeatures, self).__init__()

        self.model = model
        self.model.eval()
        self.run = run
        self.imgs_folder = imgs_folder
        self.embs_folder = embs_folder
        self.tensorboard_folder = tensorboard_folder

        self.name = experiment_name

        self.writer = None

    def generate_embeddings(self, x):
        '''
        Generate embeddings for an input batched tensor

        Args:
            x (torch.Tensor) : A batched pytorch tensor

        Returns:
            (torch.Tensor): The output of self.model against x
        '''
        return (self.model(x))

    def write_batch_embedding(self, x, outsize=(28, 28)):
        '''
        Generate embeddings for an input batched tensor and write inputs and
        embeddings to self.imgs_folder and self.embs_folder respectively.

        Inputs and outputs will be stored in .npy format with randomly generated
        matching filenames for retrieval

        Args:
            x (torch.Tensor) : An input batched tensor that can be consumed by self.model
            outsize (tuple(int, int)) : A tuple indicating the size that input data arrays should be
            written out to

        Returns:
            (bool) : True if writing was succesful

        '''

        assert len(os.listdir(self.imgs_folder)) == 0, "Images folder must be empty"
        assert len(os.listdir(self.embs_folder)) == 0, "Embeddings folder must be empty"

        # Generate embeddings
        embs = self.generate_embeddings(x)

        # Detach from graph
        embs = embs.detach().cpu().numpy()

        # Start writing to output folders
        for i in range(len(embs)):
            key = str(np.random.random())[-7:]
            # tensor2np(x[i], outsize)) :  (28, 28, 3)
            # x[i] : Tensor(3, 300, 290)
            np.save(self.imgs_folder + r"/" + key + '.npy', tensor2np(x[i], outsize))
            np.save(self.embs_folder + r"/" + key + '.npy', embs[i])
        return True

    def write_data_loader_embeddings(self, args, x, output_classes, texture_classes, outsize=(28, 28)):
        '''
        Generate embeddings for an input batched tensor and write inputs and
        embeddings to self.imgs_folder and self.embs_folder respectively.

        Inputs and outputs will be stored in .npy format with randomly generated
        matching filenames for retrieval

        Args:
            x (torch.Tensor) : An input batched tensor that can be consumed by self.model
            outsize (tuple(int, int)) : A tuple indicating the size that input data arrays should be
            written out to

        Returns:
            (bool) : True if writing was succesful

        '''
        print("write_data_loader_embeddings")

        # TODO: PUT a stop in 280 instances
        embeddings = []
        input_images = []
        input_labels = []
        # Generate embeddings

        BATCH_SIZE = args.batch_size
        for idx, batch_img in enumerate(tqdm(x)):

            if idx > 5:
                break

            batch_imgs, batch_labels = batch_img[0], batch_img[1]
            input_images.append(batch_imgs[:, 0:3, :, :])
            input_labels.append(batch_labels)

            embs = self.generate_embeddings(batch_imgs)
            embeddings.append(embs)
            with open(os.path.join('../Outputs', self.run, 'metadata.tsv'), "a") as f:
                bb = list(map(lambda i: output_classes[i], batch_labels['label'].numpy()))
                for batch_label in bb:
                    f.write("{}\n".format(batch_label))
            with open(os.path.join('../Outputs', self.run, 'metadata-textures.tsv'), "a") as f:
                ll = list(map(lambda i: texture_classes[i], batch_labels['texture'].numpy()))
                for batch_label in ll:
                    f.write("{}\n".format(batch_label))

        # Detach from graph
        detached_embeddings = []

        for embs in embeddings:
            detached_embeddings.append(embs.detach().cpu().numpy())
        detached_embeddings = np.vstack(detached_embeddings)

        # Start writing to output folders
        for i in range(len(detached_embeddings)):
            # key = str(np.random.random())[-7:]
            # tensor2np(embeddings[i % 6][int(i / 6)], outsize)
            # tensor2np(x[i], outsize)) :  (28, 28, 3)
            # x[i] : Tensor(3, 300, 290)

            np.save(self.imgs_folder + r"/" + str(i).zfill(4) + '.npy',
                    tensor2np(input_images[int(i / BATCH_SIZE)][i % BATCH_SIZE], outsize))
            np.save(self.embs_folder + r"/" + str(i).zfill(4) + '.npy', detached_embeddings[i])
        return True

    def _create_writer(self, name):
        '''
        Create a TensorboardX writer object given an experiment name and assigns it to self.writer

        Args:
            name (str): Optional, an experiment name for the writer, defaults to self.name

        Returns:
            (bool): True if writer was created succesfully

        '''

        if self.name is None:
            name = 'Experiment_' + str(np.random.random())
        else:
            name = self.name

        dir_name = os.path.join(self.tensorboard_folder,
                                name)

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        else:
            print("Warning: logfile already exists")
            print("logging directory: " + str(dir_name))

        logdir = dir_name
        self.writer = SummaryWriter(logdir=logdir)
        return (True)

    def create_tensorboard_log(self, class_labels=None):

        '''
        Write all images and embeddings from imgs_folder and embs_folder into a tensorboard log
        '''

        if self.writer is None:
            self._create_writer(self.name)

        ## Read in
        all_embeddings = [np.load(os.path.join(self.embs_folder, p)) for p in
                          sorted(os.listdir(self.embs_folder), key=lambda x: int(os.path.basename(x).split('.')[0])) if
                          p.endswith('.npy')]
        all_images = [np.load(os.path.join(self.imgs_folder, p)) for p in
                      sorted(os.listdir(self.imgs_folder), key=lambda x: int(os.path.basename(x).split('.')[0])) if
                      p.endswith('.npy')]
        all_images = [np.moveaxis(a, 2, 0) for a in all_images]  # (HWC) -> (CHW)

        ## Stack into tensors
        all_embeddings = torch.Tensor(np.array(all_embeddings))  # Tensor: (32, 32)
        all_images = torch.Tensor(np.array(all_images))  # Tensor: (32, 3, 28, 28)

        print(all_embeddings.shape)
        print(all_images.shape)
        pth = os.path.join('../Outputs', self.run, 'metadata.tsv')
        df = pd.read_csv(pth, sep="\n", header=None)
        class_labels = df.iloc[:, 0].tolist()

        pth = os.path.join('../Outputs', self.run, 'metadata-textures.tsv')
        df = pd.read_csv(pth, sep="\n", header=None)
        class_textures = df.iloc[:, 0].tolist()

        grid = torchvision.utils.make_grid(all_images)
        self.writer.add_image('images', grid, 0)

        # plot image
        for im, l in zip(all_images, class_labels):
            fig = plt.figure(figsize=(10, 10))
            plt.axis('off')
            img = torch.permute(im, (1, 2, 0)).numpy()
            plt.title(l)
            plt.imshow(img)
            plt.show()

        # add_embedding(mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None)[SOURCE]
        # Add embedding projector data to summary. Parameters:
        # mat (torch.Tensor or numpy.array) – A matrix which each row is the feature vector of the data point --> (N,D), where N is number of data and D is feature dimension
        # metadata (list) – A list of labels, each element will be convert to string
        # label_img (torch.Tensor) – Images correspond to each data point: --> (N,C,H,W)
        # global_step (int) – Global step value to record
        # tag (string) – Name for the embedding
        ll = list(zip(class_labels, class_textures))
        self.writer.add_embedding(all_embeddings, metadata=ll, metadata_header=["Labels", "Textures"],
                                  label_img=all_images)


def tensor2np(tensor, resize_to=None):
    '''
    Convert an image tensor to a numpy image array and resize

    Args:
        tensor (torch.Tensor): The input tensor that should be converted
        resize_to (tuple(int, int)): The desired output size of the array

    Returns:
        (np.ndarray): The input tensor converted to a channel last resized array
    '''

    out_array = tensor.detach().cpu().numpy()
    out_array = np.moveaxis(out_array, 0, 2)  # (CHW) -> (HWC)

    if resize_to is not None:
        out_array = cv2.resize(out_array, dsize=resize_to, interpolation=cv2.INTER_CUBIC)

    return (out_array)
