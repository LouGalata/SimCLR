# Classification on deformable objects

- Contrastive Learning ([SimCLR](https://sthalles.github.io/simple-self-supervised-learning/))
- Transfer Learning on ResNet and ImageNet


![Image of SimCLR Arch](https://sthalles.github.io/assets/contrastive-self-supervised/cover.png)

## Project Information
***Project in progress!***

Training the models
- ```run.py``` : create embedding space using the SimCLR self-supervised learning algorithm 
- ```visualizaing.py```: visualize the created embedding space on 3d space using tensorboard
- ```linear_evaluation.py```: supervised learning using linear classifier on the embedded space
- ```transfer_learning_classifier```: transfer learning directly on the input instances

Results
- ```Outputs```: stored the runs correlated to the creation of the embedding space 
- ```runs```: stored the models and accuracy from the transfer learning and the contrastive learning algorithms
- ```images```: Confusion matrices and classification reports 


## Installation

```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
$ python run.py
```

## Config file

Before running SimCLR, make sure you choose the correct running configurations. You can change the running configurations by passing keyword arguments to the ```run.py``` file.

```python

$ python run.py -data ./datasets --dataset-name stl10 --log-every-n-steps 100 --epochs 100 

```

