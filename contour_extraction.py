# %matplotlib inline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import time

import skimage.data
import skimage.color
from skimage.transform import rescale, resize

import scipy.misc
import scipy.signal

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
from torchvision import models
import torchvision.transforms as transforms

from matplotlib import rcParams
rcParams['axes.grid'] = False


def batch(batch_size, images, labels, training=True):
    """Create a batch of examples.

    This creates a batch of input images and a batch of corresponding
    ground-truth labels. We assume CUDA is available (with a GPU).

    Args:
        batch_size: An integer.
        images: images to train on
        labels: class labels of each image
        training: A boolean. If True, grab examples from the training
        set; otherwise, grab them from the validation set.

    Returns:
        A tuple,
        input_batch: A Variable of floats with shape
        [batch_size, 1, height, width]
        label_batch: A Variable of ints with shape
        [batch_size].
    """
    if training:
        random_ind = np.random.choice(images.shape[0], size=batch_size, replace=False)
        input_batch = images[random_ind]
        label_batch = labels[random_ind]
    else:
        input_batch = images[:batch_size]
        label_batch = labels[:batch_size]
    
    input_batch = torch.tensor(input_batch, requires_grad=False, device='cuda')
    label_batch = torch.tensor(label_batch, requires_grad=False, device='cuda')
    
    return input_batch, label_batch

def train_step(model, train_images, train_labels, optimizer, batch_size=128):
    """Conducts one training iteration
    
    Trains on one batch of input images, computes loss, and back propagates.
    
    Args:
        model: network
        train_images: images used for training
        train_labels: class labels for each training image
        optimizer: type of optimization
        batch_size: an integer
    
    Returns:
        loss, error
    """
    model.train()

    input_batch, label_batch = batch(batch_size, train_images, train_labels, training=True)
    input_batch = input_batch.float()
    output_batch = model(input_batch)
    loss = F.cross_entropy(output_batch, label_batch)
    # _, pred_batch = torch.max(output_batch, dim=1)
    # error_rate = 1.0 - (pred_batch == label_batch).float().mean()

    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    
    return loss.item() # , error_rate.item()

def val(model, val_images, val_labels):
    """Conducts validation step

    Passes validation images through model, computes loss, and error rate.
    
    Args:
        model: network
        val_images: images used for validation
        val_labels: class labels for validation images
    
    Returns:
        loss, error
    """
    
    model.eval()
    input_batch, label_batch = batch(val_images.shape[0], val_images, val_labels, training=False)
    input_batch = input_batch.float()
    output_batch = model(input_batch)

    loss = F.cross_entropy(output_batch, label_batch)
    # _, pred_batch = torch.max(output_batch, dim=1)
    # error_rate = 1.0 - (pred_batch == label_batch).float().mean()
    
    return loss.item() # , error_rate.item()

def train(model, optimizer, train_images, train_labels, val_images, val_labels, num_steps, num_steps_per_val):
    """Executes entire training procedure.
    
    Iterates through each step, number of steps determined by user.
    Plots training, validation loss and training, validation error.
    
    Args:
        model: network
        optimizer: type of optimization
        train_images: images used for training
        train_labels: class labels for training images
        val_images: images used for validation
        val_labels: class labels for validation images
        num_steps: number of iterations
        num_steps_per_val: determines how often validation is done
    
    Returns:
        None
    """
    # initialization
    for module in model.children():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight.data)

    info = []
    fig, ax = plt.subplots(2, 1, sharex=True)
    best_val_err = float('inf')
    for step in range(num_steps):
        # train_loss, train_err = train_step(model, train_images, train_labels, optimizer)
        train_loss = train_step(model, train_images, train_labels, optimizer)
        if step % num_steps_per_val == 0:
        # val_loss, val_err = val(model, val_images, val_labels)
        val_loss = val(model, val_images, val_labels)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('Step {:5d}: Obtained a best validation loss of {:.3f}.'.format(step, best_val_loss))
            torch.save(model, '/content/gdrive/My Drive/bland_model' + str(img_size))
        info.append([step, train_loss, val_loss]) # , train_err, val_err])
        print('Step {:5d}: Train Loss - {:.3f}, Val Loss - {:.3f}.'.format(step, train_loss, val_loss))
        x, y11, y12 = zip(*info)
        ax[0].plot(x, y11, c='r')
        ax[0].plot(x, y12, c='g')
        ax[0].legend(['Train loss', 'Val loss'])


class VolAutoEncoder(nn.Module):
    """
       This is the standard way to define your own network in PyTorch. You typically choose the components
       (e.g. LSTMs, linear layers etc.) of your network in the __init__ function.
       You then apply these layers on the input step-by-step in the forward function.
       You can use torch.nn.functional to apply functions such as F.relu, F.sigmoid, F.softmax.
       Be careful to ensure your dimensions are correct after each step.
    """

    def __init__(self):
        super(VolAutoEncoder, self).__init__()

        self.encoder=nn.Sequential(
            nn.Dropout(p=0.5),  # dropout_P=0.5
            nn.Conv3d(1, 64, (9, 9, 9), stride=3),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 256, (8, 8, 8), stride=4),
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Linear(186624, 186624),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # dropout_P=0.5
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 64, (5, 5, 5), stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 1, (20, 20, 20), stride=5),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        This function defines how we use the components of our network to operate on an input batch.
        """
        encoded = self.encoder(x)
        y = encoded.reshape(186624)
        z = self.linear(y)
        w = z.reshape((256, 9, 9, 9))
        decoded = self.decoder(w)
        r = decoded.reshape(1728000)  # featuresOut=inD*inD*inD=120*120*120=1728000
        out = self.sigmoid(r)

        return out


# load from pickle


# separate into training and validation

assert torch.cuda.is_available()

# convert to cuda
RGB_train_x = train_x.cuda()
RGB_valid_x = valid_x.cuda()
test_x = test_x.cuda()

m = VolAutoEncoder()
m.cuda()
optimizer = torch.optim.Adam(m.parameters(), lr=0.1, weight_decay=0.0000001)
train(m, optimizer, train_x, train_y, valid_x, valid_y, 500, 25)

# m_loaded = torch.load("/content/gdrive/My Drive/" ...)
m_loaded.eval()

output = []
for i in range(len(test_x)):
  result = m_loaded(test_x[i:i+1,:,:,:].float())
  # result should be 120x120x120