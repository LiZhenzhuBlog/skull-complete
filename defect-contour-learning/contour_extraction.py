import argparse
import numpy as np
import time

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
from torchvision import models
import torchvision.transforms as transforms

from net import VolAutoEncoder as V1
from net2 import VolAutoEncoder as V2
from UNet_3d import UNet

weights = [1, 150, 10000 ]
weights = np.asarray(weights)
weights = torch.Tensor(weights)

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
        print(random_ind)
        input_batch = images[random_ind]
        label_batch = labels[random_ind]
    else:
        input_batch = images[:batch_size]
        label_batch = labels[:batch_size]
    
    input_batch = input_batch.clone().detach().requires_grad_(True)
    label_batch = label_batch.clone().detach().requires_grad_(True)

    # input_batch = torch.tensor(input_batch, requires_grad=False, device='cpu')
    # label_batch = torch.tensor(label_batch, requires_grad=False, device='cpu')
    
    input_batch = torch.unsqueeze(input_batch, 1)
    # label_batch = torch.unsqueeze(label_batch, 1)

    return input_batch, label_batch

def train_step(model, train_images, train_labels, optimizer, batch_size=4):
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
    label_batch = label_batch.long()
    input_batch = input_batch.float()
    output_batch = model(input_batch) # batch_size 3 120 120 120
   
     
    loss = F.cross_entropy(output_batch, label_batch, weights)
    # print(loss)

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
    label_batch = label_batch.long()

    loss = F.cross_entropy(output_batch, label_batch, weights)
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
    #TODO: insert option to resume training
    #TODO: automate naming convention for saving checkpoints of models

    for module in model.children():
        if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.ConvTranspose3d) or isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)

    info = []
    best_val_loss = float('inf')

    start = time.time()
    for step in range(num_steps+1):
        # train_loss, train_err = train_step(model, train_images, train_labels, optimizer)
        train_loss = train_step(model, train_images, train_labels, optimizer)
        if step % num_steps_per_val == 0:
            elapsed = (time.time() - start)/60.0
            start = time.time()
            # val_loss, val_err = val(model, val_images, val_labels)
            val_loss = val(model, val_images, val_labels)
            print("Step {:5d} - Current loss: {:.3f}, Minutes elapsed: {:.3f}".format(step, val_loss, elapsed))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print('Step {:5d}: Obtained a best validation loss of {:.3f}.'.format(step, best_val_loss))
                torch.save(model, 'ACE_model_' + args.model + '2.pt')
        if step == num_steps:
            torch.save(model, 'ACE_model_'+ args.model + '2_iters' + str(num_steps) + '.pt')


parser = argparse.ArgumentParser()
parser.add_argument('--model', help="File of model to test")
args = parser.parse_args()

print("Loading numpy files")
# load from npy files
train_x = Variable(torch.from_numpy(np.load("train_x.npy")))
train_y = Variable(torch.from_numpy(np.load("train_y.npy")))
valid_x = Variable(torch.from_numpy(np.load("valid_x.npy")))
valid_y = Variable(torch.from_numpy(np.load("valid_y.npy")))

torch.manual_seed(1)

print("Done loading")
if args.model == "net":
    m = V1()
elif args.model == "net2":
    m = V2()
elif args.model == "UNet":
    m = UNet(in_dim=1, out_dim=3, num_filters=4)
else:
    raise Exception("No such network has been implemented.")

optimizer = torch.optim.Adam(m.parameters(), lr=0.0002, weight_decay=0.0001)
train(m, optimizer, train_x, train_y, valid_x, valid_y, 800, 25)

# optimizer = torch.optim.SGD(m.parameters(), lr=0.07, momentum=0.9)
# train(m, optimizer, train_x, train_y, valid_x, valid_y, 500, 25)
