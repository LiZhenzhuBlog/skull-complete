import argparse

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--model', help="File of model to test")

def test_instance(model, test_data, test_labels):
    # set model to evaluation mode
    model.eval()

    print("Total: " + str(test_data.shape[0]))
    loss = 0
    conf_matrix = np.zeros((3,3))
    for i in range(test_data.shape[0]):    
        print(i)
        if "UNet" not in model_file:
            model.encoder.__delitem__(0) #remove the dropout layer
        input_data = torch.unsqueeze(test_data[i], 0)
        input_data = torch.unsqueeze(input_data, 0)
        result = model(input_data.float())
        _, output = torch.max(result, dim=1)
        output = torch.squeeze(output, dim=0)

        result = result.float()
        label = torch.unsqueeze(test_labels[i], dim=0)
        label = label.long()
        loss = loss + F.cross_entropy(result, label)

        output = output.data.cpu().numpy()
        label = torch.squeeze(label, dim=0)
        label = label.data.cpu().numpy()

        result = torch.squeeze(result, dim=0)
        result = result.data.cpu().numpy()

        for j in range(120):
            for k in range(120):
                for l in range(120):
                    conf_matrix[int(output[j,k,l]), int(label[j,k,l])] = conf_matrix[int(output[j,k,l]), int(label[j,k,l])] + 1

    print("Average loss: " + str(loss/i))

    print(conf_matrix)

    print(conf_matrix/conf_matrix.sum(axis=0))


args = parser.parse_args()

# load net
model_file = args.model
m = torch.load(model_file)

print("Loading test datasets")
test_x = Variable(torch.from_numpy(np.load("test_x.npy")))
test_y = Variable(torch.from_numpy(np.load("test_y.npy")))

# eval net
print("Evaluating")
test_instance(m, test_x, test_y)
