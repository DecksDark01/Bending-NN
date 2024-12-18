import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch.utils.data import Dataset

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def check_and_modify_tensor(self):
    if self.requires_grad and self.dtype == torch.float32:
        return
    if not self.requires_grad:
        self.requires_grad_(True)
    if self.dtype != torch.float32:
        self.data = self.data.float()  
    return

def view_errors(loss):
    """
    Function that automatically plots loss at each epoch.

    Parameters:
    loss (list): loss history for every epoch recorded during training

    returns: none
    """

    plt.figure(figsize=(10,6))
    plt.plot(np.log10(loss), color='blue', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Loss at each epoch')
    
    plt.legend()
    plt.grid()
    plt.show()

def init_weights_biases(layer):   
    """
    Function that initializes the weights and biases. Created to be applied at the MLP class
    """ 
    if isinstance(layer, nn.Linear):
        # Initialize weights with a uniform distribution 
        nn.init.uniform_(layer.weight, -1, 1)
        # Initialize biases with a uniform distribution 
        nn.init.uniform_(layer.bias, -1, 1)

class CreateDataset(Dataset):
    def __init__(self, branch_data_, trunk_data_, output_data_):
        super(CreateDataset, self).__init__()

        self.branch_data = torch.tensor(branch_data_, dtype=torch.float32)
        self.trunk_data = torch.tensor(trunk_data_, dtype=torch.float32)
        self.output_data = torch.tensor(output_data_, dtype=torch.float32)

    def __len__(self):
        return len(self.trunk_data)
    
    def __getitem__(self, index):
        return {
            'trunk_in_data': self.trunk_data[index],
            'branch_in_data': self.branch_data[index],
            'output_train_data': self.output_data[index]
        }
