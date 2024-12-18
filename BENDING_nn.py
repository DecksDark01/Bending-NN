import torch 
import torch.nn as nn

from BENDING_user_fc import init_weights_biases
from BENDING_user_fc import Sine
from BENDING_user_fc import check_and_modify_tensor
torch.Tensor.check_and_modify = check_and_modify_tensor


class MLP(nn.Module):
    """
    Creates a feed-forward Multi-Layer Perceptor Neural network.
    """
    def __init__(self,input_dim: int,output_dim: int,activation: str,last_layer: False):
        super(MLP,self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.last_layer = last_layer

        width = 20
        depth = 6

        if activation == 'Tanh':
            activation_fc = nn.Tanh()
        elif activation == 'sine':
            activation_fc = Sine()

        layers = []

        layers.append(nn.Linear(self.input_dim,width))
        layers.append(activation_fc)

        for _ in range(depth-2):
            layers.append(nn.Linear(width,width))
            layers.append(activation_fc)
        
        layers.append(nn.Linear(width,self.output_dim))

        self.model = nn.Sequential(*layers)
        self.model.apply(init_weights_biases)

    def forward(self, x):
        if self.last_layer:
            return torch.sigmoid(self.model(x))
        else:
            return self.model(x)


class DeepONet(nn.Module):
    """
    Implementation of the Deep Operator Network. This class unifies the branch and trunk network and computes the dot product of the two network outputs 
    (using the logic of basis functions expansions).
    """
    def __init__(self,k: int, n: int, p: int, activation_branch: str, activation_trunk: str, last_activation_branch: bool, last_activation_trunk: bool):
        """
            Creates the DON using the following parameters
            Parameters:
            k   (int) : the input size of the branch network (k,1)
            n   (int) : the input size of the trunk network
            p   (int) : output dimension of branch and trunk network (assumed to be the same)
        """
        
        super(DeepONet, self).__init__()
        self.branch_net = MLP(input_dim=k, output_dim=p,activation=activation_branch,last_layer=last_activation_branch)
        self.trunk_net = MLP(input_dim=n, output_dim= p,activation=activation_trunk,last_layer=last_activation_trunk)
        print(f'Branch and trunk network are created in DeepOnet class with input dim branch network : {k} and trunk network : {n} and awaiting output dim : {p}')
    
    def forward(self,branch_input_: torch.Tensor,trunk_input_: torch.Tensor):
        """
            evaluates the operator network output
            branch_input_ : P (external load) evaluated at sensor points (k)
            trunk_input_ : nodes where w is evaluated (w is governed by an ellicptic steady state pde)
            returns a torch scalar (prediction of w (discplacement))

            returns (float): a torch scalar (prediction of w (discplacement))
        """
        trunk_input_.check_and_modify_tensor()
        
        branch_out = self.branch_net.forward(branch_input_)
        trunk_out = self.trunk_net.forward(trunk_input_)

        return torch.sum(branch_out * trunk_out, dim=1, keepdim=True)  