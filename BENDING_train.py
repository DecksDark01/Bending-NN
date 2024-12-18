import numpy as np
import torch 

from BENDING_user_fc import view_errors
from BENDING_nn import DeepONet,CreateDataset

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader



def Pde_loss(model_train: torch.nn.Module,collocation_points_: torch.Tensor, lb_pde_: float):

    w_pred = model_train(collocation_points_)

    w_x = torch.autograd.grad(w_pred,collocation_points_, grad_outputs=torch.ones_like(collocation_points_),create_graph=True)[0][:, 0]
    w_xx = torch.autograd.grad(w_x,collocation_points_, grad_outputs=torch.ones_like(collocation_points_),create_graph=True)[0][:, 0]
    w_xxx = torch.autograd.grad(w_xx,collocation_points_, grad_outputs=torch.ones_like(collocation_points_),create_graph=True)[0][:, 0]
    w_xxxx = torch.autograd.grad(w_xxx,collocation_points_, grad_outputs=torch.ones_like(collocation_points_),create_graph=True)[0][:, 0]
    
    w_y = torch.autograd.grad(w_pred,collocation_points_, grad_outputs=torch.ones_like(collocation_points_),create_graph=True)[0][:, 1]
    w_yy = torch.autograd.grad(w_y,collocation_points_, grad_outputs=torch.ones_like(collocation_points_),create_graph=True)[0][:, 1]
    w_yyy = torch.autograd.grad(w_yy,collocation_points_, grad_outputs=torch.ones_like(collocation_points_),create_graph=True)[0][:, 1]
    w_yyyy = torch.autograd.grad(w_yyy,collocation_points_, grad_outputs=torch.ones_like(collocation_points_),create_graph=True)[0][:, 1]
    
    w_xxy = torch.autograd.grad(w_xx,collocation_points_, grad_outputs=torch.ones_like(collocation_points_),create_graph=True)[0][:, 1]
    w_xxyy = torch.autograd.grad(w_xxy,collocation_points_, grad_outputs=torch.ones_like(collocation_points_),create_graph=True)[0][:, 1]
    
    residual = w_xxxx + 0.5*w_xxyy + w_yyyy

    return lb_pde_ * torch.mean(residual**2)

def train_model(model_train: DeepONet,branch_in_: np.ndarray, coll_points_: np.ndarray, output_: np.ndarray, lb_in: float):
    """
        function that trains a deep operator network (DeepONet)

        Parameters:
            model   (DeepONet)     : the DeepONet to be trained
            x_branch    (torch.tensor) : the branch input data
            x_trunk (torch.tensor) : the trunk input data
            u_out   (torch.tensor) : the target values
    """

    dataset = CreateDataset(branch_in_,coll_points_,output_)
    train_loader = DataLoader(dataset, batch_size=100, shuffle=True)

    criterion = nn.MSELoss()

    max_epochs = 4000
    reg_param = 1e-7
    learning_rate = 1e-3

    optimizer = optim.Adam(model_train.parameters(),lr=learning_rate,weight_decay=reg_param,betas=(0.5,0.9))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)   

    loss_hist = []

    for epoch in range(max_epochs):
        losses = []
        for batch in train_loader:
          
            branch_in_ = batch['branch_in_data']  
            coll_points_ = batch['trunk_in_data']  
            output = batch['output_train_data']  
            
            optimizer.zero_grad()
            output_pred = model_train.forward(branch_input_=branch_in_,trunk_input_=coll_points_)
            loss = criterion(output,output_pred)
            pde_loss = Pde_loss(model_train.trunk_net, collocation_points_=coll_points_, lb_pde_=lb_in)

            total_loss = loss + pde_loss

            total_loss.backward()
            losses.append(total_loss.item())

            optimizer.step()
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]


        avg_loss = np.mean(losses)
        loss_hist.append(avg_loss)

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{max_epochs}] - Loss: {loss.item()},Learning Rate: {current_lr:.6f}")
    
    view_errors(loss=loss_hist)










