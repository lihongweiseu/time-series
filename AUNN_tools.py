# %% 
import random
import time
import numpy as np
import torch
from torch import nn, optim
from general_tools import time_str#, early_stop_check

# chose to use gpu or cpu
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# To guarantee same results for every running, which might slow down the training speed
torch.set_default_dtype(torch.float64)
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

def AUNN_create(in_s, non_layer_s, non_neuron, device, bias_status):
    # Define AUNN
    class AUNN(nn.Module):
        def __init__(self):
            super(AUNN, self).__init__()

            layer_non = [nn.Linear(in_s, non_neuron[0], bias=bias_status), torch.nn.Tanh()]
            if non_layer_s > 1:
                for ii in range(non_layer_s - 1):
                    layer_non.append(nn.Linear(non_neuron[ii], non_neuron[ii + 1], bias=bias_status))
                    layer_non.append(torch.nn.Tanh())
            layer_non.append(nn.Linear(non_neuron[-1], 1, bias=bias_status))
            self.non = nn.Sequential(*layer_non)
            self.lin = nn.Linear(in_s, 1, bias=bias_status)

        def forward(self, in_s):
            output_non = self.non(in_s)
            output_lin = self.lin(in_s)
            output = output_non + output_lin
            return output

    AUNN_model = AUNN().to(device)
    return AUNN_model


# def AUNN_cal(Amp_N, Nt, u, AUNN_model, x_s, in_s, device):
#     u_torch = torch.tensor(u).to(device)
#     y_pre_torch = torch.zeros(Amp_N, Nt).to(device)
#     temp1 = torch.zeros(Amp_N, x_s).to(device)
#     y_s=in_s-x_s
#     temp2 = torch.zeros(Amp_N, y_s).to(device)
#     for i in range(Nt):
#         temp1[:, 0:x_s - 1]=temp1[:, 1:].clone()
#         temp1[:, x_s - 1:]=u_torch[:, i:i + 1]
#         in_torch = torch.cat((temp1, temp2), dim=1)
#         y_pre_torch[:, i:i + 1] = AUNN_model(in_torch)
#         temp2[:, 0:y_s - 1]=temp2[:, 1:].clone()
#         temp2[:, y_s - 1:]=y_pre_torch[:, i:i + 1]
#     return y_pre_torch


def AUNN_cal(Amp_N, Nt, u, AUNN_model, x_s, in_s, device):
    y_pre_torch = torch.zeros(Amp_N, Nt).to(device)
    input= np.zeros((Amp_N, in_s))
    y_s=in_s-x_s

    if x_s==1:
        if y_s==0:
            for i in range(Nt):
                input[:, x_s - 1:x_s]=u[:, i:i + 1]
                in_torch=torch.tensor(input).to(device)
                y_pre_torch[:, i:i + 1] = AUNN_model(in_torch)
        elif y_s==1:
            for i in range(Nt):
                input[:, x_s - 1:x_s]=u[:, i:i + 1]
                in_torch=torch.tensor(input).to(device)
                y_pre_torch[:, i:i + 1] = AUNN_model(in_torch)
                input[:, in_s -1:]=y_pre_torch.cpu().detach().numpy()[:, i:i + 1]
        else:
            for i in range(Nt):
                input[:, x_s - 1:x_s]=u[:, i:i + 1]
                in_torch=torch.tensor(input).to(device)
                y_pre_torch[:, i:i + 1] = AUNN_model(in_torch)
                input[:, x_s:in_s - 1]=input[:, x_s + 1:in_s]
                input[:, in_s -1:]=y_pre_torch.cpu().detach().numpy()[:, i:i + 1]
    else:
        if y_s==0:
            for i in range(Nt):
                input[:, 0:x_s - 1]=input[:, 1:x_s]
                input[:, x_s - 1:x_s]=u[:, i:i + 1]
                in_torch=torch.tensor(input).to(device)
                y_pre_torch[:, i:i + 1] = AUNN_model(in_torch)
        elif y_s==1:
            for i in range(Nt):
                input[:, 0:x_s - 1]=input[:, 1:x_s]
                input[:, x_s - 1:x_s]=u[:, i:i + 1]
                in_torch=torch.tensor(input).to(device)
                y_pre_torch[:, i:i + 1] = AUNN_model(in_torch)
                input[:, in_s -1:]=y_pre_torch.cpu().detach().numpy()[:, i:i + 1]
        else:
            for i in range(Nt):
                input[:, 0:x_s - 1]=input[:, 1:x_s]
                input[:, x_s - 1:x_s]=u[:, i:i + 1]
                in_torch=torch.tensor(input).to(device)
                y_pre_torch[:, i:i + 1] = AUNN_model(in_torch)
                input[:, x_s:in_s - 1]=input[:, x_s + 1:in_s]
                input[:, in_s -1:]=y_pre_torch.cpu().detach().numpy()[:, i:i + 1]

    return y_pre_torch

def AUNN_training(Amp_N, Nt, u, AUNN_model, x_s, in_s, N, y_ref, non_layer_s, non_neuron, device, bias_status):
    y_ref_torch = torch.tensor(y_ref).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(AUNN_model.parameters(), 1e-3, weight_decay=1e-8)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-4, eps=1e-8)
    # loss_last100=0.0
    im=1
    loss_all = np.zeros((N + 1, 1))

    y_pre_torch = AUNN_cal(Amp_N, Nt, u, AUNN_model, x_s, in_s, device)
    loss = criterion(y_pre_torch, y_ref_torch)
    loss_all[0:1, :] = loss.item()
    loss_m = loss.item()
    AUNN_model_m = AUNN_create(in_s, non_layer_s, non_neuron, device, bias_status)
    AUNN_model_m.load_state_dict(AUNN_model.state_dict())

    start = time.time()
    for i in range(N):
        AUNN_model.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        y_pre_torch = AUNN_cal(Amp_N, Nt, u, AUNN_model, x_s, in_s, device)
        loss = criterion(y_pre_torch, y_ref_torch)
        i1 = i + 1
        loss_all[i1:i1 + 1, :] = loss.item()

        if loss.item() < loss_m:
            loss_m = loss.item()
            AUNN_model_m.load_state_dict(AUNN_model.state_dict())
            im=i1

        if i1 % 10 == 0 or i == 0:
            print(f"Iteration: {i1}/{N}({i1/N*100:.2f}%), loss: {loss.item()}")
            end = time.time()
            per_time = (end - start) / i1
            per_time_str = time_str(per_time)
            cost_time_str = time_str(end - start)
            print('Average time per training: '+per_time_str+', Cumulative training time: '+cost_time_str)
            left_time = (N - i1) * per_time
            left_time_str = time_str(left_time)
            print('Executed at ' + time.strftime('%d %b %Y %H:%M:%S', time.localtime()) +
                  ', left time: ' + left_time_str + '\n')
        
        # if i1==threshold_N-100:
        #     loss_last100=sum(loss_all[i1-99:i1+1])
        
        # if i1>threshold_N-100:
        #     scheduler.step(loss)

        # if i1>=threshold_N:
        #     early_stop, loss_last100 = early_stop_check(loss_all, loss_last100, i1)
        #     if early_stop==1:
        #         if i1<N:
        #             loss_all=loss_all[:i1-N, :]
        #             print("Early stopping")
        #         break

    end = time.time()
    cost_time = end - start
    cost_time_str = time_str(cost_time)
    print('Total training time: ' + cost_time_str + ', final loss: ' + str(loss.item()))

    val, idx = min((val, idx) for (idx, val) in enumerate(loss_all))
    print('Minimal loss: ' + str(val.item())+ ' (iteration: ' + str(idx) +')')
    return AUNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str