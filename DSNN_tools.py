# %% 
import random
import time
import numpy as np
import torch
from torch import nn, optim
from general_tools import time_str#, early_stop_check

# To guarantee same results for every running, which might slow down the training speed
torch.set_default_dtype(torch.float64)
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

def DSNN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, device, bias_status):
    # Define DSNN
    class DSNN(nn.Module):
        def __init__(self):
            super(DSNN, self).__init__()

            layer_non = [nn.Linear(in_s + state_s, state_non_neuron[0], bias=bias_status), torch.nn.Tanh()]
            if state_non_layer_s > 1:
                for ii in range(state_non_layer_s - 1):
                    layer_non.append(nn.Linear(state_non_neuron[ii], state_non_neuron[ii + 1], bias=bias_status))
                    layer_non.append(torch.nn.Tanh())
            layer_non.append(nn.Linear(state_non_neuron[-1], state_s, bias=bias_status))
            self.StateNet_non = nn.Sequential(*layer_non)
            self.StateNet_lin = nn.Linear(in_s + state_s, state_s, bias=bias_status)

            layer_non = [nn.Linear(in_s + state_s, out_non_neuron[0], bias=bias_status), torch.nn.Tanh()]
            if out_non_layer_s > 1:
                for ii in range(out_non_layer_s - 1):
                    layer_non.append(nn.Linear(out_non_neuron[ii], out_non_neuron[ii + 1], bias=bias_status))
                    layer_non.append(torch.nn.Tanh())
            layer_non.append(nn.Linear(out_non_neuron[-1], out_s, bias=bias_status))
            self.OutputNet_non = nn.Sequential(*layer_non)
            self.OutputNet_lin = nn.Linear(in_s + state_s, out_s, bias=bias_status)

        def forward(self, input_state):
            state_d_non = self.StateNet_non(input_state)
            state_d_lin = self.StateNet_lin(input_state)
            state_d = state_d_non + state_d_lin
            output_non = self.OutputNet_non(input_state)
            output_lin = self.OutputNet_lin(input_state)
            output = output_non + output_lin
            output_state_d = torch.cat((output, state_d), dim=1)
            return output_state_d

    DSNN_model = DSNN().to(device)
    return DSNN_model


def DSNN_cal(Amp_N, Nt, u, DSNN_model, state_s, out_s, device):
    u_torch = torch.tensor(u).to(device)
    y_pre_torch = torch.zeros(Nt, Amp_N, out_s).to(device)
    x = torch.zeros(Amp_N, state_s).to(device)
    for i in range(Nt - 1):
        u = u_torch[i, :, :]
        x0 = x
        u_x = torch.cat((u, x), dim=1)
        y_pre_torch[i, :, :] = DSNN_model(u_x)[:, 0:out_s]
        x = DSNN_model(u_x)[:, out_s:]

    u_x = torch.cat((u, x), dim=1)
    y_pre_torch[i+1, :, :] = DSNN_model(u_x)[:, 0:out_s]
    return y_pre_torch

def DSNN_training(Amp_N, Nt, u, DSNN_model, N, y_ref, in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, device, bias_status):
    y_ref_torch = torch.tensor(y_ref).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(DSNN_model.parameters(), 1e-3, weight_decay=1e-8)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-4, eps=1e-8)
    # loss_last100=0.0
    im=1
    loss_all = np.zeros((N + 1, 1))

    y_pre_torch = DSNN_cal(Amp_N, Nt, u, DSNN_model, state_s, out_s, device)
    loss = criterion(y_pre_torch, y_ref_torch)
    loss_all[0:1, :] = loss.item()
    loss_m = loss.item()
    DSNN_model_m = DSNN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, device, bias_status)
    DSNN_model_m.load_state_dict(DSNN_model.state_dict())

    start = time.time()
    for i in range(N):
        DSNN_model.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        y_pre_torch = DSNN_cal(Amp_N, Nt, u, DSNN_model, state_s, out_s, device)
        loss = criterion(y_pre_torch, y_ref_torch)
        i1 = i + 1
        loss_all[i1:i1 + 1, :] = loss.item()

        if loss.item() < loss_m:
            loss_m = loss.item()
            DSNN_model_m.load_state_dict(DSNN_model.state_dict())
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
        
    end = time.time()
    cost_time = end - start
    cost_time_str = time_str(cost_time)
    print('Total training time: ' + cost_time_str + ', final loss: ' + str(loss.item()))

    val, idx = min((val, idx) for (idx, val) in enumerate(loss_all))
    print('Minimal loss: ' + str(val.item())+ ' (iteration: ' + str(idx) +')')
    return DSNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str