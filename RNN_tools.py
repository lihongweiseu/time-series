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

def RNN_create(in_s, layer_s, hidden_s, device, bias_status):
    # Define RNN
    class Rnn(nn.Module):
        def __init__(self):
            super(Rnn, self).__init__()
            self.rnn = nn.RNN(in_s, hidden_s, layer_s, bias = bias_status)
            self.linear = nn.Linear(hidden_s, 1, bias = bias_status)

        def forward(self, u):
            h0 = torch.zeros(layer_s, np.size(u, 1), hidden_s).to(device)
            y, hn = self.rnn(u, h0)
            y = self.linear(y)
            return y

    RNN_model = Rnn().to(device)
    return RNN_model

def RNN_training(Amp_N, Nt, u, RNN_model, in_s, layer_s, hidden_s, N, y_ref, device, bias_status):
    y_ref_torch = torch.tensor(y_ref).reshape([Nt, Amp_N, in_s]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(RNN_model.parameters(), 1e-3, weight_decay=1e-8)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-4, eps=1e-8)
    # loss_last100=0.0
    im=1
    loss_all = np.zeros((N + 1, 1))

    u_torch = torch.tensor(u).reshape([Nt, Amp_N, in_s]).to(device)
    y_pre_torch = RNN_model(u_torch)
    loss = criterion(y_pre_torch, y_ref_torch)
    loss_all[0:1, :] = loss.item()
    loss_m = loss.item()
    RNN_model_m = RNN_create(in_s, layer_s, hidden_s, device, bias_status)
    RNN_model_m.load_state_dict(RNN_model.state_dict())

    start = time.time()
    for i in range(N):
        RNN_model.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        y_pre_torch = RNN_model(u_torch)
        loss = criterion(y_pre_torch, y_ref_torch)
        i1 = i + 1
        loss_all[i1:i1 + 1, :] = loss.item()

        if loss.item() < loss_m:
            loss_m = loss.item()
            RNN_model_m.load_state_dict(RNN_model.state_dict())
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
    return RNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str