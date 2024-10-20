# %%
import matplotlib.pyplot as plt
import torch
from torch import nn
import time
import numpy as np
from scipy.io import savemat, loadmat
import os, sys
import glob
current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
os.chdir(parent_path)
sys.path.append('.')
from RNN_tools import RNN_create, RNN_training
from general_tools import time_str
os.chdir(current_path)
sys.path.append('.')
device = torch.device('cpu') # 'cuda:0' 'cpu'

name='MR'
# Prepare data
dt=0.005
Nt1=2001
Nt2=3499
f = loadmat(name+'_data.mat')
u = f['u']
y_ref = f['y_ref']
tend = (Nt2 - 1) * dt
t = np.linspace(0, tend, Nt2).reshape(-1, 1)
del f, dt
bias_status=False
layer_s=1
in_s=1
# %%
# if you want to train models on your machine, you can run this section
# if you want to load trained models provided by us in the "saved_models" folder provided by us, you can skip this section
N = 20000 # training num 20000
All_start = time.time()
for i in range(2,41,2):
    hidden_s=i
    RNN_model = RNN_create(in_s, layer_s, hidden_s, device, bias_status)
    # total_num = sum(p.numel() for p in RNN_model.parameters())
    trainable_num = sum(p.numel() for p in RNN_model.parameters() if p.requires_grad)

    RNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str = RNN_training(1, Nt1, u[:Nt1,:], RNN_model, in_s, layer_s, hidden_s, N, y_ref[:Nt1,:], device, bias_status)
    torch.save(RNN_model_m.state_dict(), './saved_models/'+name+'_RNN_'+str(hidden_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.pt')
    savemat('./saved_data/'+name+'_RNN_'+str(hidden_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.mat', {'loss_all': loss_all,
     'cost_time': cost_time, 'trainable_num': trainable_num})

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
hidden_s=20
RNN_model = RNN_create(in_s, layer_s, hidden_s, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_RNN_'+str(hidden_s)+'_*.pt'))
RNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu', weights_only=True))
u_torch = torch.tensor(u).reshape([Nt2, 1, in_s])
y_pre_torch = RNN_model(u_torch)
y_pre = y_pre_torch.reshape([Nt2, 1]).detach().numpy()

plt.plot(t, y_ref)
plt.plot(t, y_pre)
plt.show()
# %%
y_ref_torch = torch.tensor(y_ref).reshape([Nt2, 1, in_s])
criterion = nn.MSELoss()
loss_RNN=np.zeros((20, 2))
for j in range(2,41,2):
    i=int(j/2)
    mat_match = glob.glob(os.path.join('./saved_data', '*_RNN_'+str(j)+'_*.mat'))
    f = loadmat(mat_match[0])
    loss_all=f['loss_all']
    loss_RNN[i-1:i, :1]=np.min(loss_all)
    pt_match=mat_match[0].replace('data', 'models').replace('mat', 'pt')
    hidden_s=j
    RNN_model = RNN_create(in_s, layer_s, hidden_s, 'cpu', bias_status)
    RNN_model.load_state_dict(torch.load(pt_match, map_location='cpu', weights_only=True))
    u_torch = torch.tensor(u).reshape([Nt2, 1, in_s])
    y_pre_torch = RNN_model(u_torch)
    loss_RNN[i-1:i, 1:2] = criterion(y_pre_torch[Nt1:Nt2,:,:], y_ref_torch[Nt1:Nt2,:,:]).item()

del mat_match, pt_match, f, loss_all
savemat(name+'_loss_RNN.mat', {name+'_loss_RNN': loss_RNN})