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
from DSNN_tools import DSNN_create, DSNN_cal, DSNN_training
from general_tools import time_str
os.chdir(current_path)
sys.path.append('.')
device = torch.device('cpu') # 'cuda:0' 'cpu'

name='BW'
# Prepare data
dt = 1/750
Nt1=3751
Nt2=8192
f = loadmat(name+'_data.mat')
u = f['u'].reshape([Nt2, 1, 1])
y_ref = f['y_ref'].reshape([Nt2, 1, 1])
tend = (Nt2 - 1) * dt
t = np.linspace(0, tend, Nt2).reshape(-1, 1)
del f, dt
bias_status=False

# Define the configuration of DSNN
in_s = 1
out_s = 1
state_non_layer_s=1
out_non_layer_s=1
state_non_neuron = np.zeros(state_non_layer_s, dtype=np.int32)  # size of each nonlinear layer
out_non_neuron = np.zeros(out_non_layer_s, dtype=np.int32)  # size of each nonlinear layer
out_non_neuron[0] = out_s  # user defined
# %%
# if you want to train models on your machine, you can run this section
# if you want to load trained models provided by us in the "saved_models" folder provided by us, you can skip this section
N = 20000 # training num 20000
All_start = time.time()
for i in range(1,21):
    state_s=i
    state_non_neuron[0] = state_s  # user defined
    DSNN_model = DSNN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, device, bias_status)
    # total_num = sum(p.numel() for p in DSNN_model.parameters())
    trainable_num = sum(p.numel() for p in DSNN_model.parameters() if p.requires_grad)

    DSNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str = DSNN_training(1, Nt1, u[0:Nt1,:,:], DSNN_model, N, y_ref[0:Nt1,:,:], in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, device, bias_status)
    torch.save(DSNN_model_m.state_dict(), './saved_models/'+name+'_DSNN_'+str(state_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.pt')
    savemat('./saved_data/'+name+'_DSNN_'+str(state_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.mat', {'loss_all': loss_all,
     'cost_time': cost_time, 'trainable_num': trainable_num})

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
state_s=16
state_non_neuron[0] = state_s  # user defined
DSNN_model = DSNN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_DSNN_'+str(state_s)+'_*.pt'))
DSNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu', weights_only=True))
u_torch = torch.tensor(u)
y_pre_torch = DSNN_cal(1, Nt2, u, DSNN_model, state_s, out_s, 'cpu')
y_pre = y_pre_torch.reshape([Nt2, 1]).detach().numpy()

plt.plot(t, y_ref.reshape([Nt2, 1]))
plt.plot(t, y_pre)
plt.show()
# %%
y_ref_torch = torch.tensor(y_ref)
criterion = nn.MSELoss()
loss_DSNN=np.zeros((20, 2))
for i in range(1,21):
    mat_match = glob.glob(os.path.join('./saved_data', '*_DSNN_'+str(i)+'_*.mat'))
    f = loadmat(mat_match[0])
    loss_all=f['loss_all']
    loss_DSNN[i-1:i, :1]=np.min(loss_all)
    pt_match=mat_match[0].replace('data', 'models').replace('mat', 'pt')
    state_s=i
    state_non_neuron[0] = state_s  # user defined
    DSNN_model = DSNN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, device, bias_status)
    DSNN_model.load_state_dict(torch.load(pt_match, map_location='cpu', weights_only=True))
    y_pre_torch = DSNN_cal(1, Nt2, u, DSNN_model, state_s, out_s, 'cpu')
    loss_DSNN[i-1:i, 1:2] = criterion(y_pre_torch[Nt1:Nt2,:,:], y_ref_torch[Nt1:Nt2,:,:]).item()

del mat_match, pt_match, f, loss_all
savemat(name+'_loss_DSNN.mat', {name+'_loss_DSNN': loss_DSNN})

# %%
trainable_num_DSNN=np.zeros((20, 1))
for i in range(1,21):
    mat_match = glob.glob(os.path.join('./saved_data', '*_DSNN_'+str(i)+'_*.mat'))
    f = loadmat(mat_match[0])
    trainable_num_DSNN[i-1:i, :1]=f['trainable_num']

del mat_match, f
savemat('trainable_num_DSNN.mat', {'trainable_num_DSNN': trainable_num_DSNN})