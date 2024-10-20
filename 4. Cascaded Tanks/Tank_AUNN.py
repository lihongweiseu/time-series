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
from AUNN_tools import AUNN_create, AUNN_cal, AUNN_training
from general_tools import time_str
os.chdir(current_path)
sys.path.append('.')
device = torch.device('cpu')# 'cuda:0' 'cpu'

name='Tank'
# Prepare data
dt = 4
Nt=1024
f = loadmat(name+'_data.mat')
u = np.transpose(f['u'])
y_ref = np.transpose(f['y_ref'])
tend = (Nt - 1) * dt
t = np.linspace(0, tend, Nt)
del f, dt

all_size=np.empty(shape=(0, 3))
for i in range(0,21):
    j=i-1
    k=i+j
    all_size=np.insert(np.array([[i, j, k]], dtype=np.int32), 0, all_size, axis=0)
N_size=np.size(all_size, 0)
bias_status=False
non_layer_s=1
# %%
# if you want to train models on your machine, you can run this section
# if you want to load trained models provided by us in the "saved_models" folder provided by us, you can skip this section
N = 20000 # training num 20000
All_start = time.time()
for i in range(1,21):
    x_s=all_size[i,0]
    y_s=all_size[i,1]
    in_s=x_s+y_s
    non_neuron = np.zeros(non_layer_s, dtype=np.int32)
    non_neuron[:]=all_size[i,2]
    AUNN_model = AUNN_create(in_s, non_layer_s, non_neuron, device, bias_status)
    # total_num = sum(p.numel() for p in AUNN_model.parameters())
    trainable_num = sum(p.numel() for p in AUNN_model.parameters() if p.requires_grad)

    AUNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str = AUNN_training(1, Nt, u[:1,:], AUNN_model, x_s, in_s, N, y_ref[:1,:], non_layer_s, non_neuron, device, bias_status)
    torch.save(AUNN_model_m.state_dict(), './saved_models/'+name+'_AUNN_'+str(x_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.pt')
    savemat('./saved_data/'+name+'_AUNN_'+str(x_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'.mat', {'loss_all': loss_all,
     'cost_time': cost_time, 'trainable_num': trainable_num})

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
x_s=18
in_s=x_s+x_s-1
non_layer_s=1
non_neuron = np.zeros(non_layer_s, dtype=np.int32)
non_neuron[:]=in_s
AUNN_model = AUNN_create(in_s, non_layer_s, non_neuron, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_AUNN_'+str(x_s)+'_*.pt'))
AUNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu', weights_only=True))
y_pre_torch = AUNN_cal(2, Nt, u, AUNN_model, x_s, in_s, 'cpu')
y_pre = y_pre_torch.detach().numpy()

i=2
plt.plot(t, y_ref[i-1, :])
plt.plot(t, y_pre[i-1, :])
plt.show()

# %%
y_ref_torch = torch.tensor(y_ref[1:2,:]).to('cpu')
criterion = nn.MSELoss()
loss_AUNN=np.zeros((20, 2))
for i in range(1,21):
    x_s=all_size[i,0]
    y_s=all_size[i,1]
    in_s=x_s+y_s
    non_neuron = np.zeros(non_layer_s, dtype=np.int32)
    non_neuron[:]=all_size[i,2]
    mat_match = glob.glob(os.path.join('./saved_data', '*_AUNN_'+str(i)+'_*.mat'))
    f = loadmat(mat_match[0])
    loss_all=f['loss_all']
    loss_AUNN[i-1:i, :1]=np.min(loss_all)
    pt_match=mat_match[0].replace('data', 'models').replace('mat', 'pt')
    hidden_s=i
    AUNN_model = AUNN_create(in_s, non_layer_s, non_neuron, 'cpu', bias_status)
    AUNN_model.load_state_dict(torch.load(pt_match, map_location='cpu', weights_only=True))
    y_pre_torch = AUNN_cal(1, Nt, u[1:2,:], AUNN_model, x_s, in_s, 'cpu')
    loss_AUNN[i-1:i, 1:2] = criterion(y_pre_torch, y_ref_torch).item()

del mat_match, pt_match, f, loss_all
savemat(name+'_loss_AUNN.mat', {name+'_loss_AUNN': loss_AUNN})