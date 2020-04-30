import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def generate_synt_data(n_total=100, n_times=9, n_freq=8,
                       ratio_train=0.8, ratio_exp=0.5, 
                       noise_scale=0.05):
    assert ratio_train <= 1 and ratio_train >= 0
    n_total = int(n_total)
    n_train = int(ratio_train * n_total)
    n_test = n_total - n_train
    assert n_train + n_test == n_total
    assert ratio_exp <=1 and ratio_exp >= 0
    ratio_unexp = 1 - ratio_exp
    ratio_exp, ratio_unexp = ratio_exp / (ratio_exp + ratio_unexp ),  ratio_unexp / (ratio_exp + ratio_unexp)
    assert ratio_exp + ratio_unexp == 1

    ## simple data generation for now:
    all_seq = np.zeros((n_total, n_times, n_freq))
    all_seq[:, 0, 0] = 1
    all_seq[:, 1, 1] = 1
    all_seq[:, 2, 0] = 1
    all_seq[:, 3, 2] = 1
    all_seq[:, 4, 0] = 1
    
    ## Train/test data:
    shuffle_ind = np.random.choice(a=n_total, size=n_total, replace=False)
    all_seq = all_seq[shuffle_ind, :, :]  # shuffle randomly, (TODO: Stratified split)
    train_seq = all_seq[:n_train, :, :]
    test_seq = all_seq[n_train:, :, :]
    x_train = train_seq[:, :-1, :] + (np.random.randn(n_train, n_times - 1, n_freq) * noise_scale)
    y_train = train_seq[:, 1:, :]
    x_test = test_seq[:, :-1, :] + (np.random.randn(n_test, n_times - 1, n_freq) * noise_scale)
    y_test = test_seq[:, 1:, :]
    x_train, y_train, x_test, y_test = map(
        torch.tensor, (x_train, y_train, x_test, y_test))
    x_train, y_train, x_test, y_test = x_train.float(), y_train.float(), x_test.float(), y_test.float()
    return (x_train, y_train, x_test, y_test)

class RNN(nn.Module):
    def __init__(self, n_stim, n_nodes):
        super().__init__()
        self.n_stim = n_stim
        self.n_nodes = n_nodes
        self.lin_input = nn.Linear(self.n_stim, self.n_nodes)
        self.lin_feedback = nn.Linear(self.n_nodes, self.n_nodes)
        self.lin_output = nn.Linear(self.n_nodes, self.n_stim)
        self.init_state()
        
    def init_state(self):
        self.state = torch.randn(self.n_nodes) * 0.1  # initialise s_{-1}
        
    def forward(self, inp, rnn_state=None):
        if rnn_state is None:
            rnn_state = self.state
        lin_comb = self.lin_input(inp) + self.lin_feedback(rnn_state)
        new_state = torch.tanh(lin_comb)
        self.state = new_state
        output = F.softmax(self.lin_output(new_state.squeeze()), dim=0)
        return new_state, output
    
def tau_loss(y_est, y_true, tau_array=np.array([2, 3]), 
             model=None, reg_param=0.001):
    y_est_trunc = y_est[:, tau_array, :]
    y_true_trunc = y_true[:, tau_array, :]
    n_samples = y_true.shape[0]
    # if n_samples != 1:
    #     print(f'n samples: {n_samples}')
    ce = torch.sum(-1 * y_true_trunc * torch.log(y_est_trunc)) / n_samples  # take the mean CE over samples 
    if model is not None:
        params = [pp for pp in model.parameters()]
        for _, p_set in enumerate(params):
            ce += reg_param * p_set.norm(p=1) 
    return ce
    
def compute_full_pred(xdata, ydata, model):
    full_pred = torch.zeros_like(ydata)
    # if xdata.shape[0] != 1:
    #     print(f'shape xdata: {xdata.shape}')
    for kk in range(xdata.shape[0]): # loop over trials 
        model.init_state()  # initiate rnn state per trial 
        for tt in range(xdata.shape[1]):  # loop through time
            _, full_pred[kk, tt, :] = model(xdata[kk, tt, :])  # compute prediction at this time 
    return full_pred