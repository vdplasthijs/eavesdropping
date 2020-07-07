# @Author: Thijs L van der Plas <TL>
# @Date:   2020-05-14
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Filename: bptt_rnn.py
# @Last modified by:   thijs
# @Last modified time: 2020-05-25



import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle, datetime, time, os, sys, git
from tqdm import tqdm, trange
import sklearn.svm, sklearn.model_selection
import rot_utilities as ru

def generate_synt_data(n_total=100, n_times=9, n_freq=8,
                       ratio_train=0.8, ratio_exp=0.5,
                       noise_scale=0.05, double_length=False):
    '''Generate synthetic data, see notebook for description.'''
    assert ratio_train <= 1 and ratio_train >= 0
    n_total = int(n_total)
    n_half_total = int(np.round(n_total / 2))
    assert ratio_exp <=1 and ratio_exp >= 0
    ratio_unexp = 1 - ratio_exp
    ratio_exp, ratio_unexp = ratio_exp / (ratio_exp + ratio_unexp ),  ratio_unexp / (ratio_exp + ratio_unexp)
    assert ratio_exp + ratio_unexp == 1

    ## Create data sequences of 5, 7 or 9 elements
    ## 0-0   1-A1    2-A2    3-B1    4-B2    5-C1   6-C2    7-D
    all_seq = np.zeros((n_total, n_times, n_freq))
    labels = np.zeros(n_total, dtype='object')
    for t in [xx for xx in range(n_times) if xx % 2 == 0]:  # blanks at even positions
        all_seq[:, t, 0] = 1
    all_seq[:n_half_total, 1, 1] = 1  # A1
    all_seq[n_half_total:, 1, 2] = 1  # A2
    all_seq[:n_half_total, 3, 3] = 1  # B1
    all_seq[n_half_total:, 3, 4] = 1  # B2
    if n_times >= 7: # Add C
        if ratio_exp == 1:
            all_seq[:n_half_total, 5, 5] = 1  # C1
            labels[:n_half_total] = '11'
            all_seq[n_half_total:, 5, 6] = 1  # C2
            labels[n_half_total:] = '22'
        else:  # Expected & Unexpected cases for both 1 and 2 lines
            n_exp_half = int(np.round(ratio_exp * n_half_total))
            all_seq[:n_exp_half, 5, 5] = 1  # exp C1
            labels[:n_exp_half] = '11'
            all_seq[n_exp_half:n_half_total, 5, 6] = 1  #unexp C2
            labels[n_exp_half:n_half_total] = '12'
            all_seq[n_half_total:(n_half_total + n_exp_half), 5, 6] = 1 # exp C2
            labels[n_half_total:(n_half_total + n_exp_half)] = '22'
            all_seq[(n_half_total + n_exp_half):, 5, 5] = 1  # unexp C1
            labels[(n_half_total + n_exp_half):] = '21'
    if n_times == 9: # Add D
        all_seq[:, 7, 7] = 1

    if double_length:  # If True: double the sequence lengths by inserting a copy of each element in place
        new_all_seq = np.zeros((n_total, 2 * n_times, n_freq))  # new sequence
        for kk in range(n_times):
            new_all_seq[:, (2 * kk):(2 * (kk + 1)), :] = all_seq[:, kk, :][:, np.newaxis, :]  # create 2 copies
        all_seq = new_all_seq  # rename
        n_times = all_seq.shape[1]  # redefine

    n_train = int(ratio_train * n_total)
    n_test = n_total - n_train
    assert n_train + n_test == n_total

    ## Train/test data:
    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, train_size=ratio_train).split(X=np.zeros_like(labels),
                                                                                 y=labels) # stratified split
    train_inds, test_inds = next(sss)  # generate
    train_seq = all_seq[train_inds, :, :]
    labels_train = labels[train_inds]
    test_seq = all_seq[test_inds, :, :]
    labels_test = labels[test_inds]
    x_train = train_seq[:, :-1, :] + (np.random.randn(n_train, n_times - 1, n_freq) * noise_scale)  # add noise to input
    y_train = train_seq[:, 1:, :]  # do not add noise to output
    x_test = test_seq[:, :-1, :] + (np.random.randn(n_test, n_times - 1, n_freq) * noise_scale)
    y_test = test_seq[:, 1:, :]
    x_train, y_train, x_test, y_test = map(
        torch.tensor, (x_train, y_train, x_test, y_test))  # create tensors
    x_train, y_train, x_test, y_test = x_train.float(), y_train.float(), x_test.float(), y_test.float()  # need to be float type (instead of 'double', which is somewhat silly)
    return (x_train, y_train, x_test, y_test), (labels_train, labels_test)

class RNN(nn.Module):
    def __init__(self, n_stim, n_nodes, init_std_scale=0.1):
        '''RNN Model with input/hidden/output layers. Fully connected. '''
        super().__init__()

        ## Model parameters
        self.n_stim = n_stim
        self.n_nodes = n_nodes
        self.init_std_scale = init_std_scale
        self.info_dict = {'converged': False}  # any info can be saved later

        ## Linear combination layers:
        self.lin_input = nn.Linear(self.n_stim, self.n_nodes)
        self.lin_feedback = nn.Linear(self.n_nodes, self.n_nodes)
        self.lin_output = nn.Linear(self.n_nodes, self.n_stim)
        self.init_state()  # initialise RNN nodes

        ## Attributes to be completed later:
        self.train_loss_arr = []  # to be appended during training
        self.test_loss_arr = []
        self.test_loss_ratio_ce = []
        self.test_loss_split = {'B': [], 'C': [], 'C1': [], 'C2': [], 'D': [], '0': [], 'L1': [],
                                '0_postA': [], '0_postB': [], '0_postC': [], '0_postD': []}
        self.decoding_crosstemp_score = {}
        self.decoder_dict = {}

        ## Automatic:
        self.__datetime_created = datetime.datetime.now()
        self.__version__ = '0.1'
        self.__git_repo__ = git.Repo(search_parent_directories=True)
        self.__git_branch__ = self.__git_repo__.head.reference.name
        self.__git_commit__ = self.__git_repo__.head.object.hexsha
        self.file_name = None
        self.full_path = None
        self.rnn_name = 'RNN (not saved)'

    def __str__(self):
        """Define name"""
        return self.rnn_name

    def __repr__(self):
        """Define representation"""
        return f'Instance {self.rnn_name} of RNN Class'

    def init_state(self):
        '''Initialise hidden state to random values N(0, 0.1)'''
        self.state = torch.randn(self.n_nodes) * self.init_std_scale  # initialise s_{-1}

    def forward(self, inp, rnn_state=None):
        '''Perform one forward step given input and hidden state. If hidden state
        (rnn_state) is None, self.state will be used (regular behaviour).'''
        if rnn_state is None:
            rnn_state = self.state
        lin_comb = self.lin_input(inp) + self.lin_feedback(rnn_state)  # input + previous state
        new_state = torch.tanh(lin_comb)  # transfer function
        self.state = new_state
        output = F.softmax(self.lin_output(new_state.squeeze()), dim=0)  # output nonlin-lin
        return new_state, output

    def set_info(self, param_dict):
        '''Add information to the info dictionary. The param_dict is copied into
        info_dict (including overwriting).'''
        for key, val in param_dict.items():
            self.info_dict[key] = val  # overwrites

    def save_model(self, folder=None, verbose=True, add_nnodes=False):
        '''Export this RNN model to folder. If self.file_name is None, it is saved  under
        a timestamp.'''
        dt = datetime.datetime.now()
        timestamp = str(dt.date()) + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)
        self.info_dict['timestamp'] = timestamp
        if self.file_name is None:
            if add_nnodes is False:
                self.rnn_name = f'rnn_{timestamp}'
                self.file_name = f'rnn_{timestamp}.data'
            else:
                self.rnn_name = f'rnn_n{self.info_dict["n_nodes"]}_{timestamp}'
                self.file_name = f'rnn_n{self.info_dict["n_nodes"]}_{timestamp}.data'
        if folder is None:
            folder = 'models/'
        elif folder[-1] != '/':
            folder += '/'
        self.full_path = folder + self.file_name
        file_handle = open(self.full_path, 'wb')
        pickle.dump(self, file_handle)
        if verbose > 0:
            print(f'RNN model saved as {self.file_name}')

class RNN_MNM(RNN):
    def __init__(self, n_stim, n_nodes, init_std_scale=0.1, accumulate=False):
        self.accumulate = accumulate
        super().__init__(n_stim=n_stim, n_nodes=n_nodes, init_std_scale=init_std_scale)  # init like normal RNN
        self.lin_output = nn.Linear(self.n_nodes, self.n_stim + 2)  # override output
        self.rnn_name = 'RNN-MNM (not saved)'
        self.test_loss_split['MNM'] = []

    def init_state(self):
        '''Initialise hidden state to random values N(0, 0.1)'''
        self.state = torch.randn(self.n_nodes) * self.init_std_scale  # initialise s_{-1}
        if self.accumulate:
            self.history_mnm = torch.zeros(2)

    def forward(self, inp, rnn_state=None):
        '''Perform one forward step given input and hidden state. If hidden state
        (rnn_state) is None, self.state will be used (regular behaviour).'''
        if rnn_state is None:
            rnn_state = self.state
        lin_comb = self.lin_input(inp) + self.lin_feedback(rnn_state)  # input + previous state
        new_state = torch.tanh(lin_comb)  # transfer function
        self.state = new_state

        ## MNM specific output:
        linear_output = self.lin_output(new_state.squeeze())
        output = torch.zeros_like(linear_output)  # we will normalise the prediction task & MNM separately:
        output[:self.n_stim] = F.softmax(linear_output[:self.n_stim], dim=0)  # output nonlin-lin of the prediction task (normalised on these only )
        if self.accumulate is False:
            output[self.n_stim:] = F.softmax(linear_output[self.n_stim:], dim=0)  # probabilities units for M and NM (normalised)
        elif self.accumulate:
            new_hist = F.relu(linear_output[self.n_stim:]) + self.history_mnm
            output[self.n_stim:] = F.softmax(new_hist, dim=0) # accumulate signal
            # output[self.n_stim:] = 0.5 * (torch.tanh(new_hist) + 1)# accumulate signal
            self.history_mnm = new_hist  # save for next iter
        return new_state, output

    def save_model(self, folder=None, verbose=True, add_nnodes=False):  # redefine because we want to change saving name
        '''Export this RNN model to folder. If self.file_name is None, it is saved  under
        a timestamp.'''
        dt = datetime.datetime.now()
        timestamp = str(dt.date()) + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)
        self.info_dict['timestamp'] = timestamp
        if self.file_name is None:
            if add_nnodes is False:
                self.rnn_name = f'rnn-mnm_{timestamp}'
                self.file_name = f'rnn-mnm_{timestamp}.data'
            else:
                self.rnn_name = f'rnn-mnm_n{self.info_dict["n_nodes"]}_{timestamp}'
                self.file_name = f'rnn-mnm_n{self.info_dict["n_nodes"]}_{timestamp}.data'
        if folder is None:
            folder = 'models/'
        elif folder[-1] != '/':
            folder += '/'
        self.full_path = folder + self.file_name
        file_handle = open(self.full_path, 'wb')
        pickle.dump(self, file_handle)
        if verbose > 0:
            print(f'RNN-MNM model saved as {self.file_name}')

def tau_loss(y_est, y_true, tau_array=np.array([2, 3]), label=None, match_times=[13, 14],
             model=None, reg_param=0.001):
    '''Compute Cross Entropy of given time array tau_array, and add L1 regularisation.'''
    y_est_trunc = y_est[:, tau_array, :model.n_stim]  # only evaluated these time points, cut off at N_stim, because for M and NM these follow after
    y_true_trunc = y_true[:, tau_array, :]
    n_samples = y_true.shape[0]
    ce = torch.sum(-1 * y_true_trunc * torch.log(y_est_trunc)) / n_samples  # take the mean CE over samples

    reg_loss = 0
    if model is not None:  # add L1 regularisation
        params = [pp for pp in model.parameters()]  # for all weight (matrices) in the model
        for _, p_set in enumerate(params):
            reg_loss += reg_param * p_set.norm(p=1)

    if model.lin_output.out_features > model.n_stim: # M & NM present
        assert match_times is not None, 'no match times defined '
        assert label is not None, 'no labels defined'
        match_arr = ru.labels_to_mnm(labels=label)
        match_est = y_est[:, match_times, model.n_stim:]
        match_arr_full = torch.zeros_like(match_est)
        for tt in range(len(match_times)):
            match_arr_full[:, tt, :] = torch.tensor(match_arr)
        ce_match = torch.sum(-1 * match_arr_full * torch.log(match_est)) / n_samples  # take the mean CE over samples
    else:
        ce_match = 0

    total_loss = ce + reg_loss + ce_match
    return total_loss, (ce, reg_loss, ce_match)

def split_loss(y_est, y_true, tau_array=np.array([2, 3]), label=None, match_times=[13, 14],
               time_prediction_array_dict={'B': [5, 6], 'C': [9, 10], 'C1': [9], 'C2': [10], 'D': [13, 14],
                                           '0': [4, 7, 8, 11, 12, 15, 16], '0_postA': [4],
                                           '0_postB': [7, 8], '0_postC': [11, 12], '0_postD': [15, 16]},
               model=None, reg_param=0.001, return_ratio_ce=False):
    '''Compute Cross Entropy for each given time array, and L1 regularisation.'''
    assert model is not None
    for key, tau_array in time_prediction_array_dict.items():  # compute separate times separately
        y_est_trunc = y_est[:, tau_array, :model.n_stim]  # only evaluated these time points
        y_true_trunc = y_true[:, tau_array, :]
        n_samples = y_true.shape[0]
        ce = torch.sum(-1 * y_true_trunc * torch.log(y_est_trunc)) / n_samples  # take the mean CE over samples
        model.test_loss_split[key].append(float(ce.detach().numpy()))  # add to model

    total_loss, (ce, reg_loss, ce_match) = tau_loss(y_est=y_est, y_true=y_true, model=model, label=label,
                                                    reg_param=reg_param, match_times=match_times,
                                                    tau_array=tau_array)  # compute three loss terms

    model.test_loss_split['L1'].append(float(reg_loss.detach().numpy()))  # add to array
    model.test_loss_split['MNM'].append(float(ce_match.detach().numpy()))

    if return_ratio_ce is False:
        return total_loss
    elif return_ratio_ce:
        return (total_loss, (ce / total_loss))

def compute_full_pred(xdata, model, mnm=False):
    '''Compute forward prediction of RNN. I.e. given an input series xdata, the
    model (RNN) computes the predicted output series.'''
    if xdata.ndim == 2:
        xdata = xdata[None, :, :]
    mnm = model.lin_output.out_features > model.n_stim  # determine if MNM model
    if mnm is False:
        full_pred = torch.zeros_like(xdata)  # because input & ouput have the same shape
    elif mnm:
        assert xdata.shape[2] == 8  # stim vector
        full_pred = torch.zeros((xdata.shape[0], xdata.shape[1], xdata.shape[2] + 2))  # make space for M and NM elements
    for kk in range(xdata.shape[0]): # loop over trials
        model.init_state()  # initiate rnn state per trial
        for tt in range(xdata.shape[1]):  # loop through time
            _, full_pred[kk, tt, :] = model(xdata[kk, tt, :])  # compute prediction at this time
    return full_pred

def bptt_training(rnn, optimiser, dict_training_params,
                  x_train, x_test, y_train, y_test, labels_train=None, labels_test=None, verbose=1):
    '''Training algorithm for backpropagation through time, given a RNN model, optimiser,
    dictionary with training parameters and train and test data. RNN is NOT reset,
    so continuation training is possible. Training can be aborted prematurely by Ctrl+C,
    and it will terminate correctly.'''
    assert dict_training_params['bs'] == 1, 'batch size is not 1; this error is thrown because for MNM we assume it is 1 to let labels correspond to dataloadre loop'
    ## Create data loader objects:
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=dict_training_params['bs'])

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=dict_training_params['bs'])

    if labels_train is not None and labels_test is not None:
        assert labels_train.ndim == 1 and labels_test.ndim == 1
        assert len(labels_train) == x_train.shape[0] and len(labels_test) == x_test.shape[0]
    else:  # make arrays with None for compatiblity with loop
        labels_train = [None] * x_train.shape[0]
        labels_test = [None] * x_test.shape[0]

    prev_loss = 10  # init loss for convergence
    if 'trained_epochs' not in rnn.info_dict.keys():
        rnn.info_dict['trained_epochs'] = 0

    ## Training procedure
    init_str = f'Initialising training; start at epoch {rnn.info_dict["trained_epochs"]}'
    try:
        with trange(dict_training_params['n_epochs']) as tr:  # repeating epochs
            for epoch in tr:
                if epoch == 0:
                    tr.set_description(init_str)
                else:
                    update_str = f'Epoch {epoch}/{dict_training_params["n_epochs"]}. Train loss: {np.round(prev_loss, 6)}'
                    tr.set_description(update_str)
                rnn.train()  # set to train model (i.e. allow gradient computation/tracking)
                it_train = 0
                for xb, yb in train_dl:  # returns torch(n_bs x n_times x n_freq)
                    curr_label = labels_train[it_train]  # this works if batch size == 1
                    full_pred = compute_full_pred(model=rnn, xdata=xb)  # predict time trace
                    loss, _ = tau_loss(y_est=full_pred, y_true=yb, model=rnn,
                                    reg_param=dict_training_params['l1_param'], match_times=[13, 14],
                                    tau_array=dict_training_params['eval_times'], label=curr_label)  # compute loss
                    loss.backward()  # compute gradients
                    optimiser.step()  # update
                    optimiser.zero_grad()   # reset
                    it_train += 1

                rnn.eval()  # evaluation mode -> disable gradient tracking
                with torch.no_grad():  # to be sure
                    ## Compute losses for saving:
                    full_pred = compute_full_pred(model=rnn, xdata=x_train)
                    train_loss, _ = tau_loss(y_est=full_pred, y_true=y_train, model=rnn,
                                          reg_param=dict_training_params['l1_param'], match_times=[13, 14],
                                          tau_array=dict_training_params['eval_times'], label=labels_train)
                    rnn.train_loss_arr.append(float(train_loss.detach().numpy()))

                    full_test_pred = compute_full_pred(model=rnn, xdata=x_test)
                    test_loss, ratio = split_loss(y_est=full_test_pred, y_true=y_test, model=rnn,
                                                  reg_param=dict_training_params['l1_param'],
                                                  tau_array=dict_training_params['eval_times'],
                                                  return_ratio_ce=True, match_times=[13, 14], label=labels_test)
                    rnn.test_loss_arr.append(float(test_loss.detach().numpy()))
                    rnn.test_loss_ratio_ce.append(float(ratio.detach().numpy()))

                    ## Inspect training loss for convergence
                    new_loss = rnn.train_loss_arr[epoch]
                    diff = np.abs(new_loss - prev_loss) / (new_loss + prev_loss)
                    if dict_training_params['check_conv']:
                        if diff < dict_training_params['conv_rel_tol']:
                            rnn.info_dict['converged'] = True
                            print(f'Converged at epoch {epoch},  loss: {new_loss}')
                            break  # end training
                    prev_loss = new_loss  # update current loss
                    rnn.info_dict['trained_epochs'] += 1  # add to grand total

        ## Set to evaluate mode and cutoff for early termination
        rnn.eval()
        if verbose > 0:
            print('Training finished. Results saved in RNN Class')
        return rnn
    except KeyboardInterrupt: # end prematurely by Ctrl+C
        rnn.eval()
        if verbose > 0:
            print(f'Training ended prematurely by user at epoch {epoch}.\nResults saved in RNN Class.')
        return rnn

def train_decoder(rnn_model, x_train, x_test, labels_train, labels_test,
                  save_inplace=True, label_name='alpha', sparsity_c=1e-1):
    n_nodes = rnn_model.info_dict['n_nodes']
    forw_mat = {'train': np.zeros((x_train.shape[0], x_train.shape[1], n_nodes)),  # trials x time x neurons
                'test': np.zeros((x_test.shape[0], x_test.shape[1], n_nodes))}

    rnn_model.eval()
    with torch.no_grad():
        n_times = x_train.shape[1]

        ## Forward runs:
        for kk in range(x_train.shape[0]):  # trial loop
            rnn_model.init_state()
            hidden_state = rnn_model.state  # init state
            for tau in range(n_times):  # time loop
                hidden_state, output = rnn_model.forward(inp=x_train[kk, tau, :],
                                                   rnn_state=hidden_state)  # propagate
                forw_mat['train'][kk, tau, :] = hidden_state.numpy()  # save hidden states

        for kk in range(x_test.shape[0]):  # trial loop
            rnn_model.init_state()
            hidden_state = rnn_model.state
            for tau in range(n_times):  # time loop
                hidden_state, output = rnn_model.forward(inp=x_test[kk, tau, :],
                                                   rnn_state=hidden_state)
                forw_mat['test'][kk, tau, :] = hidden_state.numpy()

        ## Train decoder
        alpha_labels = {'train': np.array([int(x[0]) for x in labels_train]),
                        'test': np.array([int(x[0]) for x in labels_test])}
        beta_labels = {'train': np.array([int(x[1]) for x in labels_train]),
                       'test': np.array([int(x[1]) for x in labels_test])}
        if label_name == 'alpha':
            labels_use = alpha_labels
        elif label_name == 'beta':
            labels_use = beta_labels
        else:
            print(f'Label name {label_name} not implemented. Please choose alpha or beta. Aborting')
            return None
        score_mat = np.zeros((n_times, n_times))  # T x T
        decoder_dict = {}  # save decoder per time
        tmp_var = True
        for tau in range(n_times):  # train time loop
            # decoder_dict[tau] = sklearn.svm.LinearSVC(C=sparsity_c)  # define SVM
            decoder_dict[tau] = sklearn.linear_model.LogisticRegression(C=sparsity_c,
                                                    solver='saga', penalty='l1', max_iter=250)  # define log reg
            decoder_dict[tau].fit(X=forw_mat['train'][:, tau, :],
                                  y=labels_use['train'])  # train SVM
            for tt in range(n_times):  # test time loop
                # score_mat[tau, tt] = decoder_dict[tau].score(X=forw_mat['test'][:, tt, :],
                #                                              y=labels_use['test'])  # evaluate
                prediction = decoder_dict[tau].predict_proba(X=forw_mat['test'][:, tt, :])
                inds_labels = np.zeros_like(labels_use['test'])  # zero = class 0
                inds_labels[(labels_use['test'] == decoder_dict[tau].classes_[1])] = 1
                prob_correct = np.array([prediction[i_pred, ind] for i_pred, ind in enumerate(inds_labels)])
                score_mat[tau, tt] = np.mean(prob_correct)
                # score_mat[tau, tt] = np.exp(np.mean(np.log(prob_correct)))
    if save_inplace:
        rnn_model.decoding_crosstemp_score[label_name] = score_mat
        rnn_model.decoder_dict[label_name] = decoder_dict
    return score_mat, decoder_dict, forw_mat

def init_train_save_rnn(t_dict, d_dict, n_simulations=1, save_folder='models/'):
    try:
        for nn in range(n_simulations):
            print(f'\n-----------\nsimulation {nn}/{n_simulations}')
            ## Generate data:
            tmp0, tmp1 = generate_synt_data(n_total=d_dict['n_total'],
                                       n_times=d_dict['n_times'],
                                       n_freq=d_dict['n_freq'],
                                       ratio_train=d_dict['ratio_train'],
                                       ratio_exp=d_dict['ratio_exp'],
                                       noise_scale=d_dict['noise_scale'],
                                       double_length=d_dict['doublesse'])
            x_train, y_train, x_test, y_test = tmp0
            labels_train, labels_test = tmp1

            ## Initiate RNN model
            # rnn = RNN(n_stim=d_dict['n_freq'], n_nodes=t_dict['n_nodes'])  # Create RNN class
            rnn = RNN_MNM(n_stim=d_dict['n_freq'], n_nodes=t_dict['n_nodes'], accumulate=True)  # Create RNN class
            opt = torch.optim.SGD(rnn.parameters(), lr=t_dict['learning_rate'])  # call optimiser from pytorhc
            rnn.set_info(param_dict={**d_dict, **t_dict})

            ## Train with BPTT
            rnn = bptt_training(rnn=rnn, optimiser=opt, dict_training_params=t_dict,
                                x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                labels_train=labels_train, labels_test=labels_test, verbose=0)

            ## Decode cross temporally
            score_mat, decoder_dict, _ = train_single_decoder_new_data(rnn=rnn, ratio_expected=0.5,
                                                            n_samples=None, ratio_train=0.8, verbose=False)

            ## Save results:
            rnn.save_model(folder=save_folder)
        return rnn  # return latest
    except KeyboardInterrupt:
        print('KeyboardInterrupt, exit')

def train_single_decoder_new_data(rnn, ratio_expected=0.5, label='alpha',
                                  n_samples=None, ratio_train=0.8, verbose=False,
                                  sparsity_c=0.1):
    '''Generates new data, and then trains the decoder via train_decoder()'''
    if n_samples is None:
        n_samples = rnn.info_dict['n_total']

    ## Generate data:
    tmp0, tmp1 = generate_synt_data(n_total=n_samples,
                                   n_times=rnn.info_dict['n_times'],
                                   n_freq=rnn.info_dict['n_freq'],
                                   ratio_train=ratio_train,
                                   ratio_exp=ratio_expected,
                                   noise_scale=rnn.info_dict['noise_scale'],
                                   double_length=rnn.info_dict['doublesse'])
    x_train, y_train, x_test, y_test = tmp0
    labels_train, labels_test = tmp1
    if verbose > 0:
        print('train labels ', {x: np.sum(labels_train == x) for x in np.unique(labels_train)})
    ## Train decoder:
    score_mat, decoder_dict, forward_mat = train_decoder(rnn_model=rnn, x_train=x_train, x_test=x_test,
                                           labels_train=labels_train, labels_test=labels_test,
                                           save_inplace=True, sparsity_c=sparsity_c, label_name=label)
    forward_mat['labels_train'] = labels_train
    forward_mat['labels_test'] = labels_test
    return score_mat, decoder_dict, forward_mat

def train_multiple_decoders(rnn_folder='models/', ratio_expected=0.5,
                            n_samples=None, ratio_train=0.8, label='alpha', reset_decoders=False):
    '''train decoders for all RNNs in rnn_folder'''
    rnn_list = [x for x in os.listdir(rnn_folder) if x[-5:] == '.data']
    for i_rnn, rnn_name in tqdm(enumerate(rnn_list)):
        ## Load RNN:
        with open(rnn_folder + rnn_name, 'rb') as f:
            rnn = pickle.load(f)
        if reset_decoders:
            rnn.decoding_crosstemp_score = {}
            rnn.decoder_dict = {}
        _ = train_single_decoder_new_data(rnn=rnn, ratio_expected=ratio_expected,
                                          n_samples=n_samples, ratio_train=ratio_train,
                                          verbose=(i_rnn == 0), label=label) # results are saved in RNN class
        rnn.save_model(folder=rnn_folder, verbose=0)  # save results to file
    return None

def aggregate_convergence(model_folder='models/', check_info_dict=True):
    '''Aggregate all score matrices for all rnns saved in folder'''
    if check_info_dict:
        check_params = ['n_total', 'n_freq', 'n_times', 'ratio_train', 'ratio_exp',
                        'noise_scale', 'n_nodes', 'doublesse']
    if model_folder[-1] != '/':
        model_folder += '/'
    list_models = [x for x in os.listdir(model_folder) if x[-5:] == '.data']
    list_loss = {'train': [], 'test': []}
    arr_loss = {}
    for ii, mn in enumerate(list_models):
        with open(model_folder + mn, 'rb') as f:
            model = pickle.load(f)
        if ii > 0  and check_info_dict: # check if dicts with info are equal
            for cp in check_params:
                assert (model.info_dict[cp] == prev_model.info_dict[cp]), f'AssertionError: models {prev_name} and {mn} are different'
            assert (model.info_dict['eval_times'] == prev_model.info_dict['eval_times']).all(), f'AssertionError: models {prev_name} and {mn} are different'

        list_loss['train'].append(model.train_loss_arr)  # append training loss array
        list_loss['test'].append(model.test_loss_arr)
        prev_model = model
        prev_name = mn
        f.close()
    for tt in ['train', 'test']:
        len_loss = []
        for tl in list_loss[tt]:  # for loss array in list:
            len_loss.append(len(tl))  # append length
        len_loss = np.array(len_loss)  # so we can determine the max
        arr_loss[tt] = np.zeros((len(list_loss[tt]), len_loss.max())) + np.nan  # init matrix of (n_models x max_convergence_lenght) in nans
        for i_tl, tl in enumerate(list_loss[tt]):
            arr_loss[tt][i_tl, :len(tl)] = tl  # fill in convergence arrays. nans remain after termination
    return arr_loss

def aggregate_score_mats(model_folder='models/', check_info_dict=True, label='alpha'):
    '''Aggregate all score matrices for all rnns saved in folder'''
    if check_info_dict:
        check_params = ['n_total', 'n_freq', 'n_times', 'ratio_train', 'ratio_exp',
                        'noise_scale', 'n_nodes', 'doublesse']
    if model_folder[-1] != '/':
        model_folder += '/'
    list_models = [x for x in os.listdir(model_folder) if x[-5:] == '.data']
    for ii, mn in enumerate(list_models):
        with open(model_folder + mn, 'rb') as f:
            model = pickle.load(f)
        if ii == 0:  # use first to create agrr matrix
            mat_shape = model.decoding_crosstemp_score[label].shape
            assert len(mat_shape) == 2  # 2D matrix
            agg_score_mat = np.zeros((len(list_models), mat_shape[0], mat_shape[1]))
        else:  #TODO: fix check
            if check_info_dict: # check if dicts with info are equal
                for cp in check_params:
                    assert (model.info_dict[cp] == prev_model.info_dict[cp]), f'AssertionError: models {prev_name} and {mn} are different'
                assert (model.info_dict['eval_times'] == prev_model.info_dict['eval_times']).all(), f'AssertionError: models {prev_name} and {mn} are different'

        agg_score_mat[ii, :, :] = model.decoding_crosstemp_score[label]  # add score matrix
        prev_model = model
        prev_name = mn
        f.close()
    return agg_score_mat


def aggregate_decoders(model_folder='models/', check_info_dict=True, label='alpha'):
    '''Aggregate all score matrices for all rnns saved in folder'''
    if check_info_dict:
        check_params = ['n_total', 'n_freq', 'n_times', 'ratio_train', 'ratio_exp',
                        'noise_scale', 'n_nodes', 'doublesse']
    if model_folder[-1] != '/':
        model_folder += '/'
    list_models = [x for x in os.listdir(model_folder) if x[-5:] == '.data']
    for ii, mn in enumerate(list_models):
        with open(model_folder + mn, 'rb') as f:
            model = pickle.load(f)
        if ii == 0:  # use first to create agrr matrix
            n_decoders = len(model.decoder_dict[label])
            n_nodes = len(model.decoder_dict[label][0].coef_[0])
            assert n_nodes == model.info_dict['n_nodes'], f'n nodes error: coef: {n_nodes}, info dict: {model.info_dict["n_nodes"]}'
            decoder_mat = np.zeros((len(list_models), n_nodes, n_decoders))
        else:  #TODO: fix check
            if check_info_dict: # check if dicts with info are equal
                for cp in check_params:
                    assert (model.info_dict[cp] == prev_model.info_dict[cp]), f'AssertionError: models {prev_name} and {mn} are different'
                assert (model.info_dict['eval_times'] == prev_model.info_dict['eval_times']).all(), f'AssertionError: models {prev_name} and {mn} are different'
        for i_dec, dec in model.decoder_dict[label].items():
            decoder_mat[ii, :, i_dec] = dec.coef_[0] # add score matrix
        prev_model = model
        prev_name = mn
        f.close()
    return decoder_mat

def aggregate_weights(model_folder='models/', weight='U', check_info_dict=True, label='alpha'):
        '''Aggregate all score matrices for all rnns saved in folder'''
        if check_info_dict:
            check_params = ['n_total', 'n_freq', 'n_times', 'ratio_train', 'ratio_exp',
                            'noise_scale', 'n_nodes', 'doublesse']
        if model_folder[-1] != '/':
            model_folder += '/'
        list_models = [x for x in os.listdir(model_folder) if x[-5:] == '.data']
        for ii, mn in enumerate(list_models):
            with open(model_folder + mn, 'rb') as f:
                model = pickle.load(f)
            if ii == 0:  # use first to create agrr matrix
                # n_decoders = len(model.decoder_dict[label])
                # n_nodes = len(model.decoder_dict[label][0].coef_[0])
                # assert n_nodes == model.info_dict['n_nodes'], f'n nodes error: coef: {n_nodes}, info dict: {model.info_dict["n_nodes"]}'
                weight_mat = np.zeros((len(list_models), 8))
            else:  #TODO: fix check
                if check_info_dict: # check if dicts with info are equal
                    for cp in check_params:
                        assert (model.info_dict[cp] == prev_model.info_dict[cp]), f'AssertionError: models {prev_name} and {mn} are different'
                    assert (model.info_dict['eval_times'] == prev_model.info_dict['eval_times']).all(), f'AssertionError: models {prev_name} and {mn} are different'
            for i_dec, dec in model.decoder_dict[label].items():
                input_mat = next(model.lin_input.parameters()).detach().numpy()
                weight_mat[ii, :] = np.mean(np.abs(input_mat), 0)# add score matrix
            prev_model = model
            prev_name = mn
            f.close()
        return weight_mat
