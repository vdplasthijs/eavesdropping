# @Author: Thijs L van der Plas <thijs>
# @Date:   2021-04-13
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Filename: bptt_rnn_mtl.py
# @Last modified by:   thijs
# @Last modified time: 2021-04-13




import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle, datetime, time, os, sys, git
from tqdm import tqdm, trange
import sklearn.svm, sklearn.model_selection, sklearn.discriminant_analysis
# import rot_utilities as ru


def generate_synt_data_general(n_total=100, t_delay=2, t_stim=2, 
                               ratio_train=0.8, ratio_exp=0.75,
                               noise_scale=0.05, late_s2=False,
                               nature_stim='onehot', task='dmc'):
    '''Generate synthetic data
    
    nature_stim: onehot, periodic, tuning
    task: dms, dmc, dmrs, dmrc, discr'''
    assert late_s2 is False, 'Late beta not implemented'
    assert ratio_train <= 1 and ratio_train >= 0
    pd = {}  #parameter dictionariy
    pd['n_total'] = int(n_total)
    pd['n_half_total'] = int(np.round(pd['n_total'] / 2))
    assert ratio_exp <=1 and ratio_exp >= 0
    pd['ratio_unexp'] = 1 - ratio_exp
    pd['ratio_exp'], pd['ratio_unexp'] = ratio_exp / (ratio_exp + pd['ratio_unexp'] ),  pd['ratio_unexp'] / (ratio_exp + pd['ratio_unexp'])
    assert pd['ratio_exp'] + pd['ratio_unexp'] == 1
    pd['n_exp_half'] = int(np.round(pd['ratio_exp'] * pd['n_half_total']))
    
    pd['n_input'] = 6 
    pd['n_times'] = int(4 * t_delay + 3 * t_stim)
    pd['period'] = int(t_delay + t_stim)
    pd['t_delay'] = t_delay 
    pd['t_stim'] = t_stim
    pd['slice_s1'] = slice(pd['t_delay'], (pd['t_delay'] + pd['t_stim']))
    pd['slice_s2'] = slice((2 * pd['t_delay'] + pd['t_stim']), (2 * pd['t_delay'] + 2 * pd['t_stim']))
    ## Create data sequences of 5, 7 or 9 elements
    ## 0-0   1-A1    2-A2    3-B1    4-B2    5-G
    all_seq = np.zeros((pd['n_total'], pd['n_times'], pd['n_input']))
    labels = np.zeros(pd['n_total'], dtype='object')
    for i_delay in range(4):  # 4 delay periods
        all_seq[:, :, 0][:, (i_delay * pd['period']):(i_delay * pd['period'] + t_delay)] = 1
    all_seq[:, :, 5][:, (3 * t_delay + 2 * t_stim):(3 * t_delay + 3 * t_stim)] = 1  # Go cue
    ## First fill in sequence of trials, shuffle later 
    if nature_stim == 'onehot':
        all_seq, labels = fill_onehot_trials(all_seq=all_seq, labels=labels, task=task, pd=pd)
    elif nature_stim == 'periodic':
        pass
    elif nature_stim == 'tuning':
        pass
    else:
        print(f'{nature_stim} not recognised - exiting')
        return None
    
    n_train = int(ratio_train * pd['n_total'])
    n_test = pd['n_total'] - n_train
    assert n_train + n_test == pd['n_total']
    
    ## Train/test data:
    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, train_size=ratio_train).split(X=np.zeros_like(labels),
                                                                                 y=labels) # stratified split
    train_inds, test_inds = next(sss)  # generate
    train_seq = all_seq[train_inds, :, :]
    labels_train = labels[train_inds]
    test_seq = all_seq[test_inds, :, :]
    labels_test = labels[test_inds]
    
    ## 
    x_train = train_seq[:, :-1, :] + (np.random.randn(n_train, pd['n_times'] - 1, pd['n_input']) * noise_scale)  # add noise to input
    x_test = test_seq[:, :-1, :] + (np.random.randn(n_test, pd['n_times'] - 1, pd['n_input']) * noise_scale)
    y_train_pred = train_seq[:, 1:, :]  # do not add noise to output
    y_test_pred = test_seq[:, 1:, :]
    
    y_train = np.zeros((y_train_pred.shape[0], y_train_pred.shape[1], y_train_pred.shape[2] + 2))
    y_test = np.zeros((y_test_pred.shape[0], y_test_pred.shape[1], y_test_pred.shape[2] + 2))
    y_train[:, :, :pd['n_input']] = y_train_pred  # prediction task target
    y_test[:, :, :pd['n_input']] = y_test_pred
    
    assert y_test.shape[0] == len(labels_test)
    slice_go_output = slice((3 * pd['t_delay'] + 2 * pd['t_stim'] - 1), (3 * pd['t_delay'] + 3 * pd['t_stim'] - 1))  # -1 b/c output is one time step ahaead from input
    if task == 'dms' or task == 'dmc':  # determine matches & non matches 
        match_train = np.where(np.array([x[0] == x[1] for x in labels_train]))[0]
        nonmatch_train = np.where(np.array([x[0] != x[1] for x in labels_train]))[0]
        y_train[match_train, slice_go_output, 6] = 1
        y_train[nonmatch_train, slice_go_output, 7] = 1
        
        match_test = np.where(np.array([x[0] == x[1] for x in labels_test]))[0]
        nonmatch_test = np.where(np.array([x[0] != x[1] for x in labels_test]))[0]
        y_test[match_test, slice_go_output, 6] = 1
        y_test[nonmatch_test, slice_go_output, 7] = 1
    
    x_train, y_train, x_test, y_test = map(
        torch.tensor, (x_train, y_train, x_test, y_test))  # create tensors
    x_train, y_train, x_test, y_test = x_train.float(), y_train.float(), x_test.float(), y_test.float()  # need to be float type (instead of 'double', which is somewhat silly)
    return (x_train, y_train, x_test, y_test), (labels_train, labels_test)


def fill_onehot_trials(all_seq=None, labels=None, task='dmc', pd=None):
    if task == 'dmc':
        n_cat = 2
    elif task == 'dms':
        n_cat = 2
    else:
        assert False, f'{task} not implement for onehot'
    
    if n_cat == 2:
        all_seq[:pd['n_half_total'], :, 1][:, pd['slice_s1']] = 1  # A1
        all_seq[pd['n_half_total']:, :, 2][:, pd['slice_s1']] = 1  # A2
    
        if task == 'dmc':
            add_task = 2
        elif task == 'dms':
            add_task = 0
    
        all_seq[:pd['n_exp_half'], :, (1 + add_task)][:, pd['slice_s2']] = 1  # exp C1
        labels[:pd['n_exp_half']] = '11'
        all_seq[pd['n_exp_half']:pd['n_half_total'], :, (2 + add_task)][:, pd['slice_s2']] = 1  #unexp C2
        labels[pd['n_exp_half']:pd['n_half_total']] = '12'
        all_seq[pd['n_half_total']:(pd['n_half_total'] + pd['n_exp_half']), :, (2 + add_task)][:, pd['slice_s2']] = 1 # exp C2
        labels[pd['n_half_total']:(pd['n_half_total'] + pd['n_exp_half'])] = '22'
        all_seq[(pd['n_half_total'] + pd['n_exp_half']):, :, (1 + add_task)][:, pd['slice_s2']] = 1  # unexp C1
        labels[(pd['n_half_total'] + pd['n_exp_half']):] = '21'
    
    elif n_cat == 4:
        assert task == 'dms'
        assert False, 'not implemented'
    
    return all_seq, labels


class RNN_MTL(nn.Module):
    def __init__(self, n_nodes, task='pred_only', init_std_scale=0.1):
        '''RNN Model with input/hidden/output layers. Fully connected. 
        
        Terminology:
        pred = prediction task 
        spec = specialisation task 
        pred_only = network that only solves prediction task 
        spec_only = .. 
        pred_spec = network that does both 
        spec in {dms, dmc, dmrs, dmrc, discr}
        '''
        super().__init__()

        ## Model parameters
        self.n_input = 6  # A1 A2 B1 B2 G 0
        self.n_output = 8  # = n_input + M1 M2
        self.n_nodes = n_nodes
        self.init_std_scale = init_std_scale
        self.task = task
        self.info_dict = {'converged': False, 'task': task}  # any info can be saved later
        task_names = self.task.split('_')
        assert len(task_names) == 2
        if task_names[1] != 'only':
            assert task_names[0] == 'pred'
        if 'pred' in task_names:
            self.train_pred_task = True 
        else:
            self.train_pred_task = False
        for spec_name in ['dms', 'dmc', 'dmrs', 'dmrc', 'discr']:
            if spec_name in task_names:  ## there can only be 1 due to asserts above
                self.info_dict['spec_task_name'] = spec_name
                self.train_spec_task = True 
                break 
            else:
                self.train_spec_task = False
    
        ## Linear combination layers:
        ## initialised uniformly from U(-K, K) where K = sqrt(1/n_input)  by default  
        self.lin_input = nn.Linear(self.n_input, self.n_nodes)
        self.lin_feedback = nn.Linear(self.n_nodes, self.n_nodes)
        self.lin_output = nn.Linear(self.n_nodes, self.n_output)
        # if self.train_pred_task is False:  # set output weight that should not be used to 0 to prevent unnecessary regularisaiton loss 
        #     self.lin_output.weight[:, :self.n_input] = 0
        # if self.train_spec_task is False:
        #     self.lin_output.weight[:, self.n_input:] = 0
        self.init_state()  # initialise RNN nodes

        ## Attributes to be completed later:
        self.train_loss_arr = []  # to be appended during training
        self.test_loss_arr = []
        self.test_loss_ratio_reg = []
        if self.train_pred_task:
            self.test_loss_split = {x: [] for x in ['pred', 'S2', 'G', 'G1', 'G2', 
                                                    '0', '0_postS1', '0_postS2', '0_postG']}
        else:
            self.test_loss_split = {}
        self.test_loss_split['L1'] = []
        if self.train_spec_task:
            self.test_loss_split[self.info_dict['spec_task_name']] = []
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
        self.rnn_name = 'RNN_MTL (not saved)'

    def __str__(self):
        """Define name"""
        return self.rnn_name

    def __repr__(self):
        """Define representation"""
        return f'Instance {self.rnn_name} of RNN_MTL Class'

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

        linear_output = self.lin_output(new_state.squeeze())
        output = torch.zeros_like(linear_output)  # we will normalise the prediction task & specialisation task separately:
        output[:self.n_input] = F.softmax(F.relu(linear_output[:self.n_input]), dim=0)  # output nonlin-lin of the prediction task (normalised on these only )
        output[self.n_input:] = F.softmax(F.relu(linear_output[self.n_input:]), dim=0)  # probabilities units for M and NM (normalised)
        return new_state, output

    def set_info(self, param_dict):
        '''Add information to the info dictionary. The param_dict is copied into
        info_dict (including overwriting).'''
        for key, val in param_dict.items():
            self.info_dict[key] = val  # overwrites

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
            print(f'RNN-MTL model saved as {self.file_name}')
            
def prediction_loss(y_est, y_true, model, eval_times=np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])):
    '''Compute Cross Entropy of given time array eval_times.'''
    # assert not (simulated_annealing and mnm_only), f'cannot do mnm only and SA simultaneously. sa = {simulated_annealing}, mnm = {mnm_only}'
    assert model.train_pred_task
    y_est_trunc = y_est[:, eval_times, :][:, :, :model.n_input]  # only evaluated these time points, cut off at n_input, because spec task follows after
    y_true_trunc = y_true[:, eval_times, :][:, :, :model.n_input]
    n_samples = y_true.shape[0]
    ce = torch.sum(-1 * y_true_trunc * torch.log(y_est_trunc)) / n_samples  # take the mean CE over samples
    return ce
    
def regularisation_loss(model, reg_param=None):  # default 0.001
    '''Compute L1 norm of all model parameters'''
    if reg_param is None:
        reg_param = model.info_dict['l1_param']
    reg_loss = 0
    params = [pp for pp in model.parameters()]  # for all weight (matrices) in the model
    for _, p_set in enumerate(params):
        reg_loss += reg_param * p_set.norm(p=1)
    return reg_loss 
    
def specialisation_loss(y_est, y_true, model, eval_times=np.array([9, 10])):
    '''Compute Cross Entropy of given time array eval_times.'''
    # assert not (simulated_annealing and mnm_only), f'cannot do mnm only and SA simultaneously. sa = {simulated_annealing}, mnm = {mnm_only}'
    assert model.train_spec_task
    y_est_trunc = y_est[:, eval_times, :][:, :, model.n_input:]  # only evaluated these time points, cut off at n_input, because spec task follows after
    y_true_trunc = y_true[:, eval_times, :][:, :, model.n_input:]
    n_samples = y_true.shape[0]
    ce = torch.sum(-1 * y_true_trunc * torch.log(y_est_trunc)) / n_samples  # take the mean CE over samples
    return ce
    
def total_loss(y_est, y_true, model):
    if model.train_pred_task:
        pred_loss = prediction_loss(y_est=y_est, y_true=y_true, model=model)
    else:
        pred_loss = 0
    if model.train_spec_task:
        spec_loss = specialisation_loss(y_est=y_est, y_true=y_true, model=model)
    else:
        spec_loss = 0
    reg_loss = regularisation_loss(model=model)
    total_loss = pred_loss + spec_loss + reg_loss 
    ratio_reg = reg_loss / total_loss 
    return total_loss, ratio_reg
    
def test_loss_append_split(y_est, y_true, model, time_prediction_array_dict=None, late_s2=False):
    if model.train_pred_task:
        if time_prediction_array_dict is None and late_s2 is False:
            time_prediction_array_dict={'S2': [5, 6], 'G': [9, 10], 'G1': [9], 'G2': [10], 
                                        '0': [3, 4, 7, 8, 11, 12], '0_postS1': [3, 4],
                                        '0_postS2': [7, 8], '0_postG': [11, 12]}
        elif time_prediction_array_dict is None and late_s2 is True:
            assert False, 'late beta not implemented'
        assert time_prediction_array_dict is not None and type(time_prediction_array_dict) == dict
        
        for key, eval_times in time_prediction_array_dict.items():  # compute separate times separately
            ce = prediction_loss(y_est=y_est, y_true=y_true, model=model, eval_times=eval_times)
            model.test_loss_split[key].append(float(ce.detach().numpy()))  # add to model

    reg_loss = regularisation_loss(model=model, reg_param=None)  # default uses model param 
    model.test_loss_split['L1'].append(float(reg_loss.detach().numpy()))  # add to array
    
    if model.train_spec_task:
        spec_loss = specialisation_loss(y_est=y_est, y_true=y_true, model=model)
        task_name = model.info_dict['spec_task_name']
        model.test_loss_split[task_name].append(float(reg_loss.detach().numpy()))  # add to array
        
    tot_loss, ratio_reg = total_loss(y_est=y_est, y_true=y_true, model=model)
    model.test_loss_arr.append(float(tot_loss.detach().numpy()))
    model.test_loss_ratio_reg.append(float(ratio_reg.detach().numpy()))
    

# def dmc_loss(y_est, model,label=None, match_times=[13, 14], mnm_loss_separate=False):
#     ''' Compute loss of MNM task'''
#     assert match_times is not None, 'no match times defined '
#     assert label is not None, 'no labels defined'
#     match_arr = ru.labels_to_mnm(labels=label)  # get M & NM binary rep
#     match_est = y_est[:, match_times, model.n_input:] # estimates for M, NM
#     match_arr_full = torch.zeros_like(match_est)  # has to be rescaled because of time axis
#     for tt in range(len(match_times)):
#         match_arr_full[:, tt, :] = torch.tensor(match_arr)  # concatenated along time axis
#     if mnm_loss_separate is False: # if P(M) + P(NM) == 1
#         ce_match = torch.sum(-1 * match_arr_full * torch.log(match_est)) / n_samples  # take the mean CE over samples
#     elif mnm_loss_separate:  # if P(M) <= 1 & P(NM) <=1
#         assert match_est.shape[2] == 2  # (M, NM)
#         match_only_est = match_est
#         match_only_est[:, :, 1] = 1 - match_only_est[:, :, 0]
#         nonmatch_only_est = match_est
#         nonmatch_only_est[:, :, 0] = 1 - nonmatch_only_est[:, :, 1]
#         ce_match_only = torch.sum(-1 * match_arr_full * torch.log(match_only_est)) / n_samples
#         ce_nonmatch_only = torch.sum(-1 * match_arr_full * torch.log(nonmatch_only_est)) / n_samples
#         ce_match = 0.5 * (ce_match_only + ce_nonmatch_only) # mean
#     return ce_match


def compute_full_pred(input_data, model):
    '''Compute forward prediction of RNN. I.e. given an input series input_data, the
    model (RNN) computes the predicted output series.'''
    if input_data.ndim == 2:
        input_data = input_data[None, :, :]
    full_pred = torch.zeros((input_data.shape[0], input_data.shape[1], input_data.shape[2] + 2))  # make space for two M elements
    for kk in range(input_data.shape[0]): # loop over trials
        model.init_state()  # initiate rnn state per trial
        for tt in range(input_data.shape[1]):  # loop through time
            _, full_pred[kk, tt, :] = model(input_data[kk, tt, :])  # compute prediction at this time
    return full_pred



def bptt_training(rnn, optimiser, dict_training_params,
                  x_train, x_test, y_train, y_test, verbose=1, late_s2=False):
    '''Training algorithm for backpropagation through time, given a RNN model, optimiser,
    dictionary with training parameters and train and test data. RNN is NOT reset,
    so continuation training is possible. Training can be aborted prematurely by Ctrl+C,
    and it will terminate correctly.'''
    assert dict_training_params['bs'] == 1, 'batch size is not 1; this error is thrown because for MNM we assume it is 1 to let labels correspond to dataloadre loop'
    assert late_s2 is False, 'not implemented'
    ## Create data loader objects:
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=dict_training_params['bs'])

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=dict_training_params['bs'])

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
                    # curr_label = labels_train[it_train]  # this works if batch size == 1
                    full_pred = compute_full_pred(model=rnn, input_data=xb)  # predict time trace
                    loss, _ = total_loss(y_est=full_pred, y_true=yb, model=rnn)
                    loss.backward()  # compute gradients
                    optimiser.step()  # update
                    optimiser.zero_grad()   # reset
                    it_train += 1

                rnn.eval()  # evaluation mode -> disable gradient tracking
                with torch.no_grad():  # to be sure
                    ## Compute losses for saving:
                    full_train_pred = compute_full_pred(model=rnn, input_data=x_train)
                    train_loss, _ = total_loss(y_est=full_train_pred, y_true=y_train, model=rnn)
                    rnn.train_loss_arr.append(float(train_loss.detach().numpy()))

                    full_test_pred = compute_full_pred(model=rnn, input_data=x_test)
                    test_loss_append_split(y_est=full_test_pred, y_true=y_test, model=rnn)  # append loss within function 

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



def init_train_save_rnn(t_dict, d_dict, n_simulations=1, save_folder='models/',
                        late_s2=False, nature_stim='onehot', type_task='dmc', train_task='pred_only'):
    assert late_s2 is False, 'not implemented'
    assert type_task in ['dms', 'dmc']
    assert train_task in ['pred_only', 'spec_only', 'pred_spec']
    if train_task == 'pred_only':
        task_name = 'pred_only'
    elif train_task == 'spec_only':
        task_name = f'{type_task}_only'
    elif train_task == 'pred_spec':
        task_name = f'pred_{type_task}'
    try:
        for nn in range(n_simulations):
            print(f'\n-----------\nsimulation {nn}/{n_simulations}')
            ## Generate data:
            tmp0, tmp1 = generate_synt_data_general(n_total=d_dict['n_total'], t_delay=d_dict['t_delay'], t_stim=d_dict['t_stim'], 
                                        ratio_train=d_dict['ratio_train'], ratio_exp=d_dict['ratio_exp'],
                                        noise_scale=d_dict['noise_scale'], late_s2=False,
                                        nature_stim=nature_stim, task=type_task)
                    
            x_train, y_train, x_test, y_test = tmp0
            labels_train, labels_test = tmp1

            ## Initiate RNN model
            rnn = RNN_MTL(task=task_name, n_nodes=t_dict['n_nodes'])  # Create RNN class
            opt = torch.optim.SGD(rnn.parameters(), lr=t_dict['learning_rate'])  # call optimiser from pytorhc
            rnn.set_info(param_dict={**d_dict, **t_dict})
            rnn.info_dict['type_task'] = type_task
            rnn.info_dict['train_task'] = train_task
            rnn.info_dict['late_s2'] = late_s2
            
            ## Train with BPTT
            rnn = bptt_training(rnn=rnn, optimiser=opt, dict_training_params=t_dict,
                                x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                verbose=0, late_s2=late_s2)
                                
            # ## Decode cross temporally
            # score_mat, decoder_dict, _ = train_single_decoder_new_data(rnn=rnn, ratio_expected=0.5,
            #                                                 n_samples=None, ratio_train=0.8, verbose=False,
            #                                                 late_s2=late_s2)

            ## Save results:
            rnn.save_model(folder=save_folder)
        return rnn  # return latest
    except KeyboardInterrupt:
        print('KeyboardInterrupt, exit')






















