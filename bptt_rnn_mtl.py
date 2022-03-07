# @Author: Thijs L van der Plas <thijs>
# @Date:   2021-04-13
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Filename: bptt_rnn_mtl.py
# @Last modified by:   thijs
# @Last modified time: 2021-06-01


import os, sys, string
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle, datetime, time, os, sys, git
from tqdm import tqdm, trange
import sklearn.svm, sklearn.model_selection, sklearn.discriminant_analysis
import rot_utilities as ru
# from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import itertools
from itertools import repeat as irep
import copy


device = 'cpu'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def generate_synt_data_general(n_total=100, t_delay=2, t_stim=2,
                               ratio_train=0.8, ratio_exp=0.75,
                               noise_scale=0.05, late_s2=False,
                               early_match=False,
                               nature_stim='onehot', task='dmc'):
    '''Generate synthetic data

    nature_stim: onehot, periodic, tuning
    task: dms, dmc, dmrs, dmrc, discr
    late_s2: if true, present s2 during original GO window (and GO after)
    early_match: if true, prompt for MNM during S2 presentation'''
    assert (late_s2 and early_match) is False
    # assert late_s2 is False, 'Late s2 not implemented'
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
    if late_s2 is False:
        pd['slice_s2'] = slice((2 * pd['t_delay'] + pd['t_stim']), (2 * pd['t_delay'] + 2 * pd['t_stim']))
    elif late_s2:
        pd['slice_s2'] = slice((3 * pd['t_delay'] + 2 * pd['t_stim']), (3 * pd['t_delay'] + 3 * pd['t_stim']))
    ## Create data sequences of 5, 7 or 9 elements
    ## 0-0   1-A1    2-A2    3-B1    4-B2    5-G
    all_seq = np.zeros((pd['n_total'], pd['n_times'], pd['n_input']))
    labels = np.zeros(pd['n_total'], dtype='object')
    if late_s2 is False:
        for i_delay in range(4):  # 4 delay periods
            all_seq[:, :, 0][:, (i_delay * pd['period']):(i_delay * pd['period'] + t_delay)] = 1
        if early_match is False:
            all_seq[:, :, 5][:, (3 * t_delay + 2 * t_stim):(3 * t_delay + 3 * t_stim)] = 1  # Go cue
        elif early_match:
            all_seq[:, :, 5][:, pd['slice_s2']] = 1  # Go cue
    elif late_s2:
        for i_delay in range(3):  # 3 delay periods
            all_seq[:, :, 0][:, (i_delay * pd['period']):(i_delay * pd['period'] + t_delay)] = 1
        all_seq[:, :, 5][:, (3 * pd['period']):(3 * pd['period'] + t_delay)] = 1  # late Go cue
        all_seq[:, :, 0][:, (2 * t_delay + 1 * t_stim):(2 * t_delay + 2 * t_stim)] = 1  # extra delay during regular s2 time
    ## First fill in sequence of trials, shuffle later
    if nature_stim == 'onehot':
        all_seq, labels = fill_onehot_trials(all_seq=all_seq, labels=labels, task=task, pd=pd, late_s2=late_s2)
    elif nature_stim == 'periodic':
        all_seq, labels = fill_periodic_trials(all_seq=all_seq, labels=labels, task=task, pd=pd, late_s2=late_s2)
    elif nature_stim == 'tuning':
        pass
    elif nature_stim == 'binary':
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
    if late_s2 is False:
        if early_match is False:
            slice_go_output = slice((3 * pd['t_delay'] + 2 * pd['t_stim'] - 1), (3 * pd['t_delay'] + 3 * pd['t_stim'] - 1))  # -1 b/c output is one time step ahaead from input
        elif early_match:
            slice_go_output = slice((2 * pd['t_delay'] + 1 * pd['t_stim'] - 1), (2 * pd['t_delay'] + 2 * pd['t_stim'] - 1))  # -1 b/c output is one time step ahaead from input
    elif late_s2 is True:
        slice_go_output = slice((3 * pd['t_delay'] + 3 * pd['t_stim'] - 1), (4 * pd['t_delay'] + 3 * pd['t_stim'] - 1))  # -1 b/c output is one time step ahaead from input
    if task == 'dms' or task == 'dmc' or task == 'dmrs' or task == 'dmrc':  # determine matches & non matches
        # match_train = np.where(np.array([x[0] == x[1] for x in labels_train]))[0]
        # nonmatch_train = np.where(np.array([x[0] != x[1] for x in labels_train]))[0]
        match_train = np.where(np.array([x[1] != 'x' for x in labels_train]))[0]
        nonmatch_train = np.where(np.array([x[1] == 'x' for x in labels_train]))[0]
        y_train[match_train, slice_go_output, 6] = 1
        y_train[nonmatch_train, slice_go_output, 7] = 1

        match_test = np.where(np.array([x[1] != 'x' for x in labels_test]))[0]
        nonmatch_test = np.where(np.array([x[1] == 'x'  for x in labels_test]))[0]
        y_test[match_test, slice_go_output, 6] = 1
        y_test[nonmatch_test, slice_go_output, 7] = 1

    x_train, y_train, x_test, y_test = map(
        torch.tensor, (x_train, y_train, x_test, y_test))  # create tensors
    x_train, y_train, x_test, y_test = x_train.float(), y_train.float(), x_test.float(), y_test.float()  # need to be float type (instead of 'double', which is somewhat silly)
    return (x_train, y_train, x_test, y_test), (labels_train, labels_test)


def fill_onehot_trials(all_seq=None, labels=None, task='dmc', pd=None, late_s2=False):
    """Add OH data into all_seq."""
    if task == 'dmc':
        n_cat = 2
    elif task == 'dms':
        n_cat = 2
    else:
        assert False, f'{task} not implement for onehot'

    if n_cat == 2:
        all_seq[:pd['n_half_total'], :, 1][:, pd['slice_s1']] = 1  # S1
        all_seq[pd['n_half_total']:, :, 2][:, pd['slice_s1']] = 1  # S2

        if task == 'dmc':
            add_task = 2
        elif task == 'dms':
            add_task = 0

        all_seq[:pd['n_exp_half'], :, (1 + add_task)][:, pd['slice_s2']] = 1  # exp S2-1
        labels[:pd['n_exp_half']] = '11'
        all_seq[pd['n_exp_half']:pd['n_half_total'], :, (2 + add_task)][:, pd['slice_s2']] = 1  #unexp S2-2
        labels[pd['n_exp_half']:pd['n_half_total']] = '1x'
        all_seq[pd['n_half_total']:(pd['n_half_total'] + pd['n_exp_half']), :, (2 + add_task)][:, pd['slice_s2']] = 1 # exp S2-2
        labels[pd['n_half_total']:(pd['n_half_total'] + pd['n_exp_half'])] = '22'
        all_seq[(pd['n_half_total'] + pd['n_exp_half']):, :, (1 + add_task)][:, pd['slice_s2']] = 1  # unexp S2-1
        labels[(pd['n_half_total'] + pd['n_exp_half']):] = '2x'

    elif n_cat == 4:
        assert task == 'dms'
        assert False, 'not implemented'

    return all_seq, labels


def fill_periodic_trials(all_seq=None, labels=None, task='dmc', pd=None, n_cat=4, late_s2=False):
    """Add periodic/4 sample data into all_seq"""
    assert pd['n_total'] % n_cat == 0, 'number of categories not a factor of number of trials'
    assert task == 'dmc' or task == 'dms' or task == 'dmrs' or task == 'dmrc'
    assert n_cat < 10  # to stay within 1 digit with labelling
    assert late_s2 is False, 'late s2 not implemented for periodic trials'
    ## Create periodic stim by cos & sin of angle
    rad_stim = np.array([x * 2 * np.pi / n_cat for x in range(n_cat)])
    cos_stim = np.cos(rad_stim)
    sin_stim = np.sin(rad_stim)

    if task == 'dmc' or task == 'dmrc':
        add_task = 2
    elif task == 'dms' or task == 'dmrs':
        add_task = 0
    n_trials_per_cat = int(np.round(pd['n_total'] / n_cat))
    n_trials_exp_per_cat = int(np.round(pd['ratio_exp'] * n_trials_per_cat))
    n_trials_unexp_per_cat = n_trials_per_cat - n_trials_exp_per_cat
    assert n_trials_exp_per_cat + n_trials_unexp_per_cat == n_trials_per_cat

    ## Fill sequences per S1 category
    for i_cat in range(n_cat):
        i_trial = i_cat * n_trials_per_cat
        i_next_cat_trial = (i_cat + 1) * n_trials_per_cat
        ## Fill S1
        all_seq[i_trial:i_next_cat_trial, :, 1][:, pd['slice_s1']] = cos_stim[i_cat]
        all_seq[i_trial:i_next_cat_trial, :, 2][:, pd['slice_s1']] = sin_stim[i_cat]

        ## Fill S2
        if task == 'dms' or task == 'dmc':
            match_cat = i_cat
        elif task == 'dmrs' or task == 'dmrc':
            match_cat = (i_cat + 1) % n_cat
        all_seq[i_trial:(i_trial + n_trials_exp_per_cat), :, (1 + add_task)][:, pd['slice_s2']] = cos_stim[match_cat]  # same stim
        all_seq[i_trial:(i_trial + n_trials_exp_per_cat), :, (2 + add_task)][:, pd['slice_s2']] = sin_stim[match_cat]
        labels[i_trial:(i_trial + n_trials_exp_per_cat)] = f'{i_cat}{match_cat}'

        random_cat = np.random.choice(a=np.delete(np.arange(n_cat), match_cat, 0), size=n_trials_unexp_per_cat, replace=True)
        all_seq[(i_trial + n_trials_exp_per_cat):i_next_cat_trial, :, (1 + add_task)][:, pd['slice_s2']] = np.array([cos_stim[x] for x in random_cat])[:, None]  # different stim
        all_seq[(i_trial + n_trials_exp_per_cat):i_next_cat_trial, :, (2 + add_task)][:, pd['slice_s2']] = np.array([sin_stim[x] for x in random_cat])[:, None]
        # labels[(i_trial + n_trials_exp_per_cat):i_next_cat_trial] = [f'{i_cat}{i_random_cat.copy()}' for i_random_cat in random_cat]  # specify other cat
        labels[(i_trial + n_trials_exp_per_cat):i_next_cat_trial] = [f'{i_cat}x' for i_random_cat in random_cat]  # do not specifiy => best for shuffling with StratifiedShuffleSplit

    return all_seq, labels


class RNN_MTL(nn.Module):
    def __init__(self, n_nodes=20, nature_stim='onehot', task='pred_dmc', init_std_scale=0.1):
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
        self.info_dict = {'converged': False, 'task': task, 'output_nonlin_pred': 'softmax',
                          'output_nonlin_spec': 'softmax_relu', 'pred_loss_function': 'cross_entropy',
                          'nature_stim': nature_stim}  # any info can be saved later
        if nature_stim == 'periodic':  # periodic stim are constrained by sum = 1, so softmax & cross-entropy are not applicable
            self.info_dict['pred_loss_function'] = 'mean_squared_error'  # also like duncker & driscoll 2020 NIPS I believe
            self.info_dict['output_nonlin_pred'] = 'tanh'  # bound -1 to 1, like cos & sin
        task_names = self.task.split('_')
        assert len(task_names) == 2, f'task_names {task_names} cannot be interpreted'
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
        self.__version__ = '1.0'
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
        # rnn_state.to(device)  # if use_gpu
        lin_comb = self.lin_input(inp) + self.lin_feedback(rnn_state)  # input + previous state
        new_state = torch.tanh(lin_comb)  # transfer function
        self.state = new_state

        linear_output = self.lin_output(new_state.squeeze())
        output = torch.zeros_like(linear_output)  # we will normalise the prediction task & specialisation task separately:
        if self.info_dict['output_nonlin_pred'] == 'softmax':
            output[:self.n_input] = F.softmax(linear_output[:self.n_input], dim=0)  # output nonlin-lin of the prediction task (normalised on these only )
        elif self.info_dict['output_nonlin_pred'] == 'softmax_relu':
            output[:self.n_input] = F.softmax(F.relu(linear_output[:self.n_input]), dim=0)  # output nonlin-lin of the prediction task (normalised on these only )
        elif self.info_dict['output_nonlin_pred'] == 'tanh':
            output[:self.n_input] = torch.tanh(linear_output[:self.n_input])
        else:
            assert False, 'output nonlinearity not defined'
        if self.info_dict['output_nonlin_spec'] == 'softmax':
            output[self.n_input:] = F.softmax(linear_output[self.n_input:], dim=0)
        elif self.info_dict['output_nonlin_spec'] == 'softmax_relu':
            output[self.n_input:] = F.softmax(F.relu(linear_output[self.n_input:]), dim=0)  # probabilities units for M and NM (normalised)
        else:
            assert False, 'output nonlinearly not defined'
        return new_state, output

    def set_info(self, param_dict):
        '''Add information to the info dictionary. The param_dict is copied into
        info_dict (including overwriting).'''
        for key, val in param_dict.items():
            self.info_dict[key] = val  # overwrites

    def save_model(self, folder=None, verbose=True, add_nnodes=False, allow_name_change=True):  # redefine because we want to change saving name
        '''Export this RNN model to folder. If self.file_name is None, it is saved  under
        a timestamp.'''
        dt = datetime.datetime.now()
        timestamp = str(dt.date()) + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)
        self.info_dict['timestamp'] = timestamp
        current_suffix = None
        if self.file_name is None:
            if add_nnodes is False:
                self.rnn_name = f'rnn-mnm_{timestamp}'
                self.file_name = f'rnn-mnm_{timestamp}.data'
            else:
                self.rnn_name = f'rnn-mnm_n{self.info_dict["n_nodes"]}_{timestamp}'
                self.file_name = f'rnn-mnm_n{self.info_dict["n_nodes"]}_{timestamp}.data'
        else:
            current_timestamp_suffix = self.full_path.split('/')[-1].split('-')[-1][:-5]
            if len(current_timestamp_suffix) > 4:  # some suffic after hhmm timestamp
                current_suffix = current_timestamp_suffix[4:]
        if folder is None:
            folder = 'models/'
        elif folder[-1] != '/':
            folder += '/'
        self.full_path = folder + self.file_name
        if current_suffix is not None:
            self.full_path = folder + self.file_name[:-5] + current_suffix + '.data'
        if allow_name_change:
            i_ascii = 0
            suffix_list = string.ascii_letters
            while os.path.exists(self.full_path):
                self.full_path = folder + self.file_name[:-5] + suffix_list[i_ascii] + '.data'
                i_ascii += 1
                if i_ascii == len(suffix_list):
                    print('WARNING: EXPANDING ASCII LIST')
                    suffix_list = [x + x for x in suffix_list]
                    i_ascii = 0  # reset because suffix_list has been changed
                    assert False, 'safety stop'
        file_handle = open(self.full_path, 'wb')
        pickle.dump(self, file_handle)
        if verbose > 0:
            print(f'RNN-MTL model saved as {self.file_name}')

def prediction_loss(y_est, y_true, model, eval_times=np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
                    loss_function=None):
    '''Compute Cross Entropy of prediction loss given time array eval_times.'''
    # assert not (simulated_annealing and mnm_only), f'cannot do mnm only and SA simultaneously. sa = {simulated_annealing}, mnm = {mnm_only}'
    assert model.train_pred_task
    assert y_est.shape == y_true.shape
    assert y_est.shape[1] == 13
    if loss_function is None:
        loss_function = model.info_dict['pred_loss_function']
    y_est_trunc = y_est[:, eval_times, :][:, :, :model.n_input]  # only evaluated these time points, cut off at n_input, because spec task follows after
    y_true_trunc = y_true[:, eval_times, :][:, :, :model.n_input]
    n_samples = y_true.shape[0]
    if loss_function == 'cross_entropy':
        assert y_true_trunc.sum(2).mean() == 1, y_true_trunc.sum(2).mean() # sum should be 1 to use cross entropy
        loss = torch.sum(-1 * y_true_trunc * torch.log(y_est_trunc)) / n_samples  # take the mean CE over samples
    elif loss_function == 'mean_squared_error':
        # loss_f = nn.MSELoss(reduction='mean')
        # loss = loss_f(y_true_trunc, y_est_trunc)
        loss = ((y_true_trunc - y_est_trunc) ** 2).sum() / n_samples
    return loss

def regularisation_loss(model, reg_param=None):  # default 0.001
    '''Compute L1 norm of all model parameters'''
    if reg_param is None:
        reg_param = model.info_dict['l1_param']
    reg_loss = 0
    params = [pp for pp in model.parameters()]  # for all weight (matrices) in the model
    for _, p_set in enumerate(params):
        reg_loss += reg_param * p_set.norm(p=1)
    return reg_loss

def specialisation_loss(y_est, y_true, model, eval_times=np.array([9, 10]), late_s2=False):
    '''Compute Cross Entropy of specialisation loss given time array eval_times.'''
    # assert not (simulated_annealing and mnm_only), f'cannot do mnm only and SA simultaneously. sa = {simulated_annealing}, mnm = {mnm_only}'
    assert model.train_spec_task
    assert y_est.shape == y_true.shape
    assert y_est.shape[1] == 13
    if late_s2:
        eval_times = np.array([11, 12])
    if 'early_match' in model.info_dict.keys():
        if model.info_dict['early_match'] is True:
            early_match = True
    else:
        early_match = False
    if early_match:
        # print('using ealry match in spec loss!')
        eval_times = np.array([5, 6])
    y_est_trunc = y_est[:, eval_times, :][:, :, model.n_input:]  # only evaluated these time points, cut off at n_input, because spec task follows after
    y_true_trunc = y_true[:, eval_times, :][:, :, model.n_input:]
    assert y_true_trunc.sum(2).mean() == 1, f'mean: {y_true_trunc.sum(2).mean()}, eval: {eval_times}, neurons {model.n_input}, shape {y_true_trunc.shape}, late_s2: {late_s2}'  # sum should be 1 to use cross entropy
    n_samples = y_true.shape[0]
    ce = torch.sum(-1 * y_true_trunc * torch.log(y_est_trunc)) / n_samples  # take the mean CE over samples, natural log
    return ce

def total_loss(y_est, y_true, model, late_s2):
    """Compute total loss, pred and spec are taken into account based on info in rnn model"""
    if model.train_pred_task:
        pred_loss = prediction_loss(y_est=y_est, y_true=y_true, model=model)
    else:
        pred_loss = 0
    if model.train_spec_task:
        spec_loss = specialisation_loss(y_est=y_est, y_true=y_true, model=model, late_s2=late_s2)
    else:
        spec_loss = 0
    reg_loss = regularisation_loss(model=model)
    total_error = pred_loss + spec_loss + reg_loss
    ratio_reg = reg_loss / total_error
    return total_error, ratio_reg

def test_loss_append_split(y_est, y_true, model, time_prediction_array_dict=None, late_s2=False):
    """append split pred losses and spec losses to rnn model"""
    if model.train_pred_task:
        if time_prediction_array_dict is None and late_s2 is False:
            time_prediction_array_dict={'S2': [5, 6], 'G': [9, 10], 'G1': [9], 'G2': [10],
                                        '0': [3, 4, 7, 8, 11, 12], '0_postS1': [3, 4],
                                        '0_postS2': [7, 8], '0_postG': [11, 12]}
        elif time_prediction_array_dict is None and late_s2 is True:
            time_prediction_array_dict={'S2': [9, 10], 'G': [11, 12], 'G1': [11], 'G2': [12],
                                        '0': [3, 4, 5, 6, 7, 8], '0_postS1': [3, 4]}
        assert time_prediction_array_dict is not None and type(time_prediction_array_dict) == dict

        for key, eval_times in time_prediction_array_dict.items():  # compute separate times separately
            ce = prediction_loss(y_est=y_est, y_true=y_true, model=model, eval_times=eval_times)
            model.test_loss_split[key].append(float(ce.detach().numpy()))  # add to model

        ce = prediction_loss(y_est=y_est, y_true=y_true, model=model)  # full prediction error
        model.test_loss_split['pred'].append(float(ce.detach().numpy()))  # add to model

    reg_loss = regularisation_loss(model=model, reg_param=None)  # default uses model param
    model.test_loss_split['L1'].append(float(reg_loss.detach().numpy()))  # add to array

    if model.train_spec_task:
        spec_loss = specialisation_loss(y_est=y_est, y_true=y_true, model=model, late_s2=late_s2)
        task_name = model.info_dict['spec_task_name']
        model.test_loss_split[task_name].append(float(spec_loss.detach().numpy()))  # add to array

    tot_loss, ratio_reg = total_loss(y_est=y_est, y_true=y_true, model=model, late_s2=late_s2)
    model.test_loss_arr.append(float(tot_loss.detach().numpy()))
    model.test_loss_ratio_reg.append(float(ratio_reg.detach().numpy()))

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

def bptt_training(rnn, optimiser, dict_training_params, d_dict=None,
                  x_train=None, x_test=None, y_train=None, y_test=None,
                  simulated_annealing=False, ratio_exp_array=None,
                  verbose=1, late_s2=False, use_gpu=False, save_state=False):
    '''Training algorithm for backpropagation through time, given a RNN model, optimiser,
    dictionary with training parameters and train and test data. RNN is NOT reset,
    so continuation training is possible. Training can be aborted prematurely by Ctrl+C,
    and it will terminate correctly.'''
    # assert dict_training_params['bs'] == 1, 'batch size is not 1; this error is thrown because for MNM we assume it is 1 to let labels correspond to dataloadre loop'
    if simulated_annealing is False:
        assert x_train is not None  #and also the others technically
    else:
        assert ratio_exp_array is not None
        assert d_dict is not None

    if simulated_annealing is False:  # use same data [that is passed as arg] on each epoch
        ## Create data loader objects:
        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=dict_training_params['bs'])

        test_ds = TensorDataset(x_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=dict_training_params['bs'])
        total_epochs = dict_training_params['n_epochs']
    else: # generate new data on each epoch (with varying ratio_exp)
        print('Sim annealing')
        total_epochs = len(ratio_exp_array)
        assert total_epochs == dict_training_params['n_epochs']
        rnn.info_dict['ratio_exp_array'] = ratio_exp_array

    prev_loss = 10  # init loss for convergence
    if 'trained_epochs' not in rnn.info_dict.keys():
        rnn.info_dict['trained_epochs'] = 0
    else:
        assert simulated_annealing is False, 'multiple SA sequences not implemented'

    if save_state:
        if hasattr(rnn, 'saved_states_dict') is False:
            rnn.saved_states_dict = {}

    ## Training procedure
    init_str = f'Initialising training; start at epoch {rnn.info_dict["trained_epochs"]}'
    try:
        with trange(total_epochs) as tr:  # repeating epochs
            for epoch in tr:
                if simulated_annealing:
                    ## create data for this epoch
                    assert 'early_match' not in rnn.info_dict.keys()
                    tmp0, tmp1 = generate_synt_data_general(n_total=d_dict['n_total'], t_delay=d_dict['t_delay'], t_stim=d_dict['t_stim'],
                                                ratio_train=d_dict['ratio_train'], ratio_exp=ratio_exp_array[epoch],  # with current exp ratio
                                                noise_scale=d_dict['noise_scale'], late_s2=late_s2,
                                                nature_stim=rnn.info_dict['nature_stim'], task=rnn.info_dict['type_task'])

                    x_train, y_train, x_test, y_test = tmp0
                    train_ds = TensorDataset(x_train, y_train)
                    train_dl = DataLoader(train_ds, batch_size=dict_training_params['bs'])

                    test_ds = TensorDataset(x_test, y_test)
                    test_dl = DataLoader(test_ds, batch_size=dict_training_params['bs'])

                if epoch == 0:
                    tr.set_description(init_str)
                else:
                    update_str = f'Epoch {epoch}/{dict_training_params["n_epochs"]}. Train loss: {np.round(prev_loss, 6)}'
                    tr.set_description(update_str)
                if save_state:
                    rnn.saved_states_dict[epoch] = copy.deepcopy(rnn.state_dict())  # make copy of parameters

                rnn.train()  # set to train model (i.e. allow gradient computation/tracking)
                it_train = 0

                for xb, yb in train_dl:  # returns torch(n_bs x n_times x n_freq)
                    if use_gpu:
                        xb, yb = xb.to(device), yb.to(device)
                        rnn.to(device)
                    # curr_label = labels_train[it_train]  # this works if batch size == 1
                    full_pred = compute_full_pred(model=rnn, input_data=xb)  # predict time trace
                    loss, _ = total_loss(y_est=full_pred, y_true=yb, model=rnn, late_s2=late_s2)
                    loss.backward()  # compute gradients
                    optimiser.step()  # update
                    optimiser.zero_grad()   # reset
                    it_train += 1

                rnn.eval()  # evaluation mode -> disable gradient tracking
                with torch.no_grad():  # to be sure
                    ## Compute losses for saving:
                    full_train_pred = compute_full_pred(model=rnn, input_data=x_train)
                    train_loss, _ = total_loss(y_est=full_train_pred, y_true=y_train, model=rnn, late_s2=late_s2)
                    if use_gpu:
                        train_loss = train_loss.cpu()
                    rnn.train_loss_arr.append(float(train_loss.detach().numpy()))

                    full_test_pred = compute_full_pred(model=rnn, input_data=x_test)
                    test_loss_append_split(y_est=full_test_pred, y_true=y_test, model=rnn, late_s2=late_s2)  # append loss within function

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
        if save_state:
            rnn.saved_states_dict[epoch + 1] = copy.deepcopy(rnn.state_dict())  # make copy of parameters

        if verbose > 0:
            print('Training finished. Results saved in RNN Class')
        return rnn
    except KeyboardInterrupt: # end prematurely by Ctrl+C
        rnn.eval()
        if verbose > 0:
            print(f'Training ended prematurely by user at epoch {epoch}.\nResults saved in RNN Class.')
        return rnn


def train_decoder(rnn_model, x_train, x_test, labels_train, labels_test,
                  save_inplace=False, label_name='s1', sparsity_c=1e-1,
                  bool_train_decoder=True, decoder_type='logistic_regression'):

    """Train decoder on rnn_model given data, for label_name representaoitn.
    if bool_train_decoder is False, then the decoder is not trained (but a forward pass
    is done). If save_inplace is True the results are saved in the RNN (and they are always returned)"""
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

        if bool_train_decoder:
            ## Train decoder
            assert rnn_model.info_dict['nature_stim'] == 'onehot', 'periodic not yet implemented because S2 decoding is determined by label (is not specific because of =x)'
            s1_labels = {'train': np.array([int(x[0]) for x in labels_train]),
                         'test': np.array([int(x[0]) for x in labels_test])}
             # s2_labels = {'train': np.array([int(x[1]) for x in labels_train]),
             #              'test': np.array([int(x[1]) for x in labels_test])}
            s2_labels = {'train': np.zeros(len(labels_train), dtype='int'),
                         'test': np.zeros(len(labels_test), dtype='int')}
            mnm_labels = {'train': np.array([int(x[0] == x[1]) for x in labels_train]),
                          'test': np.array([int(x[0] == x[1]) for x in labels_test])}
            for ds_type, labels_type in zip(('train', 'test'), (labels_train, labels_test)):
                for i_lab, lab in enumerate(labels_type):
                    if lab[1] == 'x':
                        numeric_lab = ('1' if lab[0] == '2' else '2')  # opposite from first label
                        s2_labels[ds_type][i_lab] = numeric_lab
                    else:
                        s2_labels[ds_type][i_lab] = lab[1]
            if label_name == 's1':
                labels_use = s1_labels
            elif label_name == 's2':
                labels_use = s2_labels
                # assert False, 'double check to make sure s2_labels is handled correctly by decoder. because it is filled with floats not str'
            elif label_name == 'go':
                labels_use = mnm_labels
            else:
                assert False, f'Label name {label_name} not implemented. Please choose s1 or s2 or go. Aborting'
            score_mat = np.zeros((n_times, n_times))  # T x T
            decoder_dict = {}  # save decoder per time
            for tau in range(n_times):  # train time loop
                # decoder_dict[tau] = sklearn.svm.LinearSVC(C=sparsity_c)  # define SVM
                if decoder_type == 'logistic_regression':
                    decoder_dict[tau] = sklearn.linear_model.LogisticRegression(C=sparsity_c,
                                                            solver='saga', penalty='l1', max_iter=250)  # define log reg
                elif decoder_type == 'LDA':
                     decoder_dict[tau] = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()  # define log reg
                decoder_dict[tau].fit(X=forw_mat['train'][:, tau, :],
                                      y=labels_use['train'])  # train SVM
                for tt in range(n_times):  # test time loop
                    # score_mat[tau, tt] = decoder_dict[tau].score(X=forw_mat['test'][:, tt, :],
                    #                                              y=labels_use['test'])  # evaluate
                    prediction = decoder_dict[tau].predict_proba(X=forw_mat['test'][:, tt, :])
                    inds_labels = np.zeros_like(labels_use['test'], dtype='int')  # zero = class 0
                    inds_labels[(labels_use['test'] == decoder_dict[tau].classes_[1])] = 1
                    prob_correct = np.array([prediction[i_pred, ind] for i_pred, ind in enumerate(inds_labels)])
                    score_mat[tau, tt] = np.mean(prob_correct)
                    # score_mat[tau, tt] = np.exp(np.mean(np.log(prob_correct)))
    if bool_train_decoder:
        if save_inplace:
            rnn_model.decoding_crosstemp_score[label_name] = score_mat
            rnn_model.decoder_dict[label_name] = decoder_dict
        return score_mat, decoder_dict, forw_mat
    else:
        return None, None, forw_mat


def train_single_decoder_new_data(rnn, ratio_expected=0.5, label='s1',
                                  n_samples=None, ratio_train=0.8, verbose=False,
                                  sparsity_c=0.1, bool_train_decoder=True,
                                  decoder_type='logistic_regression', save_inplace=True):
    '''Generates new data, and then trains the decoder via train_decoder()'''
    if n_samples is None:
        n_samples = rnn.info_dict['n_total']

    ## Generate data:
    if 'early_match' in rnn.info_dict.keys():
        early_match = rnn.info_dict['early_match']
    else:
        early_match = False
    tmp0, tmp1 = generate_synt_data_general(n_total=n_samples,
                                   t_delay=rnn.info_dict['t_delay'],
                                   t_stim=rnn.info_dict['t_stim'],
                                   ratio_train=ratio_train,
                                   ratio_exp=ratio_expected,
                                   noise_scale=rnn.info_dict['noise_scale'],
                                   late_s2=rnn.info_dict['late_s2'],  nature_stim=rnn.info_dict['nature_stim'],
                                   task=rnn.info_dict['type_task'], early_match=early_match)
    x_train, y_train, x_test, y_test = tmp0
    labels_train, labels_test = tmp1
    if verbose > 0:
        print('train labels ', {x: np.sum(labels_train == x) for x in np.unique(labels_train)})
    ## Train decoder:
    score_mat, decoder_dict, forward_mat = train_decoder(rnn_model=rnn, x_train=x_train, x_test=x_test,
                                           labels_train=labels_train, labels_test=labels_test,
                                           save_inplace=save_inplace, sparsity_c=sparsity_c, label_name=label,
                                           bool_train_decoder=bool_train_decoder, decoder_type=decoder_type)
    forward_mat['labels_train'] = labels_train
    forward_mat['labels_test'] = labels_test
    return score_mat, decoder_dict, forward_mat

def train_multiple_decoders(rnn_folder='models/', ratio_expected=0.5,
                            n_samples=None, ratio_train=0.8, label='s1',
                            reset_decoders=False, skip_if_already_decoded=True):
    '''train decoders for all RNNs in rnn_folder. If reset_decoders; train always. Else
    if skip_if_already_decoded is True, then skip if trained decoder already exists in RNN class.
    NB: This could be a different decoder, as they are not saved by representation or decoder type (eg log reg, lda)'''
    rnn_list = ru.get_list_rnns(rnn_folder=rnn_folder)
    for i_rnn, rnn_name in tqdm(enumerate(rnn_list)):
        ## Load RNN:
        with open(rnn_folder + rnn_name, 'rb') as f:
            rnn = pickle.load(f)
        assert rnn_name == rnn.full_path.split('/')[-1]
        if skip_if_already_decoded:
            if rnn.decoding_crosstemp_score != {}:  # already decoded before
                continue
        if reset_decoders:
            rnn.decoding_crosstemp_score = {}
            rnn.decoder_dict = {}

        _ = train_single_decoder_new_data(rnn=rnn, ratio_expected=ratio_expected,
                                          n_samples=n_samples, ratio_train=ratio_train,
                                          verbose=(i_rnn == 0), label=label) # results are saved in RNN class
        rnn.save_model(folder=rnn_folder, verbose=0, allow_name_change=False)  # save results to file
    return None

def save_pearson_corr(rnn, representation='s1', set_nans=True, save_inplace=False):
    """Compute cross correlation and save"""
    assert representation == 's1' or representation == 's1' or representation == 'go'
    assert rnn.info_dict['nature_stim'] == 'onehot' and rnn.info_dict['type_task'] in ['dmc', 'dms'], 'not implemented'
    ## get forward activity
    _, __, forw  = train_single_decoder_new_data(rnn=rnn, ratio_expected=0.5,
                                                 sparsity_c=0.1, bool_train_decoder=False)  # just gets data without training decoder

    plot_diff, labels_use_1, labels_use_2 = ru.calculate_diff_activity(forw=forw, representation=representation)
    corr_mat = np.corrcoef(plot_diff.T)
    assert corr_mat.shape[0] == corr_mat.shape[1] and corr_mat.shape[0] == 13

    if set_nans:
        if representation == 's1':
            ## first two are before stim so correlation is just rnn noise
            corr_mat[:2, :] = np.nan
            corr_mat[:, :2] = np.nan
        elif representation == 's2' or representation == 'go':
            assert False, 'ERROR: beta & mnm NaN setting not impelemtns yet'
            ## late_s2 dependence

    if hasattr(rnn, 'rep_corr_mat_dict') is False:
        rnn.rep_corr_mat_dict = {}
    rnn.rep_corr_mat_dict[representation] = corr_mat

    if save_inplace:
        assert False
        # rnn.save_model(folder=rnn_folder, verbose=0, allow_name_change=False)  # save results to file
    return corr_mat

def execute_rnn_training(nn, n_simulations, t_dict, d_dict, nature_stim='',
                        type_task='', task_name='', device='', late_s2=False,
                        train_task='', save_folder='', use_gpu=False,
                        simulated_annealing=False, ratio_exp_array=None,
                        save_state=False):
    """Create data, RNN and train using all input parameters"""
    print(f'\n-----------\nsimulation {nn}/{n_simulations}')

    ## Ensure seeds change with multi processing
    np.random.seed(np.random.get_state()[1][0] + nn)
    print('seed:', np.random.get_state()[1][0])

    if simulated_annealing is False:
        ## Generate data:
        if 'early_match' in t_dict.keys():
            early_match = t_dict['early_match']
        else:
            early_match = False
        tmp0, tmp1 = generate_synt_data_general(n_total=d_dict['n_total'], t_delay=d_dict['t_delay'], t_stim=d_dict['t_stim'],
                                    ratio_train=d_dict['ratio_train'], ratio_exp=d_dict['ratio_exp'],
                                    noise_scale=d_dict['noise_scale'], late_s2=late_s2,
                                    nature_stim=nature_stim, task=type_task, early_match=early_match)

        x_train, y_train, x_test, y_test = tmp0
        labels_train, labels_test = tmp1
        if use_gpu:
            x_train, y_train = x_train.to(device), y_train.to(device)
            x_test, y_test = x_test.to(device), y_test.to(device)
        ratio_exp_array = None
    else:
        x_train, y_train, x_test, y_test = None, None, None, None
        if ratio_exp_array is None:
            total_epochs = t_dict['n_epochs']
            assert total_epochs == 200, 'number of epochs not equal to 200 (hard set)'
            ratio_exp_array = np.zeros(total_epochs)
            ratio_exp_array[:int(0.2 * total_epochs)] = 0.5
            ratio_exp_array[int(0.2 * total_epochs):int(0.4 * total_epochs)] = np.linspace(0.5, d_dict['ratio_exp'], int(0.2 * total_epochs) + 1)[:-1]
            ratio_exp_array[int(0.4 * total_epochs):int(0.6 * total_epochs)] = d_dict['ratio_exp']
            ratio_exp_array[int(0.6 * total_epochs):int(0.8 * total_epochs)] = np.linspace(d_dict['ratio_exp'], 0.5, int(0.2 * total_epochs) + 1)[:-1]
            ratio_exp_array[int(0.8 * total_epochs):] = 0.5

    ## Initiate RNN model
    rnn = RNN_MTL(task=task_name, nature_stim=nature_stim, n_nodes=t_dict['n_nodes'])  # Create RNN class
    if use_gpu:
        rnn.to(device)
    opt = torch.optim.SGD(rnn.parameters(), lr=t_dict['learning_rate'])  # call optimiser from pytorhc
    rnn.set_info(param_dict={**d_dict, **t_dict})
    rnn.info_dict['type_task'] = type_task
    rnn.info_dict['train_task'] = train_task
    rnn.info_dict['late_s2'] = late_s2
    rnn.info_dict['simulated_annealing'] = simulated_annealing

    ## Train with BPTT
    if 'early_match' in rnn.info_dict:
        if rnn.info_dict['early_match'] is True:
            print('Starting training with early match')
    rnn = bptt_training(rnn=rnn, optimiser=opt, dict_training_params=t_dict, d_dict=d_dict,
                        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                        verbose=0, late_s2=late_s2, use_gpu=use_gpu,
                        simulated_annealing=simulated_annealing, ratio_exp_array=ratio_exp_array,
                        save_state=save_state)

    # ## Decode cross temporally
    # score_mat, decoder_dict, _ = train_single_decoder_new_data(rnn=rnn, ratio_expected=0.5,
    #                                                 n_samples=None, ratio_train=0.8, verbose=False,
    #                                                 late_s2=late_s2)

    ## Save results:
    rnn.save_model(folder=save_folder)



def init_train_save_rnn(t_dict, d_dict, n_simulations=1, use_multiproc=True,
                        n_threads=10, save_folder='models/', use_gpu=False,
                        late_s2=False, nature_stim='onehot', type_task='dmc',
                        train_task='pred_only', simulated_annealing=False, ratio_exp_array=None,
                        save_state=False):
    """Train n_simulations of RNN given argument. Uses multiprocessing by default"""
    assert type_task in ['dms', 'dmc', 'dmrs', 'dmrc']
    assert train_task in ['pred_only', 'spec_only', 'pred_spec']
    if train_task == 'pred_only':
        task_name = 'pred_only'
    elif train_task == 'spec_only':
        task_name = f'{type_task}_only'
    elif train_task == 'pred_spec':
        task_name = f'pred_{type_task}'

    np.random.seed(np.random.get_state()[1][0] + 100)

    try:
        if use_multiproc:
            pool = Pool(n_threads)
            results = pool.starmap(execute_rnn_training, zip(range(n_simulations), irep(n_simulations),
                            irep(t_dict), irep(d_dict), irep(nature_stim), irep(type_task), irep(task_name),
                            irep(device), irep(late_s2), irep(train_task), irep(save_folder), irep(False),
                            irep(simulated_annealing), irep(ratio_exp_array), irep(save_state)))
            pool.close()
        else:
            assert False, 'update this part'
            # for nn in range(n_simulations):
                # execute_rnn_training(nn=nn, n_simulations=n_simulations, t_dict=t_dict, d_dict=d_dict, nature_stim=nature_stim,
                #                     type_task=type_task, task_name=task_name, device=device, late_s2=late_s2,
                #                     train_task=train_task, save_folder=save_folder, use_gpu=use_gpu)
    except KeyboardInterrupt:
        print('KeyboardInterrupt, exit')



def summary_many(type_task_list=['dmc'], nature_stim_list=['onehot'],
                #  train_task_list=['pred_only', 'spec_only', 'pred_spec'],
                 train_task_list=['spec_only', 'pred_spec'],
                 sparsity_list=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 0], 
                 n_nodes_list=[5, 10, 20, 30, 40, 50], 
                 n_sim=1, use_gpu=False, #sweep_n_nodes=False,
                 new_gridsweep_2022=True,
                 late_s2=False, ratio_exp=0.75, simulated_annealing=False,
                 save_state=False, early_match=False):
    """Train n_simulations RNNs per set of conditions, for each set of conditions that are in arg"""
    assert (late_s2 and simulated_annealing) is False
    # assert (sweep_n_nodes and simulated_annealing) is False
    assert (early_match and simulated_annealing) is False
    assert (early_match and late_s2) is False
    if use_gpu:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # if sweep_n_nodes:
    #     assert len(train_task_list) == 1 and train_task_list[0] == 'pred_only'
    # Data parameters dictionary
    d_dict = {'n_total': 1000,  # total number of data sequences
             'ratio_train': 0.8,
             'ratio_exp': ratio_exp,  # probabilities of switching between alpha nd beta
             'noise_scale': 0.15,
             't_delay': 2,
             't_stim': 2}

    ## Set training parameters:
    t_dict = {}
    # t_dict['n_nodes'] = 20  # number of nodes in the RNN
    t_dict['learning_rate'] = 0.002  # algorithm lr
    t_dict['bs'] = 1  # batch size
    if simulated_annealing:
        t_dict['n_epochs'] = 200  # training epochs
    else:
        t_dict['n_epochs'] = 80  # training epochs
    t_dict['check_conv'] = False  # check for convergence (and abort if converged)
    t_dict['conv_rel_tol'] = 5e-4  # assess convergence by relative difference between two epochs is smaller than this
    if early_match:
        t_dict['early_match'] = True

    exp_perc = int(d_dict['ratio_exp'] * 100)
    exp_str = f'{exp_perc}{100 - exp_perc}'

    for n_nodes in tqdm(n_nodes_list):
        t_dict['n_nodes'] = n_nodes  # number of nodes in the RNN

        for sparsity in tqdm(sparsity_list):
            t_dict['l1_param'] = sparsity  # L1 regularisation in loss function
            sci_not_spars = np.format_float_scientific(t_dict['l1_param'], precision=0)
            sci_not_spars = sci_not_spars[0] + sci_not_spars[2:]  # skip dot
            print(f'N nodes: {n_nodes}, sparsity: {sparsity}')
            for nature_stim in nature_stim_list:
                for type_task in type_task_list:
                    for train_task in train_task_list:
                        # if sweep_n_nodes:
                        #     parent_folder = f'models/sweep_n_nodes/{exp_str}/{type_task}_task/{nature_stim}/sparsity_{sci_not_spars}/'
                        #     if not os.path.exists(parent_folder):
                        #         os.makedirs(parent_folder)

                        #     for n_nodes in [5, 10, 15, 20, 25]:
                        #         child_folder = f'{n_nodes}_nodes/pred_only/'
                        #         if not os.path.exists(parent_folder + child_folder):
                        #             os.makedirs(parent_folder + child_folder)

                        #         t_dict['n_nodes'] = n_nodes
                        #         init_train_save_rnn(t_dict=t_dict, d_dict=d_dict, n_simulations=n_sim,
                        #                             save_folder=parent_folder + child_folder, use_gpu=use_gpu,
                        #                             late_s2=late_s2, nature_stim=nature_stim, type_task=type_task,
                        #                             train_task='pred_only', simulated_annealing=False, ratio_exp_array=None,
                        #                             save_state=save_state)
                        # else:
                        if save_state:
                            mod_f = 'models/save_state'
                        else:
                            mod_f = 'models'
                        if late_s2:
                            parent_folder = f'{mod_f}/late_s2/{exp_str}/{type_task}_task/{nature_stim}/sparsity_{sci_not_spars}/'
                        elif simulated_annealing:
                            parent_folder = f'{mod_f}/simulated_annealing/{exp_str}/{type_task}_task/{nature_stim}/sparsity_{sci_not_spars}/'
                        elif early_match:
                            parent_folder = f'{mod_f}/early_match/{exp_str}/{type_task}_task/{nature_stim}/sparsity_{sci_not_spars}/'
                        elif new_gridsweep_2022:
                            parent_folder = f'{mod_f}/new_gridsweep_2022/{exp_str}/{type_task}_task/{nature_stim}/sparsity_{sci_not_spars}/n_nodes_{n_nodes}/'
                        else:
                            parent_folder = f'{mod_f}/{exp_str}/{type_task}_task/{nature_stim}/sparsity_{sci_not_spars}/'
                        if not os.path.exists(parent_folder):
                            os.makedirs(parent_folder)
                        for child_folder in ['pred_only', f'{type_task}_only', f'pred_{type_task}']:
                            if not os.path.exists(parent_folder + child_folder):
                                os.makedirs(parent_folder + child_folder)

                        if train_task == 'pred_only':
                            init_train_save_rnn(t_dict=t_dict, d_dict=d_dict, n_simulations=n_sim,
                                                save_folder=parent_folder + f'pred_only/', use_gpu=use_gpu,
                                                late_s2=late_s2, nature_stim=nature_stim, type_task=type_task,
                                                train_task='pred_only', simulated_annealing=simulated_annealing,
                                                ratio_exp_array=None, save_state=save_state)
                        elif train_task == 'spec_only':
                            init_train_save_rnn(t_dict=t_dict, d_dict=d_dict, n_simulations=n_sim,
                                                save_folder=parent_folder + f'{type_task}_only/', use_gpu=use_gpu,
                                                late_s2=late_s2, nature_stim=nature_stim, type_task=type_task,
                                                train_task='spec_only', simulated_annealing=simulated_annealing,
                                                ratio_exp_array=None, save_state=save_state)
                        elif train_task == 'pred_spec':
                            init_train_save_rnn(t_dict=t_dict, d_dict=d_dict, n_simulations=n_sim,
                                                save_folder=parent_folder + f'pred_{type_task}/', use_gpu=use_gpu,
                                                late_s2=late_s2, nature_stim=nature_stim, type_task=type_task,
                                                train_task='pred_spec', simulated_annealing=simulated_annealing,
                                                ratio_exp_array=None, save_state=save_state)
