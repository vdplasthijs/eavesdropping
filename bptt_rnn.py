import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle, datetime, time, os, sys, git
from tqdm import tqdm, trange
import sklearn.svm, sklearn.model_selection

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
        self.decoding_crosstemp_score = None
        self.decoder_dict = None

        ## Automatic:
        self.__datetime_created = datetime.datetime.now()
        self.__version__ = '0.1'
        self.__git_repo__ = git.Repo(search_parent_directories=True)
        self.__git_branch__ = self.__git_repo__.head.reference.name
        self.__git_commit__ = self.__git_repo__.head.object.hexsha

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

    def save_model(self, filename=None, folder=None):
        '''Export this RNN model to filename. If filename is None, it is saved  under
        a timestamp.'''
        dt = datetime.datetime.now()
        timestamp = str(dt.date()) + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)
        self.info_dict['timestamp'] = timestamp
        if filename is None or type(filename) != str:
            filename = f'rnn_{timestamp}'
        if folder is None:
            folder = 'models/'
        elif folder[-1] != '/':
            folder += '/'
        filename = folder + filename + '.data'
        file_handle = open(filename, 'wb')
        pickle.dump(self, file_handle)
        print(f'RNN model saved as {filename}')

def tau_loss(y_est, y_true, tau_array=np.array([2, 3]),
             model=None, reg_param=0.001, return_ratio_ce=False):
    '''Compute Cross Entropy of given time array tau_array, and add L1 regularisation.'''
    y_est_trunc = y_est[:, tau_array, :]  # only evaluated these time points
    y_true_trunc = y_true[:, tau_array, :]
    n_samples = y_true.shape[0]
    ce = torch.sum(-1 * y_true_trunc * torch.log(y_est_trunc)) / n_samples  # take the mean CE over samples
    reg_loss = 0
    if model is not None:  # add L1 regularisation
        params = [pp for pp in model.parameters()]  # for all weight (matrices) in the model
        for _, p_set in enumerate(params):
            reg_loss += reg_param * p_set.norm(p=1)
    total_loss = ce + reg_loss
    if return_ratio_ce is False:
        return total_loss
    elif return_ratio_ce:
        return (total_loss, (ce / total_loss))

def compute_full_pred(xdata, model):
    '''Compute forward prediction of RNN. I.e. given an input series xdata, the
    model (RNN) computes the predicted output series.'''
    if xdata.ndim == 2:
        xdata = xdata[None, :, :]

    full_pred = torch.zeros_like(xdata)  # because input & ouput have the same shape
    for kk in range(xdata.shape[0]): # loop over trials
        model.init_state()  # initiate rnn state per trial
        for tt in range(xdata.shape[1]):  # loop through time
            _, full_pred[kk, tt, :] = model(xdata[kk, tt, :])  # compute prediction at this time
    return full_pred

def bptt_training(rnn, optimiser, dict_training_params,
                  x_train, x_test, y_train, y_test, verbose=1):
    '''Training algorithm for backpropagation through time, given a RNN model, optimiser,
    dictionary with training parameters and train and test data. RNN is NOT reset,
    so continuation training is possible. Training can be aborted prematurely by Ctrl+C,
    and it will terminate correctly.'''
    ## Create data loader objects:
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=dict_training_params['bs'])

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=dict_training_params['bs'])

    prev_loss = 10  # init loss for convergence
    if 'trained_epochs' not in rnn.info_dict.keys():
        rnn.info_dict['trained_epochs'] = 0

    ## Training procedure
    if verbose > 0:
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
                for xb, yb in train_dl:  # returns torch(n_bs x n_times x n_freq)
                    full_pred = compute_full_pred(model=rnn, xdata=xb)  # predict time trace
                    loss = tau_loss(y_est=full_pred, y_true=yb, model=rnn,
                                       reg_param=dict_training_params['l1_param'],
                                       tau_array=dict_training_params['eval_times'])  # compute loss
                    loss.backward()  # compute gradients
                    optimiser.step()  # update
                    optimiser.zero_grad()   # reset

                rnn.eval()  # evaluation mode -> disable gradient tracking
                with torch.no_grad():  # to be sure
                    ## Compute losses for saving:
                    full_pred = compute_full_pred(model=rnn, xdata=x_train)
                    train_loss = tau_loss(y_est=full_pred, y_true=y_train, model=rnn,
                                       reg_param=dict_training_params['l1_param'],
                                       tau_array=dict_training_params['eval_times'])
                    rnn.train_loss_arr.append(float(train_loss.detach().numpy()))

                    full_test_pred = compute_full_pred(model=rnn, xdata=x_test)
                    test_loss, ratio = tau_loss(y_est=full_test_pred, y_true=y_test, model=rnn,
                                               reg_param=dict_training_params['l1_param'],
                                               tau_array=dict_training_params['eval_times'],
                                               return_ratio_ce=True)
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
                  dict_training_params, save_inplace=True):
    forw_mat = {'train': np.zeros((x_train.shape[0], x_train.shape[1], dict_training_params['n_nodes'])),
                'test': np.zeros((x_test.shape[0], x_test.shape[1], dict_training_params['n_nodes']))}

    rnn_model.eval()
    with torch.no_grad():
        n_times = x_train.shape[1]

        ## Forward runs:
        for kk in range(x_train.shape[0]):  # trial loop
            rnn_model.init_state()
            hidden_state =  rnn_model.state  # init state
            for tau in range(n_times):  # time loop
                hidden_state, output =  rnn_model.forward(inp=x_train[kk, tau, :],
                                                   rnn_state=hidden_state)  # propagate
                forw_mat['train'][kk, tau, :] = hidden_state.numpy()  # save hidden states

        for kk in range(x_test.shape[0]):  # trial loop
            rnn_model.init_state()
            hidden_state =  rnn_model.state
            for tau in range(n_times):  # time loop
                hidden_state, output =  rnn_model.forward(inp=x_test[kk, tau, :],
                                                   rnn_state=hidden_state)
                forw_mat['test'][kk, tau, :] = hidden_state.numpy()

        ## Train decoder
        alpha_labels = {'train': np.array([int(x[0]) for x in labels_train]),
                        'test': np.array([int(x[0]) for x in labels_test])}
        beta_labels = {'train': np.array([int(x[1]) for x in labels_train]),
                       'test': np.array([int(x[1]) for x in labels_test])}
        score_mat = np.zeros((n_times, n_times))  # T x T
        decoder_dict = {}  # save decoder per time
        for tau in range(n_times):  # train time loop
            decoder_dict[tau] = sklearn.svm.LinearSVC(C=1e-6)  # define SVM
            decoder_dict[tau].fit(X=forw_mat['train'][:, tau, :],
                                  y=alpha_labels['train'])  # train SVM
            for tt in range(n_times):  # test time loop
                score_mat[tau, tt] = decoder_dict[tau].score(X=forw_mat['test'][:, tt, :],
                                                             y=alpha_labels['test'])  # evaluate
    if save_inplace:
        rnn_model.decoding_crosstemp_score = score_mat
        rnn_model.decoder_dict = decoder_dict
    return score_mat, decoder_dict

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
            rnn = RNN(n_stim=d_dict['n_freq'], n_nodes=t_dict['n_nodes'])  # Create RNN class
            opt = torch.optim.SGD(rnn.parameters(), lr=t_dict['learning_rate'])  # call optimiser from pytorhc
            rnn.set_info(param_dict={**d_dict, **t_dict})

            ## Train with BPTT
            rnn = bptt_training(rnn=rnn, optimiser=opt, dict_training_params=t_dict,
                        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                        verbose=0)

            ## Decode cross temporally
            score_mat, decoder_dict = train_decoder(rnn_model=rnn, x_train=x_train, x_test=x_test,
                                               labels_train=labels_train, labels_test=labels_test,
                                               dict_training_params=t_dict, save_inplace=True)

            ## Save results:
            rnn.save_model(folder=save_folder)
        return rnn  # return latest
    except KeyboardInterrupt:
        print('KeyboardInterrupt, exit')


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

def aggregate_score_mats(model_folder='models/', check_info_dict=True):
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
            mat_shape = model.decoding_crosstemp_score.shape
            assert len(mat_shape) == 2  # 2D matrix
            agg_score_mat = np.zeros((len(list_models), mat_shape[0], mat_shape[1]))
        else:  #TODO: fix check
            if check_info_dict: # check if dicts with info are equal
                for cp in check_params:
                    assert (model.info_dict[cp] == prev_model.info_dict[cp]), f'AssertionError: models {prev_name} and {mn} are different'
                assert (model.info_dict['eval_times'] == prev_model.info_dict['eval_times']).all(), f'AssertionError: models {prev_name} and {mn} are different'

        agg_score_mat[ii, :, :] = model.decoding_crosstemp_score  # add score matrix
        prev_model = model
        prev_name = mn
        f.close()
    return agg_score_mat
