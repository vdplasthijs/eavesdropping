# @Author: Thijs L van der Plas <TL>
# @Date:   2020-05-05
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Filename: rot_utilities.py
# @Last modified by:   thijs
# @Last modified time: 2020-05-22



import numpy as np
import pickle, os
import pandas as pd
import bptt_rnn_mtl as bpm

def angle_vecs(v1, v2):
    """Compute angle between two vectors with cosine similarity.

    Parameters
    ----------
    v1 : np.array
        vector 1.
    v2 : np.array
        vector 2.

    Returns
    -------
    deg: float
        angle in degrees .

    """
    assert v1.shape == v2.shape
    v1, v2 = np.squeeze(v1), np.squeeze(v2)
    tmp = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    rad = np.arccos(tmp)
    deg = rad * 360 / (2 * np.pi)
    return deg


def angle_sensory_memory(forw,  ol=None):

    ind_sens_alpha_1 = np.array([x[0] == '1' for x in forw['labels_train']])  # alpha == 1
    ind_sens_alpha_2 = np.array([x[0] == '2' for x in forw['labels_train']])  # alpha == 2

    ind_sens_beta_1 = np.array([x[1] == '1' for x in forw['labels_train']])  # beta == 1
    ind_sens_beta_2 = np.array([x[1] == '2' for x in forw['labels_train']])  # beta == 2

    plot_diff_sens_alpha = (forw['train'][ind_sens_alpha_1, :, :].mean(0) -
                            forw['train'][ind_sens_alpha_2, :, :].mean(0))

    plot_diff_sens_beta = (forw['train'][ind_sens_beta_1, :, :].mean(0) -
                           forw['train'][ind_sens_beta_2, :, :].mean(0))

    if ol is None:
        # ol = opt_leaf(plot_diff_sens_alpha, dim=1)  # optimal leaf sorting
        ol = np.arange(plot_diff_sens_alpha.shape[1])
    plot_diff_sens_alpha = plot_diff_sens_alpha[:, ol]
    plot_diff_sens_beta = plot_diff_sens_beta[:, ol]

    n_timepoints = plot_diff_sens_beta.shape[0]
    angle_alpha_beta = np.array([angle_vecs(v1=plot_diff_sens_alpha[t, :],
                                            v2=plot_diff_sens_beta[t, :]) for t in range(n_timepoints)])

    av_alpha_act = np.sum(plot_diff_sens_alpha, 0)
    av_beta_act = np.sum(plot_diff_sens_beta, 0)
    return angle_alpha_beta, (av_alpha_act, av_beta_act)

def find_max_or_min(array, dimension=0):
    assert array.ndim == 2

    amin = np.amin(array, axis=dimension)
    amax = np.amax(array, axis=dimension)
    ind_amax = (amax >= np.abs(amin))
    minmax = amin.copy()
    minmax[ind_amax] = amax[ind_amax]
    return minmax

def find_stable_switch_neurons_activity(forw_mat, diff_th=1, n_tp_sign=2, tt='test'):
    """Identify switch and stable cells, and return numbers & indices of these.

    Parameters
    ----------
    forw_mat : dict
        activity matrix .
    diff_th : float, default: 1
        threshold that determins when tuning is significant.
    n_tp_sign : int, default; 2
        number of time points for which tuning has to be significant (no particular order/adjacency).
    tt : str, 'train' or 'test' (default)
        trial type.

    Returns
    -------
    2-tuple, 2-tuple
        (number of stable cells, number of switch cells), (inds stable cells, inds switch cells)
    """
    labels_use_1 = np.array([x[0] == '1' for x in forw_mat['labels_' + tt]])
    labels_use_2 = np.array([x[0] == '2' for x in forw_mat['labels_' + tt]])
    mean_response_1 = forw_mat[tt][labels_use_1, :, :].mean(0)
    mean_response_2 = forw_mat[tt][labels_use_2, :, :].mean(0)
    diff_mat = mean_response_1 - mean_response_2  # differen ce between mean response of 1X and 2X
    assert diff_mat.shape == (17, 20), 'activity matrix has a different shape than 17x20' # times x neurons

    arr_1_code = np.sum(diff_mat > diff_th, 0)  #  how many time points difference larger than threhsold => 1 coding
    arr_2_code = np.sum(diff_mat < -1 * diff_th, 0) # 2 coding
    bool_1_code = arr_1_code >= n_tp_sign  # at least this many time points
    bool_2_code = arr_2_code >= n_tp_sign
    switch_neurons = np.logical_and(bool_1_code, bool_2_code)  # switch if they contain both
    assert switch_neurons.shape[0] == 20
    stable_neurons = np.logical_and(np.logical_or(bool_1_code, bool_2_code),
                                    np.logical_not(switch_neurons))  # stable if they contain either and not switch
    n_stable_neurons = np.sum(stable_neurons)
    n_switch_neurons = np.sum(switch_neurons)
    stable_inds = np.where(stable_neurons == True)[0]
    switch_inds = np.where(switch_neurons == True)[0]
    return (n_stable_neurons, n_switch_neurons), (stable_inds, switch_inds)

def connect_mnm_stsw(rnn, stable_inds=np.array([]), switch_inds=np.array([]),
                     weight_threshold=0.1, verbose=0, proj_type='threshold'):
    assert rnn.lin_output.out_features > rnn.n_stim, 'this RNN does not have M and NM output neurons'
    assert rnn.lin_output.out_features == 10
    output_ind = {'match': 8, 'nonmatch': 9}  # inds of output neurons
    st_sw_neurons = {'stable': np.array(stable_inds), 'switch': np.array(switch_inds)}
    sign_weight_types = {kk: {'stable': 0, 'switch': 0} for kk in output_ind.keys()} # to store the counts
    output_weight_mat = [x for x in rnn.lin_output.parameters()][0].detach().numpy()  # get parameters from generator object
    assert output_weight_mat.shape == (rnn.lin_output.out_features, rnn.n_nodes)
    for key, ind in output_ind.items():  # loop through M and NM
        out_proj = output_weight_mat[ind, :]  #output weights

        if proj_type == 'gradual':
            for neuron_type, type_inds in st_sw_neurons.items():
                type_proj = np.abs(out_proj[type_inds])
                normaliser = np.sum(np.abs(out_proj[np.intersect1d(st_sw_neurons['stable'], st_sw_neurons['switch'])]))
                if normaliser == 0:
                    normaliser = 1
                sign_weight_types[key][neuron_type] = np.sum(type_proj) / normaliser
        elif proj_type == 'threshold':
            sign_neurons = np.where(np.abs(out_proj) > weight_threshold)[0]  # significant output neurons
            if verbose > 0:
                print(f'{key}, sign neurons: {sign_neurons}, neuron types: {st_sw_neurons}')
            for neuron_type in st_sw_neurons.keys():
                mutual_neurons = np.intersect1d(sign_neurons, st_sw_neurons[neuron_type])  # find intersection with stablea nd switch indices
                sign_weight_types[key][neuron_type] = len(mutual_neurons) # save number of such neurons
    return sign_weight_types

def create_color_mat(x, c):
    c_mat = np.zeros((len(x) - 1, 4))
    if c == 'green':
        c_mat[:, 1] = 0.5
    elif c == 'm':
        c_mat[:, (0, 2)] = 0.746
    elif c == 'k':
        pass
    else:
        c_mat[:, 0] = 0.8
    c_mat[:, 3] = np.linspace(0.17, 1, len(x) - 1)
    return c_mat


def rmse_matrix_symm(matrix, subtract=0.5):
    '''Return RMSE of the two upper-/lower-diagonal triangular halves'''
    n, _ = np.shape(matrix)
    rmse = 0
    low_tri_sum = 0
    n_el = 0
    matrix = matrix - subtract
    shuffled_matrix = matrix.copy()
    np.random.shuffle(shuffled_matrix)
    assert n == _
    for i_row in range(1, n):
        for i_col in range(i_row):
            low_item = matrix[i_row, i_col]
            up_item = matrix[i_col, i_row]
            low_tri_sum += np.abs(shuffled_matrix[i_row, i_col] - shuffled_matrix[i_col, i_row])
            rmse += np.abs(low_item - up_item)
            n_el += 1
    rmse /= n_el
    rmse = np.sqrt(rmse)
    low_tri_sum /= n_el
    return (rmse, low_tri_sum)

def load_rnn(rnn_name):
    with open(rnn_name, 'rb') as f:
        rnn = pickle.load(f)
    return rnn

def make_df_network_size(rnn_folder):
    rnn_names = [x for x in os.listdir(rnn_folder) if x[-5:] == '.data']
    df_data = pd.DataFrame({x: np.zeros(len(rnn_names)) for x in ['n_nodes', 'min_test_perf']})
    for i_rnn, rnn_name in enumerate(rnn_names):
        rnn = load_rnn(rnn_name=os.path.join(rnn_folder, rnn_name))
        df_data['n_nodes'].iat[i_rnn] = rnn.info_dict['n_nodes']
        ind_min = np.argmin(rnn.test_loss_arr)
        df_data['min_test_perf'].iat[i_rnn] = rnn.test_loss_arr[ind_min] * rnn.test_loss_ratio_ce[ind_min]
    df_data = df_data.sort_values('n_nodes')
    return df_data

def labels_to_mnm(labels):
    if type(labels) == str:  # just one label
        match = (labels[0] == labels[1])
        return np.array([int(match), int(not match)])
    else:
        n_labels = len(labels)
        match_array = np.zeros((n_labels, 2))
        for i_label, label in enumerate(labels):
            match_array[i_label, 0] = (label[0] == label[1])
        match_array[:, 1] = np.logical_not(match_array[:, 0])
        return match_array

def get_train_test_diag():
    tmp_train, tmp_test = (np.array([8, 9, 10, 9, 10, 11, 10, 11, 12, 11, 12, 13, 12, 13, 14, 13, 14, 15, 14, 15, 16, 15, 16, 16]),
                           np.array([4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 12]))
    return (tmp_train, tmp_test)

def rotation_index(mat, times_early=[4], times_late=[6]):
    ## currently; assume square of early/late. Alternatively, ask for 2 tuples
    times_early = np.array(times_early)
    times_late = np.array(times_late)
    assert times_early.ndim == 1 and times_late.ndim == 1
    ## assume symmetric matix; otherwise it would be wise to build a test to make sure early/late : x/y axis is correct
    mat_nonnan = mat[np.logical_not(np.isnan(mat))]
    assert np.allclose(mat_nonnan, mat_nonnan.T, atol=1e-5), 'ERROR: matrix is not symmetric'
    n_te = len(times_early)
    n_tl = len(times_late)
    elements_cross = np.zeros(int(n_te * n_tl))
    elements_early = np.zeros(n_te + n_tl)

    i_cross = 0
    for i_tau, tau in enumerate(times_early):
        elements_early[i_tau] = mat[tau, tau]  # auto temp during early
        for i_t, t in enumerate(times_late):
            elements_cross[i_cross] = mat[tau, t]  # get square
            i_cross += 1
    for i_t, t in enumerate(times_late):
        assert elements_early[i_tau + i_t + 1] == 0  # not used
        assert (i_tau + i_t + 1) < len(elements_early)
        elements_early[i_tau + i_t] = mat[t, t]
    rot = np.mean(elements_cross) / np.mean(elements_early)
    return rot

def compute_learning_index(rnn_folder=None, list_loss=['pred'], normalise_start=False,
                           method='integral'):
    list_rnns = [x for x in os.listdir(rnn_folder) if x[-5:] == '.data']
    if len(list_rnns) > 20:
        list_rnns = list_rnns[:20]
        print(f'list rnns shortened for {rnn_folder}')
    n_rnn = len(list_rnns)
    for i_rnn, rnn_name in enumerate(list_rnns):
        rnn = load_rnn(os.path.join(rnn_folder, rnn_name))
        if i_rnn == 0:
            n_epochs = rnn.info_dict['n_epochs']
            conv_dict = {key: np.zeros((n_rnn, n_epochs)) for key in list_loss}
        else:
            assert rnn.info_dict['n_epochs'] == n_epochs, 'number of epochs not equal, this is not implemented explicitly when computing the integral'
        for key in list_loss:
            assert key in rnn.test_loss_split.keys(), f'{key} not saved for {rnn}'
            arr = rnn.test_loss_split[key]
            conv_dict[key][i_rnn, :] = arr.copy()
    learn_eff = {}
    for key in list_loss:
        mat = conv_dict[key]
        if normalise_start:
            assert False, 'normalise start not implemented'
            mat = mat / np.mean(mat[:, 0])#[:, np.newaxis]
        # plot_arr = np.mean(mat, 0)
        if method == 'integral':
            learn_eff[key] = np.mean(mat, 1)  # sum = integral, mean = divide by n epochs
        elif method == 'mean_integral':
            learn_eff[key] = np.zeros(mat.shape[0]) + np.mean(np.mean(mat, 1))
        elif method == 'final_loss':
            learn_eff[key] = np.mean(mat[:, -5:], 1)
        elif method == 'half_time':
            half_time_ar = np.zeros(mat.shape[0])
            for i_rnn in range(mat.shape[0]):
                # half_time_ar[i_rnn] = np.argmin(np.abs(mat[i_rnn, :] - 0.5))
                half_time_ar[i_rnn] = np.argmin(np.diff(mat[i_rnn, :]))
            learn_eff[key] = half_time_ar
        elif method == 'argmin_gradient':
            min_grad_arr = np.zeros(mat.shape[0])
            for i_rnn in range(mat.shape[0]):
                grad = np.gradient(mat[i_rnn, :])
                min_grad_arr[i_rnn] = np.argmin(grad)
            learn_eff[key] = min_grad_arr
        else:
            assert False, f'Method {method} not implemented!'
        assert len(learn_eff[key]) == len(list_rnns)
    return learn_eff

def calculate_all_learning_eff_indices(task_list=['dmc', 'dms'], ratio_exp_str='7525',
                                       nature_stim_list=['onehot', 'periodic'], method='integral',
                                       sparsity_str_list = ['0e+00', '1e-06', '5e-06', '1e-05', '5e-05', '1e-04', '5e-04', '1e-03', '5e-03', '1e-02', '1e-01']):

    n_sim = 20
    n_loss_functions_per_sim = 4
    n_data = len(task_list) * len(nature_stim_list) * len(sparsity_str_list) * n_loss_functions_per_sim * n_sim

    learn_eff_dict = {**{x: np.zeros(n_data, dtype='object') for x in ['task', 'nature_stim', 'loss_comp', 'setting']},
                      **{x: np.zeros(n_data) for x in ['learning_eff' ,'sparsity']}}

    i_conf = 0
    for i_task, task in enumerate(task_list):
        for i_nat, nature_stim in enumerate(nature_stim_list):
            for i_spars, sparsity_str in enumerate(sparsity_str_list):
                spars = float(sparsity_str)
                base_folder = f'models/{ratio_exp_str}/{task}_task/{nature_stim}/sparsity_{sparsity_str}/'
                if not os.path.exists(base_folder):
                    # print(base_folder, 'does not exist', nature_stim)
                    continue
                folders_dict = {}
                folders_dict['pred_only'] = base_folder + 'pred_only/'
                folders_dict[f'{task}_only'] = base_folder + f'{task}_only/'
                folders_dict[f'pred_{task}'] = base_folder + f'pred_{task}/'
                for key, folder_rnns in folders_dict.items():
                    list_keys = key.split('_')
                    if 'only' in list_keys:
                        list_keys.remove('only')
                        suffix = '_single'
                    else:
                        suffix = '_multi'
                    learn_eff = compute_learning_index(rnn_folder=folder_rnns,
                                                          list_loss=list_keys,
                                                          method=method)
                    for loss_comp in list_keys:
                        for le in learn_eff[loss_comp]:
                            learn_eff_dict['task'][i_conf] = task
                            learn_eff_dict['nature_stim'][i_conf] = nature_stim
                            learn_eff_dict['sparsity'][i_conf] = spars
                            # learn_eff_dict['mean_learning_eff'][i_conf] = np.mean(learn_eff[loss_comp])
                            # learn_eff_dict['std_learning_eff'][i_conf] = np.std(learn_eff[loss_comp])
                            learn_eff_dict['learning_eff'][i_conf] = le
                            learn_eff_dict['setting'][i_conf] = suffix[1:]
                            learn_eff_dict['loss_comp'][i_conf] = loss_comp + suffix
                            i_conf += 1
    learn_eff_df = pd.DataFrame(learn_eff_dict)
    if i_conf > n_data:
        assert False, 'dictionary was not large enough'
    elif i_conf < n_data:
        learn_eff_df = learn_eff_df[:i_conf]
        print(f'Cutting of DF because of empty rows')


    return learn_eff_df

def two_digit_sci_not(x):
    sci_not_spars = np.format_float_scientific(x, precision=0)
    sci_not_spars = sci_not_spars[0] + sci_not_spars[2:]  # skip dot
    return sci_not_spars

def count_datasets_sparsity_sweep(super_folder='/home/thijs/repos/rotation/models/7525'):

    task_folders = os.listdir(super_folder)
    task_nat_folder_dict = {}
    sparsity_list = []

    ## Explore which tasks, nat & spars are present
    for task_folder in task_folders:
        nat_list = os.listdir(os.path.join(super_folder, task_folder))
        for nat in nat_list:
            key = task_folder.split('_')[0] + '_' + nat
            task_nat_folder_dict[key] = os.path.join(super_folder, task_folder, nat)

            sparsity_folder_list = os.listdir(task_nat_folder_dict[key])
            sparsity_list = sparsity_list + [float(x.split('_')[1]) for x in sparsity_folder_list]

    sparsity_arr = np.sort(np.unique(np.array(sparsity_list)))
    n_ds_dict = {**{'sparsity': sparsity_arr, 'sparsity_str': [two_digit_sci_not(x) for x in sparsity_arr]},
                 **{task_nat: np.zeros_like(sparsity_arr, dtype='int') for task_nat in task_nat_folder_dict.keys()}}  # dictionary to save numberof datasets

    ## Find number of ds per setting, and add 0 for settings taht are not present
    for task_nat, task_nat_folder in task_nat_folder_dict.items():
        for i_spars, float_spars in enumerate(sparsity_arr):
            sparsity_folder = 'sparsity_' + two_digit_sci_not(float_spars)
            task_nat_spars_folder = os.path.join(task_nat_folder, sparsity_folder)
            if os.path.exists(task_nat_spars_folder):
                tt_folders = os.listdir(task_nat_spars_folder)  # [pred_only, dmc_only etc]
                n_ds_arr = np.zeros(len(tt_folders))
                for i_tt, tt_folder in enumerate(tt_folders):
                    n_ds_arr[i_tt] = len([x for x in os.listdir(os.path.join(task_nat_spars_folder, tt_folder)) if x[-5:] == '.data'])
                if len(np.unique(n_ds_arr)) != 1:
                    print(f'{task_nat_spars_folder} does not have equal number of trainings: {np.unique(n_ds_arr)}')
                n_ds_dict[task_nat][i_spars] = np.mean(n_ds_arr)  #because they are all the same anyway
            else:
                n_ds_dict[task_nat][i_spars] = 0

    return pd.DataFrame(n_ds_dict)
    # print(n_ds_dict)

def ensure_corr_mat_exists(rnn, representation='s1'):
    ## if not pre-calculated, then calculate now:
    if hasattr(rnn, 'rep_corr_mat_dict') is False:
        bpm.save_pearson_corr(rnn=rnn, representation=representation)
    elif representation not in rnn.rep_corr_mat_dict.keys():
        bpm.save_pearson_corr(rnn=rnn, representation=representation)
