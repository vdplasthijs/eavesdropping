# @Author: Thijs L van der Plas <TL>
# @Date:   2020-05-05
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Filename: rot_utilities.py
# @Last modified by:   thijs
# @Last modified time: 2020-05-22



import numpy as np
import pickle, os
import pandas as pd

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

def find_max_or_min(array, dimension=0):
    assert array.ndim == 2

    amin = np.amin(array, axis=dimension)
    amax = np.amax(array, axis=dimension)
    ind_amax = (amax >= np.abs(amin))
    minmax = amin.copy()
    minmax[ind_amax] = amax[ind_amax]
    return minmax

def find_stable_switch_neurons_activity(forw_mat, diff_th=1):
    tt = 'train'
    n_tp_sign = 2
    labels_use_1 = np.array([x[0] == '1' for x in forw_mat['labels_' + tt]])
    labels_use_2 = np.array([x[0] == '2' for x in forw_mat['labels_' + tt]])
    mean_response_1 = forw_mat[tt][labels_use_1, :, :].mean(0)
    mean_response_2 = forw_mat[tt][labels_use_2, :, :].mean(0)
    diff_mat = mean_response_1 - mean_response_2  # differen ce between mean response of 1X and 2X
    assert diff_mat.shape == (17, 20) # times x neurons

    arr_1_code = np.sum(diff_mat > diff_th, 0)  # different larger than threhsold => 1 coding
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
