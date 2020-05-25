# @Author: Thijs L van der Plas <TL>
# @Date:   2020-05-05
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Filename: rot_utilities.py
# @Last modified by:   thijs
# @Last modified time: 2020-05-22



import numpy as np

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

def find_stable_switch_neurons_decoder(rnn):
    decoder_axes = np.zeros((rnn.decoder_dict[0].coef_.size, len(rnn.decoder_dict)))
    for k, v in rnn.decoder_dict.items():
        decoder_axes[:, k] = v.coef_
    n_nodes, n_times = decoder_axes.shape
    period_1 = np.arange(4, 8)  # 00BB
    period_2 = np.arange(10, 13)  # CCB

    minmax_1 = find_max_or_min(decoder_axes[:, period_1], dimension=1) # minmax over 1st period for each neuron
    minmax_2 = find_max_or_min(decoder_axes[:, period_2], dimension=1)  # minmax over 2nd period
    nz_inds = np.intersect1d(np.where(np.abs(minmax_1) > 0)[0],
                             np.where(np.abs(minmax_2) > 0)[0])  # use sparsity, only select non zero cells
    print(minmax_1)
    sign_match = np.sign(minmax_1[nz_inds]) == np.sign(minmax_2[nz_inds])
    n_stable_neurons = np.sum(sign_match == True)
    n_switch_neurons = np.sum(sign_match == False)
    stable_inds = np.where(sign_match == True)[0]
    switch_inds = np.where(sign_match == False)[0]
    assert n_stable_neurons + n_switch_neurons == len(sign_match)

    return (n_stable_neurons, n_switch_neurons), (stable_inds, switch_inds)


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
