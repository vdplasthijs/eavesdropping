# @Author: Thijs van der Plas <TL>
# @Date:   2020-05-15
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Filename: plot_routines.py
# @Last modified by:   thijs
# @Last modified time: 2020-05-21



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bptt_rnn as bp
import scipy.cluster, scipy.spatial

## Some labels needed by a lot of funtions:
single_time_labels = ['0', 'A', '0', 'B', '0', 'C', '0', 'D', '0']
double_time_labels = []
double_time_labels_half = []
for stl in single_time_labels:
    double_time_labels.append(stl)
    double_time_labels.append(stl)
    double_time_labels_half.append(stl)
    double_time_labels_half.append('')
double_time_labels_blank = [x.replace('0', '') for x in double_time_labels]
single_time_labels_blank = [x.replace('0', '') for x in single_time_labels]
assert len(double_time_labels_half) == len(double_time_labels)
freq_labels = ['0', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D']

def plot_weights(rnn_layer, ax=None, title='weights', xlabel='',
                 ylabel='', xticklabels=None, yticklabels=None,
                 weight_order=None):
    '''Plot a weight matrix; given a RNN layer, with zero-symmetric clipping.'''
    if ax is None:
        ax = plt.subplot(111)
    weights = [x for x in rnn_layer.parameters()][0].detach().numpy()
    if weight_order is not None and weights.shape[0] == len(weight_order):
        weights = weights[weight_order, :]
    if weight_order is not None and weights.shape[1] == len(weight_order):
        weights = weights[:, weight_order]
    cutoff = np.percentile(np.abs(weights), 95)
    sns.heatmap(weights, ax=ax, cmap='PiYG', vmax=cutoff, vmin=-1 * cutoff)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title(title, weight='bold');
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    return ax

def plot_all_UWV(rnn_model, freq_labels='', weight_order=None):
    '''Plot the 3 weight matrices  U, W and V.'''
    fig, ax_w = plt.subplots(1, 3)
    plot_weights(ax=ax_w[0], rnn_layer=rnn_model.lin_input,
                title='U - Input stimulus-neuron weights',
                xticklabels=freq_labels, ylabel='Neuron',
                weight_order=weight_order, xlabel='Input')


    plot_weights(ax=ax_w[1], rnn_layer=rnn_model.lin_feedback,
                 title='W - Feedback neuron-neuron weights',
                 ylabel='Neuron', xlabel='Neuron',
                 weight_order=weight_order)

    plot_weights(ax=ax_w[2], rnn_layer=rnn_model.lin_output,
                 title='V - Ouput neuron-prediction weights',
                 yticklabels=freq_labels, xlabel='Neuron',
                 ylabel='Output', weight_order=weight_order)
    return (fig, ax_w)

def opt_leaf(w_mat, dim=0):
    '''create optimal leaf order over dim, of matrix w_mat. if w_mat is not an
    np.array then its assumed to be a RNN layer. see also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html#scipy.cluster.hierarchy.optimal_leaf_ordering'''
    if type(w_mat) != np.ndarray:  # assume it's an rnn layer
        w_mat = [x for x in w_mat.parameters()][0].detach().numpy()
    assert w_mat.ndim == 2
    if dim == 1:  # transpose to get right dim in shape
        w_mat = w_mat.T
    dist = scipy.spatial.distance.pdist(w_mat, metric='euclidean')  # distanc ematrix
    link_mat = scipy.cluster.hierarchy.ward(dist)  # linkage matrix
    opt_leaves = scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.optimal_leaf_ordering(link_mat, dist))
    return opt_leaves

def plot_train_test_perf(rnn_model, ax=None):
    '''Plot train and test loss as function of epoch.'''
    if ax is None:
        ax = plt.subplot(111)
    ax.plot(rnn_model.train_loss_arr, label='train', linewidth=3)
    ax.plot(rnn_model.test_loss_arr, label='test', linewidth=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel("Loss"); ax.legend();
    return ax

def plot_decoder_crosstemp_perf(score_matrix, ax=None, ticklabels=''):
    '''Plot matrix of cross temporal scores for decoding'''
    if ax is None:
        ax = plt.subplot(111)
    # cmap_hm = sns.diverging_palette(145, 280, s=85, l=25, n=20)
    cmap_hm = 'BrBG'
    # cmap_hm = 'Greys'
    hm = sns.heatmap(score_matrix, cmap=cmap_hm, xticklabels=ticklabels,
                           yticklabels=ticklabels, ax=ax, vmin=0,# vmax=1,
                           linewidths=0.1, linecolor='k')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_ylabel('Training time tau'); ax.set_xlabel('Testing time t')
    ax.set_title('Cross temporal decoding score');
    return ax

def plot_dynamic_decoding_axes(rnn, ticklabels=''):
    '''Plot the decoding axis w for each time point; and the diagonal auto-decoding
    accuracy on top. Returns these two axes. '''
    # if ax is None:
    #     ax = plt.subplot(111)

    decoder_axes = np.zeros((rnn.decoder_dict[0].coef_.size, len(rnn.decoder_dict)))
    for k, v in rnn.decoder_dict.items():
        decoder_axes[:, k] = v.coef_
    cutoff_w = np.percentile(np.abs(decoder_axes), 95)

    ax_dec_diag = plt.subplot(3, 1, 1)
    ax_dec_diag.plot(np.diag(rnn.decoding_crosstemp_score), linewidth=3,
                             linestyle='-', marker='.', markersize=10, color='k', alpha=0.6)
    ax_dec_diag.set_ylabel('Score')
    ax_dec_diag.set_title('Decoding performance (t = tau)')
    ax_dec_diag.set_xticks(np.arange(len(np.diag(rnn.decoding_crosstemp_score))));
    ax_dec_diag.set_xticklabels(ticklabels);
    ax_dec_diag.set_xlim([-0.5, len(np.diag(rnn.decoding_crosstemp_score)) - 0.5])

    plt.subplot(3, 1, (2, 3))
    ax_dec_w = sns.heatmap(decoder_axes, xticklabels=ticklabels,
                          vmin=-1 * cutoff_w, vmax=cutoff_w, cmap='PiYG', cbar=False)
    bottom, top = ax_dec_w.get_ylim()
    ax_dec_w.set_ylim(bottom + 0.5, top - 0.5)
    ax_dec_w.set_xlabel('Time t'); ax_dec_w.set_ylabel('Neuron');
    return ax_dec_diag, ax_dec_w

def plot_example_trial(trial, ax=None, yticklabels=freq_labels,
                       xticklabels=double_time_labels_blank[1:],
                       vmin=0, vmax=1):
    '''Plot 1 example trial'''
    if ax is None:
        ax = plt.subplot(111)

    sns.heatmap(trial.T, yticklabels=yticklabels,
            xticklabels=xticklabels, ax=ax, vmin=vmin, vmax=vmax)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Stimulus vector')
    return ax

def plot_time_trace_1_decoding_neuron(rnn, n_neuron=3, ax=None):
    if ax is None:
        ax = plt.subplot(111)
    n_tp = len(rnn.decoder_dict)
    time_trace_pos = np.zeros(n_tp)
    time_trace_neg = np.zeros(n_tp)
    for i_tp in range(n_tp):
        time_trace_pos[i_tp] = np.clip(rnn.decoder_dict[i_tp].coef_[0][n_neuron], a_min=0, a_max=np.inf)
        time_trace_neg[i_tp] = -1 * np.clip(rnn.decoder_dict[i_tp].coef_[0][n_neuron], a_max=0, a_min=-1 * np.inf)
    ax.plot(time_trace_pos, linewidth=3, c='green')
    ax.plot(time_trace_neg, linewidth=3, c='m')
    ax.set_xlabel('time'); ax.set_ylabel('Decoding strenght')
    return ax
