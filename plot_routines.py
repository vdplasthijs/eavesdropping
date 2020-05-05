import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import BPTTRNN as bp
import scipy.cluster, scipy.spatial

def plot_weights(ax, rnn_layer, title='weights', xlabel='',
                 ylabel='', xticklabels=None, yticklabels=None,
                 weight_order=None):
    '''Plot a weight matrix; given a RNN layer, with zero-symmetric clipping.'''
    weights = [x for x in rnn_layer.parameters()][0].detach().numpy()
    if weight_order is not None and weights.shape[0] == len(weight_order):
        weights = weights[weight_order, :]
    if weight_order is not None and weights.shape[1] == len(weight_order):
        weights = weights[:, weight_order]
    cutoff = np.percentile(np.abs(weights), 95)
    sns.heatmap(weights, ax=ax, cmap='PiYG', vmax=cutoff, vmin=-1 * cutoff)
    ax.set_title(title);
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
    opt_leaves =scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.optimal_leaf_ordering(link_mat, dist))
    return opt_leaves

def plot_train_test_perf(rnn_model, ax):
    '''Plot train and test loss as function of epoch.'''
    ax.plot(rnn_model.train_loss_arr, label='train', linewidth=3)
    ax.plot(rnn_model.test_loss_arr, label='test', linewidth=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel("Loss"); ax.legend();
    return ax
