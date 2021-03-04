# @Author: Thijs van der Plas <TL>
# @Date:   2020-05-15
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Filename: plot_routines.py
# @Last modified by:   thijs
# @Last modified time: 2020-05-21



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar as mpl_colorbar
import seaborn as sns
import pickle, os, sys
import scipy.cluster, scipy.spatial
import bptt_rnn as bp
import rot_utilities as ru
import pandas as pd
from cycler import cycler
## Create list with standard colors:
color_dict_stand = {}
for ii, x in enumerate(plt.rcParams['axes.prop_cycle']()):
    color_dict_stand[ii] = x['color']
    if ii > 8:
        break  # after 8 it repeats (for ever)
plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))

def set_fontsize(font_size=12):
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.autolimit_mode'] = 'data' # default: 'data'
    params = {'legend.fontsize': font_size,
             'axes.labelsize': font_size,
             'axes.titlesize': font_size,
             'xtick.labelsize': font_size,
             'ytick.labelsize': font_size}
    plt.rcParams.update(params)

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
freq_labels_sub = [r"$0$", r"$A_1$", r"$A_2$", r"$B_1$", r"$B_2$", r"$C_1$", r"$C_2$", r"$D$"]

def clip_axes_tick(ax, clipx=True, clipy=True):
    if clipx:
        xticks = [tick for tick in ax.get_xticks()]
        plt.xlim(xticks[0], xticks[-1])
    if clipy:
        yticks = [tick for tick in ax.get_yticks()]
        plt.ylim(yticks[0], yticks[-1])
    return ax

def plot_weights(rnn_layer, ax=None, title='weights', xlabel='',
                 ylabel='', xticklabels=None, yticklabels=None,
                 weight_order=None):
    '''Plot a weight matrix; given a RNN layer, with zero-symmetric clipping.'''
    if ax is None:
        ax = plt.subplot(111)
    weights = [x for x in rnn_layer.parameters()][0].detach().numpy()
    if weight_order is not None and weights.shape[0] == len(weight_order):
        weights = weights[weight_order, :]
    elif weight_order is not None and weights.shape[1] == len(weight_order):
        weights = weights[:, weight_order]
    else:
        print(f'weight order not implemented because the size is different. weights: {weight.shape}, order: {weight_order.shape}')
    cutoff = np.percentile(np.abs(weights), 95)
    sns.heatmap(weights, ax=ax, cmap='PuOr', vmax=cutoff, vmin=-1 * cutoff)
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

def plot_train_test_perf(rnn_model, ax=None, train=True, test=True):
    '''Plot train and test loss as function of epoch.'''
    if ax is None:
        ax = plt.subplot(111)
    if train:
        ax.plot(rnn_model.train_loss_arr, label='train', linewidth=3, color='k', linestyle=':')
    if test:
        ax.plot(rnn_model.test_loss_arr, label='test', linewidth=3, color='k')
    ax.set_xlabel('Epoch'); ax.set_ylabel("Loss");
    if train and test:
        ax.legend(frameon=False);
    return ax

def plot_split_perf(rnn_name=None, rnn_folder=None, ax_top=None, ax_bottom=None,
                    plot_top=True, plot_bottom=True, list_top=None, lw=3, plot_total=True,
                    label_dict_keys = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'L1': r'$L_1$',
                                       'MNM': 'M/NM', '0': '0', 'C1': r'$C_1$', 'C2': 'r$C_2$', 'pred': 'total',
                                      '0_postA': '0_A', '0_postB': '0_B', '0_postC': '0_C', '0_postD': '0_D'},
                    linestyle_custom_dict={}, colour_custom_dict={}):
    if ax_top is None and plot_top:
        ax_top = plt.subplot(211)
    if ax_bottom is None and plot_bottom:
        ax_bottom = plt.subplot(212)
    if rnn_folder is None:
        list_rnns = [rnn_name]
    else:
        list_rnns = [x for x in os.listdir(rnn_folder) if x[-5:] == '.data']
    n_rnn = len(list_rnns)
    for i_rnn, rnn_name in enumerate(list_rnns):
        rnn = ru.load_rnn(rnn_name=os.path.join(rnn_folder, rnn_name))
        if i_rnn == 0:
            n_tp = len(rnn.test_loss_split['B'])
            if 'simulated_annealing' in list(rnn.info_dict.keys()) and rnn.info_dict['simulated_annealing']:
                pass
            else:
                assert n_tp == rnn.info_dict['n_epochs']  # double check and  assume it is the same for all rnns in rnn_folder\
            conv_dict = {key: np.zeros((n_rnn, n_tp)) for key in rnn.test_loss_split.keys()}
            if plot_total:
                conv_dict['pred'] = np.zeros((n_rnn, n_tp))
        for key, arr in rnn.test_loss_split.items():
            conv_dict[key][i_rnn, :] = arr.copy()
        if plot_total:
            conv_dict['pred'][i_rnn, :] = np.sum([conv_dict[key][i_rnn, :] for key in ['0', 'B', 'C', 'D']], 0)

    i_plot_total = 0
    dict_keys = list(conv_dict.keys())[::-1]
    colour_dict_keys = {key: color_dict_stand[it] for it, key in enumerate(['B', 'C', 'D', 'L1', 'MNM', '0', 'pred'])}
    colour_dict_keys['0'] = color_dict_stand[7]
    for key, val in colour_custom_dict.items():
        colour_dict_keys[key] = val
    linestyle_dict_keys = {x: '-' for x in label_dict_keys.keys()}
    for key, val in linestyle_custom_dict.items():
        linestyle_dict_keys[key] = val

    for key in dict_keys:
        mat = conv_dict[key]
        mat = mat / mat[:, 0][:, np.newaxis]
        plot_arr = np.mean(mat, 0)
        if plot_top:
            if (list_top is not None and key in list_top) or (list_top is None and '_' not in key and 'L' not in key):
                ax_top.plot(plot_arr, label=label_dict_keys[key], linestyle=linestyle_dict_keys[key], linewidth=lw, color=colour_dict_keys[key])
                ax_top.fill_between(x=np.arange(len(plot_arr)), y1=plot_arr - np.std(mat, 0),
                                    y2=plot_arr + np.std(mat, 0), alpha=0.2, color=colour_dict_keys[key])
                i_plot_total += 1
        if plot_bottom:
            if key == 'L1':
                ax_bottom.plot(plot_arr, label=key, linestyle='-', linewidth=lw, color=colour_dict_keys[key])
                ax_bottom.fill_between(x=np.arange(len(plot_arr)), y1=plot_arr - np.std(mat, 0),
                                    y2=plot_arr + np.std(mat, 0), alpha=0.2, color=colour_dict_keys[key])
                i_plot_total += 1
    if plot_top:
        ax_top.set_ylabel('Cross entropy ' + r'$H$')
        ax_top.set_xlabel('Epoch'); #ax.set_ylabel('error relative')
        ax_top.legend(frameon=False, bbox_to_anchor=(0.5, 0.2)); #ax.set_xlim([0, 10])
    if plot_bottom:
        ax_bottom.legend(frameon=False)
        ax_bottom.set_ylabel('L1 regularisation')
        ax_bottom.set_xlabel('Epoch'); #ax.set_ylabel('error relative')

    return (ax_top, ax_bottom)

def plot_split_perf_custom(folder_pred, folder_mnmpred, folder_mnm, ax=None,
                           plot_legend=True, legend_anchor=(1, 1)):
    if ax is None:
        ax = plt.subplot(111)

    ## prediction only
    _ = plot_split_perf(rnn_folder=folder_pred, list_top=['pred'], lw=5,
                        linestyle_custom_dict={'pred': '-'}, colour_custom_dict={'pred': [67 / 255, 0, 0]},
                        ax_top=ax, ax_bottom=None, plot_bottom=False, label_dict_keys={'pred': r'$H_{Pred}$' + f'    (Pred-only, N={len(os.listdir(folder_pred))})'})

    ## mnm only
    _ = plot_split_perf(rnn_folder=folder_mnm, list_top=['MNM'], lw=5,
                        linestyle_custom_dict={'MNM': '-'}, colour_custom_dict={'MNM': [207 / 255, 143 / 255, 23 / 255]},
                        ax_top=ax, ax_bottom=None, plot_bottom=False, label_dict_keys={'MNM': r'$H_{M/NM}$' + f'   (MNM-only, N={len(os.listdir(folder_mnm))})'})

    ## mnm+ prediction only
    colour_comb = [73 / 255, 154 / 255, 215 / 255]
    _ = plot_split_perf(rnn_folder=folder_mnmpred, list_top=['pred', 'MNM'], lw=5,
                        linestyle_custom_dict={'pred': ':', 'MNM': '-'},
                        colour_custom_dict={'pred': colour_comb, 'MNM': colour_comb},
                        ax_top=ax, ax_bottom=None, plot_bottom=False, label_dict_keys={'pred': r'$H_{Pred}$' + f'    (Pred & MNM,  N={len(os.listdir(folder_mnmpred))})',
                                                                                       'MNM': r'$H_{M/NM}$' + f'   (Pred & MNM,  N={len(os.listdir(folder_mnmpred))})'})

    if plot_legend:
        ax.legend(frameon=False, bbox_to_anchor=legend_anchor)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([-0.2, 1.2])
    return ax

def plot_decoder_crosstemp_perf(score_matrix, ax=None, ticklabels='', cmap_hm = 'BrBG', v_max=None, c_bar=True,
                                save_fig=False, fig_name='figures/example_low_crosstempmat.pdf', fontsize_ticks=10):
    '''Plot matrix of cross temporal scores for decoding'''
    if ax is None:
        ax = plt.subplot(111)
    # cmap_hm = sns.diverging_palette(145, 280, s=85, l=25, n=20)

    # cmap_hm = 'Greys'
    hm = sns.heatmap(score_matrix, cmap=cmap_hm, xticklabels=ticklabels, cbar=c_bar,
                     yticklabels=ticklabels, ax=ax, vmin=0, vmax=v_max,
                     linewidths=0.1, linecolor='k')
    # ax.invert_yaxis()
    ax.set_yticklabels(rotation=90, labels=ax.get_yticklabels(), fontsize=fontsize_ticks)
    ax.set_xticklabels(rotation=0, labels=ax.get_xticklabels(), fontsize=fontsize_ticks)
    bottom, top = ax.get_ylim()
    ax.set_ylim(top - 0.5, bottom + 0.5)
    ax.set_ylabel('Training time ' + r"$\tau$ $\longrightarrow$"); ax.set_xlabel('Testing time t ' + r'$\longrightarrow$')
    ax.set_title('Cross-temporal decoding score\nCorrelated single example', weight='bold');
    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    return (ax, hm)

def plot_raster_trial_average(forw, ax=None, save_fig=False, reverse_order=False, c_bar=True, ol=None,
                              fig_name='figures/example_high_forward_difference.pdf', index_label=0, plot_mnm=False):
    if plot_mnm:
        labels_use_1 = np.array([x == '11' or x == '22' for x in forw['labels_train']])  # expected / match
        labels_use_2 = np.array([x == '21' or x == '12' for x in forw['labels_train']])  # unexpected / non match
        plot_cmap = 'RdGy'
    else:
        labels_use_1 = np.array([x[index_label] == '1' for x in forw['labels_train']])
        labels_use_2 = np.array([x[index_label] == '2' for x in forw['labels_train']])
        # if index_label == 0:
        #     labels_use_1 = np.array([x == '11' for x in forw['labels_train']])
        #     labels_use_2 = np.array([x == '12' for x in forw['labels_train']])
        # elif index_label == 1:
        #     labels_use_1 = np.array([x == '22' for x in forw['labels_train']])
        #     labels_use_2 = np.array([x == '21' for x in forw['labels_train']])

        plot_cmap = 'PiYG'
    if ax is None:
        ax = plt.subplot(111)

    plot_diff = (forw['train'][labels_use_1, :, :].mean(0) - forw['train'][labels_use_2, :, :].mean(0))
    if ol is None:
        ol = opt_leaf(plot_diff, dim=1)  # optimal leaf sorting
    if reverse_order:
        ol = ol[::-1]
    # rev_ol = np.zeros_like(ol) # make reverse mapping of OL
    # for i_ol, el_ol in enumerate(ol):
    #     rev_ol[el_ol] = i_ol
    plot_diff = plot_diff[:, ol]
    th = np.max(np.abs(plot_diff)) # threshold for visualisation
    sns.heatmap(plot_diff.T, cmap=plot_cmap, vmin=-1 * th, vmax=th, ax=ax,
                     xticklabels=double_time_labels_blank[:-1], cbar=c_bar)
    ax.invert_yaxis()
    ax.set_yticklabels(rotation=0, labels=ax.get_yticklabels())
    ax.set_xticklabels(rotation=0, labels=ax.get_xticklabels())
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom - 0.5, top + 1.5)
    ax.set_title('Activity difference between green and purple trials', weight='bold')
    ax.set_xlabel('Time'); ax.set_ylabel('neuron #');
    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    return ol

def plot_dynamic_decoding_axes(rnn, ticklabels=double_time_labels_blank[:-1],
                               neuron_order=None, label='alpha'):
    '''Plot the decoding axis w for each time point; and the diagonal auto-decoding
    accuracy on top. Returns these two axes. '''
    # if ax is None:
    #     ax = plt.subplot(111)

    decoder_axes = np.zeros((rnn.decoder_dict[label][0].coef_.size, len(rnn.decoder_dict[label])))
    for k, v in rnn.decoder_dict[label].items():
        decoder_axes[:, k] = v.coef_
    cutoff_w = np.percentile(np.abs(decoder_axes), 99)
    if neuron_order is not None:
        assert len(neuron_order) == decoder_axes.shape[0]
        decoder_axes = decoder_axes[neuron_order, :]

    ax_dec_diag = plt.subplot(3, 1, 1)
    ax_dec_diag.plot(np.diag(rnn.decoding_crosstemp_score[label]), linewidth=3,
                             linestyle='-', marker='.', markersize=10, color='k', alpha=0.6)
    ax_dec_diag.set_ylabel('Score')
    ax_dec_diag.set_title('Decoding performance (t = tau)')
    ax_dec_diag.set_xticks(np.arange(len(np.diag(rnn.decoding_crosstemp_score[label]))));
    ax_dec_diag.set_xticklabels(ticklabels);
    ax_dec_diag.set_xlim([-0.5, len(np.diag(rnn.decoding_crosstemp_score[label])) - 0.5])

    plt.subplot(3, 1, (2, 3))
    ax_dec_w = sns.heatmap(decoder_axes, xticklabels=ticklabels,
                          vmin=-1 * cutoff_w, vmax=cutoff_w, cmap='PiYG_r', cbar=False)
    bottom, top = ax_dec_w.get_ylim()
    ax_dec_w.set_ylim(bottom + 0.5, top - 0.5)
    ax_dec_w.set_xlabel('Time t'); ax_dec_w.set_ylabel('Neuron');
    return ax_dec_diag, ax_dec_w

def plot_example_trial(trial, ax=None, yticklabels=freq_labels_sub,
                       xticklabels=double_time_labels_blank[1:], c_bar=True,
                       vmin=0, vmax=1, c_map='magma', print_labels=True):
    '''Plot 1 example trial'''
    if ax is None:
        ax = plt.subplot(111)

    sns.heatmap(trial.T, yticklabels=yticklabels, cmap=c_map, cbar=c_bar,
            xticklabels=xticklabels, ax=ax, vmin=vmin, vmax=vmax, )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_yticklabels(labels=yticklabels, rotation=0)
    ax.set_xticklabels(labels=xticklabels, rotation=0)
    if print_labels:
        ax.set_xlabel('Time')
        ax.set_ylabel('Stimulus vector')
    return ax

def plot_time_trace_1_decoding_neuron(rnn, n_neuron=3, ax=None, label='alpha'):
    if ax is None:
        ax = plt.subplot(111)
    n_tp = len(rnn.decoder_dict[label])
    time_trace_pos = np.zeros(n_tp)
    time_trace_neg = np.zeros(n_tp)
    for i_tp in range(n_tp):
        time_trace_pos[i_tp] = np.clip(rnn.decoder_dict[label][i_tp].coef_[0][n_neuron], a_min=0, a_max=np.inf)
        time_trace_neg[i_tp] = -1 * np.clip(rnn.decoder_dict[label][i_tp].coef_[0][n_neuron], a_max=0, a_min=-1 * np.inf)
    ax.plot(time_trace_pos, linewidth=3, c='green')
    ax.plot(time_trace_neg, linewidth=3, c='m')
    ax.set_xlabel('time'); ax.set_ylabel('Decoding strenght')
    return ax

def plot_summary_ratios(agg_weights, agg_decoder_mat, agg_score,
                        time_labels=double_time_labels_blank[:-1], input_labels=freq_labels,
                        save_fig=False, fig_name='figures/details_ratio_exp_all.pdf'):
    '''Plot 3 subplots that summarise the differences between expected/Unexpected ratios
    1) average input weights; 2) average decoding weights; 3) dynamic decoding performance'''
    alpha_dict = {x: 0.2 + 0.2 * i_kk for i_kk, x in enumerate(agg_weights.keys())}  # transparency values
    list_ratios = list(agg_weights.keys()) #[75]
    ax_abs_w = plt.subplot(131)  # plot input weights
    for i_kk, kk in enumerate(list_ratios):
        ax_abs_w.plot(np.mean(agg_weights[kk], 0), alpha=alpha_dict[kk], color='k',
                      marker='.', linewidth=3, markersize=10, label=kk)
    #     ax_abs_w.bar(x=freq_labels, height=np.mean(agg_weights[kk], 0))
    plt.legend(frameon=False); ax_abs_w.set_xticks(np.arange(agg_weights[kk].shape[1]));
    ax_abs_w.set_xticklabels(input_labels); ax_abs_w.set_ylabel('Average absolute weights')
    ax_abs_w.set_title('Average U input weight values', weight='bold')
    ax_abs_w.set_xlabel('Input node')

    ax_dec_w = plt.subplot(132)  # decoding weights
    for i_kk, kk in enumerate(list_ratios):
        ax_dec_w.plot(np.abs(agg_decoder_mat[kk]).mean((0, 1)), alpha=alpha_dict[kk], color='k',
                      marker='.', linewidth=3, markersize=10, label=kk)
    ax_dec_w.set_xticks(np.arange(agg_decoder_mat[75].shape[2]))
    ax_dec_w.set_xticklabels(time_labels);
    ax_dec_w.set_xlabel('Time'); ax_dec_w.set_ylabel('Average absolute weight');
    ax_dec_w.set_title('Average decoding weight values', weight='bold')

    ax_dec_perf = plt.subplot(133)  # decoding performance
    for i_kk, kk in enumerate(list_ratios):
        ax_dec_perf.plot(np.diag(agg_score[kk].mean(0)), alpha=alpha_dict[kk], color='k',
                      marker='.', linewidth=3, markersize=10, label=kk)
    ax_dec_perf.set_xticks(np.arange(agg_decoder_mat[75].shape[2]))
    ax_dec_perf.set_xticklabels(time_labels);
    ax_dec_perf.set_xlabel('Time'); ax_dec_w.set_ylabel('Accuracy');
    ax_dec_perf.set_title(r'$\alpha$ decoding performance', weight='bold', fontsize=15);
    ax_dec_perf.axvspan(xmin=10, xmax=17, alpha=0.15)
    ax_dec_perf.set_xlim([0, 17])
    # ax_abs_w.text(s='A', x=-2, y=0.22, fontdict={'fontsize': 20, 'weight':'bold'})
    # ax_abs_w.text(s='B', x=8.5, y=0.22, fontdict={'fontsize': 20, 'weight':'bold'})
    # ax_abs_w.text(s='C', x=19.5, y=0.22, fontdict={'fontsize': 20, 'weight':'bold'})
    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')

    return (ax_abs_w, ax_dec_w, ax_dec_perf)

def plot_alpha_beta_performance(alpha_perf=None, beta_perf=None, ax=None,
                                time_labels=double_time_labels_blank[:-1],
                                save_fig=False, fig_name='figures/alpha_beta_decoding_75.pdf'):
    '''Plot two lines - alpha_perf & beta_perf'''
    if ax is None:
        ax = plt.subplot(111)
    if alpha_perf is not None:
        ax.plot(alpha_perf, alpha=0.9, color='#018571',
                      marker='', linewidth=3, markersize=18, label=r"$\alpha$")
    if beta_perf is not None:
        ax.plot(beta_perf, alpha=0.9, color='k', linestyle='-',
                      marker='', linewidth=3, markersize=18, label=r'$\beta$')
    ax.set_xticks(np.arange(len(time_labels)))
    ax.set_xticklabels(time_labels);
    ax.set_yticks([0.5, 0.75, 1])
    ax.set_xlabel('Time ' + r'$\longrightarrow$'); ax.set_ylabel('Accuracy');
    # ax.legend(bbox_to_anchor=(1,0 , 0, 1), fontsize=20)
    ax.legend(bbox_to_anchor=(0.85, 0.1, 0, 1), frameon=False)
    # ax.set_title(r'$\alpha$ decoding performance', weight='bold', fontsize=15);
    # ax.axvspan(xmin=10, xmax=17, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0, 17]);
    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    return ax

def plot_stable_switch_bar_diagram(stable_list, switch_list, ax=None, bar_width=0.35,
                                   save_fig=False, fig_name='figures/stable_switch_correlated.pdf'):
    '''Plot bar diagram of number of stable & switch neurons '''
    assert len(stable_list) == len(switch_list)
    if ax is None:
        ax = plt.subplot(111)

    bar_locs = np.arange(len(stable_list))
    bar_stable = ax.bar(bar_locs - bar_width / 2, stable_list,
                         width=bar_width, label='stable', color='#6699FF')  # plot bar
    bar_switch = ax.bar(bar_locs + bar_width / 2, switch_list,
                         width=bar_width, label='switch', color='#660033')
    ax.set_xticks(bar_locs)
    ax.set_xticklabels(['anti-correlated', 'decorrelated', 'correlated'], rotation=0) # ax_bar.set_xticklabels(inds_sel.keys())
    ax.legend(frameon=False); ax.set_ylabel('Fraction of neurons');
    ax.set_title('Distribution of stable & switch neurons', weight='bold')
    # sns.despine()
    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    return ax

def plot_neuron_diff(ax_select, act_1, act_2, mean_ls='-',
                     time_labels=double_time_labels_blank[:-1]):
    ax_select.axvspan(xmin=1.5, xmax=3.5, color='grey', alpha=0.3)   # grey vspans around letters
    ax_select.axvspan(xmin=5.5, xmax=7.5, color='grey', alpha=0.3)
    ax_select.axvspan(xmin=9.5, xmax=11.5, color='grey', alpha=0.3)
    ax_select.axvspan(xmin=13.5, xmax=15.5, color='grey', alpha=0.3)
    mean_act = (act_1 + act_2) / 2  # mean activity
    c_mat_green = ru.create_color_mat(x=act_1, c='green')  # create gradient colours
    c_mat_mag = ru.create_color_mat(x=act_1, c='m')
    c_mat_k = ru.create_color_mat(x=act_1, c='k')
    for ii in range(len(act_1) - 1):  # plot the lines per segment, in order to change the gradient colour
        ax_select.plot(np.arange(ii, (ii + 2)), mean_act[ii:(ii + 2)],
                       linewidth=3, linestyle=mean_ls, c='k', alpha=0.8)#c=c_mat_k[ii, :])
        ax_select.plot(np.arange(ii, (ii + 2)), act_2[ii:(ii + 2)],
                       linewidth=5, c=c_mat_mag[ii, :]);
        ax_select.plot(np.arange(ii, (ii + 2)), act_1[ii:(ii + 2)],
                       c=c_mat_green[ii, :], linewidth=5);
    ax_select.set_xticks(np.arange(len(time_labels)));
    ax_select.set_xlabel('Time'); ax_select.set_ylabel('Activity')
    ax_select.set_xticklabels(time_labels);
    ax_select.set_ylim([-1, 1.3])  # set this by hand to stretch the vspans
    for i_letter, letter in enumerate(['A', 'B', 'C', 'D']):
        ax_select.text(s=letter, x=1.8 + 4 * i_letter, y=1.15,
                       fontdict={'weight': 'bold', 'fontsize': 18})
    ax_select.spines['top'].set_visible(False)
    ax_select.spines['right'].set_visible(False)
    return ax_select

def plot_arrow_line(x, y, ax=None, c='blue', verbose=False, swap_x=False,
                    swap_y=False, draw_time=False, draw_names_sens_mem=False,
                    color_sense='#8da0cb', color_mem='#fc8d62',
                    sens_times=np.arange(2, 8), mem_times=np.arange(8, 11), draw_sens_mem=False):
    if ax is None:
        ax = plt.subplot(111)
    c_mat = ru.create_color_mat(x=x, c=c)
#     ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], width=.02,
#               scale_units='xy', angles='xy', scale=1, color=c_mat)
    if swap_y:
        y = -1 * y
    if swap_x:
        x = -1 * x
    x_sens_A = np.mean(x[sens_times[:2]])
    y_sens_A = np.mean(y[sens_times[:2]])
    x_sens_B = np.mean(x[sens_times[-2:]])
    y_sens_B = np.mean(y[sens_times[-2:]])
    x_mem = np.mean(x[mem_times])
    y_mem = np.mean(y[mem_times])
    if verbose:
        print('sens', x_sens.round(2), y_sens.round(2))
        print('mem', x_mem.round(2), y_mem.round(2))
    traj_width = {True: 5, False: 5}  # T: 3, F: 7
    for ii in range(len(x) - 1): # plot trajectores
        ax.plot(x[ii:(ii + 2)], y[ii:(ii + 2)], c=c_mat[ii, :], linewidth=traj_width[draw_sens_mem], zorder=1)
#     ax.plot(x, y, color=c_mat)
    # plt.scatter(x_sens, y_sens, marker='x', s=50, c=c_mat[-1, :][np.newaxis, :])
    # plt.scatter(x_mem, y_mem, marker='o', s=50, c=c_mat[-1, :][np.newaxis, :])
    if draw_sens_mem:
        if c == 'm':
            total_sense_length = np.sqrt((x_sens_A + x_sens_B) ** 2 + (y_sens_A + y_sens_B) ** 2)
            arr_sens_A = patches.Arrow(x=0, y=0, dx=(x_sens_A + x_sens_B) / total_sense_length,
                                        dy=(y_sens_A + y_sens_B) / total_sense_length,
                                        color=color_sense, width=0.3, zorder=2)
            # arr_sens_B = patches.Arrow(x=0, y=0, dx=x_sens_B, dy=y_sens_B, color=color_sense, width=0.2, zorder=2)
            arr_mem = patches.Arrow(x=0, y=0, dx=x_mem / np.sqrt(x_mem ** 2 + y_mem ** 2),
                                    dy=y_mem / np.sqrt(x_mem ** 2 + y_mem ** 2), color=color_mem, width=0.3, zorder=2)
            ax.add_patch(arr_sens_A)
            # ax.add_patch(arr_sens_B)
            ax.add_patch(arr_mem)
            if draw_names_sens_mem == 'high':
                plt.text(s='sensory\nexperience', x=0.4, y=0.74, c=color_sense)
                plt.text(s='correlated\nmemory', x=0.05, y=-0.39, c=color_mem)
            elif draw_names_sens_mem == 'med':
                plt.text(s='sensory\nexperience', x=0.5, y=0.5, c=color_sense)
                plt.text(s='decorrelated\nmemory', x=0.35, y=-1, c=color_mem)
            elif draw_names_sens_mem == 'low':
                plt.text(s='sensory\nexperience', x=0.7, y=0.6, c=color_sense)
                plt.text(s='anti-correlated\nmemory', x=-1.4, y=-1.23, c=color_mem)
    if draw_time:
        time_arrow = {}
        if draw_time == 'high':
            time_arrow[0] = patches.FancyArrowPatch(posA=(0.2, 0.9), posB=(1.05, 0.1),
                                                    arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                                                    connectionstyle="arc3, rad=-0.5", **{'color' : 'grey'})
            ax.text(s='Time', x=0.85, y=0.8)
        elif draw_time == 'med':
            # time_arrow[0] = patches.FancyArrowPatch(posA=(0.2, 0.95), posB=(0.2, -0.8), zorder=-0.5,
            #                                        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
            #                                        connectionstyle="arc3, rad=-1.0", **{'color' : 'grey'})\
            time_arrow[0] = patches.FancyArrowPatch(posA=(0.2, 0.7), posB=(1.05, 0.05), zorder=-0.5,
                                                    arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                                                    connectionstyle="arc3, rad=-0.5", **{'color' : 'grey'})
            time_arrow[1] = patches.FancyArrowPatch(posA=(0.85, -0.6), posB=(0.05, -0.9), zorder=-0.5,
                                                    arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                                                    connectionstyle="arc3, rad=-0.3", **{'color' : 'grey'})
            ax.text(s='Time', x=0.75, y=0.7)
        elif draw_time == 'low':
            time_arrow[0] = patches.FancyArrowPatch(posA=(0.15, 0.90), posB=(0.75, 0.15),
                                                    arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                                                    connectionstyle="arc3, rad=-1", **{'color' : 'grey'})
            time_arrow[1] = patches.FancyArrowPatch(posA=(0.75, 0.15), posB=(0.05, -0.77),
                                                    arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                                                    connectionstyle="arc3, rad=-0.2", **{'color' : 'grey'})
            time_arrow[2] = patches.FancyArrowPatch(posA=(-0.01, -0.78), posB=(-0.6, -0.1),
                                                    arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                                                    connectionstyle="arc3, rad=-1.3", **{'color' : 'grey'})
            ax.text(s='Time', x=0.9, y=0.8)
        for key, arrow in time_arrow.items():
            ax.add_patch(arrow)
    # ax.set_axis_off()
    # ax.get_xaxis().set_visible(False)
    ax.set_xticks([])

    # ax.get_yaxis().set_visible(False)
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
    if swap_x:
        ax.text(s='-1', x=0.86, y=-0.16, fontdict={'fontsize': 12})
        ax.text(s='1', x=-1, y=-0.16, fontdict={'fontsize': 12})
    elif not swap_x:
        ax.text(s='1', x=0.90, y=-0.16, fontdict={'fontsize': 12})
        ax.text(s='-1', x=-1.02, y=-0.16, fontdict={'fontsize': 12})
    if swap_y:
        ax.text(s='-1', x=-0.2, y=0.9, fontdict={'fontsize': 12})
        ax.text(s='1', x=-0.14, y=-1.0, fontdict={'fontsize': 12})
    elif not swap_y:
        ax.text(s='1', x=-0.14, y=0.9, fontdict={'fontsize': 12})
        ax.text(s='-1', x=-0.2, y=-1.0, fontdict={'fontsize': 12})
    return ax

def plot_two_neuron_state_space(activity_1, activity_2, mean_ls_dict, swap_x=False, swap_y=False,
                                max_tp=17, ax=None, save_fig=False, font_size=16,
                                x_name='Stable neuron', y_name='Switch neuron',
                                draw_sens_mem=False, draw_time=False, draw_names_sens_mem=False,
                                fig_name='figures/example_med_statespace_stable-switch.pdf'):
    if ax is None:
        ax = plt.subplot(111)

    n1 = list(activity_1.keys())[0]
    n2 = list(activity_1.keys())[1]
    mean_n1 = (activity_1[n1] + activity_2[n1]) / 2
    mean_n2 = (activity_1[n2] + activity_2[n2]) / 2
    ax.plot([-1, 1], [0, 0], c='k', linewidth=2, linestyle=mean_ls_dict[1], zorder=0.5)  # x axis - so correpsonding to neuron on y axis
    ax.plot([0, 0], [-1, 1], c='k', linewidth=2, linestyle=mean_ls_dict[0], zorder=0.5)  # y axis - so correspond to neuron on x axis

    _ = plot_arrow_line(activity_1[n1][:max_tp] - mean_n1[:max_tp],
             activity_1[n2][:max_tp] - mean_n2[:max_tp], c='green',# draw_time=draw_time,
             draw_sens_mem=draw_sens_mem, ax=ax, swap_x=swap_x, swap_y=swap_y)
    _ = plot_arrow_line(activity_2[n1][:max_tp] - mean_n1[:max_tp],
             activity_2[n2][:max_tp] - mean_n2[:max_tp], c='m', draw_time=draw_time,
             draw_sens_mem=draw_sens_mem, ax=ax, swap_x=swap_x, swap_y=swap_y,
             draw_names_sens_mem=draw_names_sens_mem)
    # ax.set_title('State space', weight='bold')
    ax.text(s=x_name, x=0.72, y=-0.58,
               fontdict={'fontsize': font_size})
    ax.text(s=y_name, x=-0.5, y=1.12,
               fontdict={'fontsize': font_size})

    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    return ax


def plot_trial_activity(forw, ax, neuron_order=None, n_trial=0, c_bar=True, print_labels=False):
    tmp_act = forw['test'][n_trial, :, :].T
    if neuron_order is None:
        neuron_order is np.arange(tmp_act.shape[0])
    tmp_act = np.squeeze(tmp_act[neuron_order, :])
    sns.heatmap(tmp_act, vmin=-1, vmax=1, cmap='RdBu_r', cbar=c_bar,
                xticklabels=double_time_labels_blank[:-1], ax=ax,
                cbar_kws={'pad': 0.01, 'fraction': 0.01})
    ax.set_xticklabels(rotation=0, labels=ax.get_xticklabels())
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 1.5, top - 0.5)
    if print_labels:
        ax.set_ylabel('neuron #'); ax.set_xlabel('time');
        ax.set_title(f'Activity of {forw["labels_test"][n_trial]} trial')
    return ax

def plot_convergence_rnn(rnn, save_fig=False, verbose=False,
                         fig_name='figures/convergence_training.pdf'):
    # plt.subplot(211)
    ax_conv = plot_train_test_perf(rnn_model=rnn, ax=plt.subplot(211), train=False)
    # plt.ylim([1.2, 2])
    ax_conv.text(s='Test loss during training', x=5, y=7, fontdict={'weight': 'bold'})
    ax_conv.set_ylabel('Total loss')

    ax_ratio_ce = plt.subplot(212)
    ax_ratio_ce.plot(np.arange(rnn.info_dict['trained_epochs']),
             np.zeros_like(rnn.test_loss_ratio_ce) + 0.5, c='grey', linestyle=':')
    ax_ratio_ce.plot(rnn.test_loss_ratio_ce, linewidth=3, c='k', linestyle='--')
    ax_ratio_ce.set_xlabel('Epoch'); ax_ratio_ce.set_ylabel('ratio CE loss');
    ax_ratio_ce.text(s='Ratio Cross Entropy / Total loss', x=5, y=0.8,
                     fontdict={'weight': 'bold'});
    # sns.despine()
    if verbose:
        print(f'Final test performance: {np.round(rnn.test_loss_arr[-1], 3)}')
    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    return(ax_conv, ax_ratio_ce)


def plot_multiple_rnn_properties(rnn_name_dict, rnn_folder, fontsize=10):
    n_rnn = len(rnn_name_dict)
    n_panels = 4
    fig = plt.figure(constrained_layout=False)
    gs = {}
    gs[0] = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1, 1, 1], wspace=0.5, bottom=0.84, top=1)  # [1, 2.2, 1.2]
    gs[1] = fig.add_gridspec(ncols=3, nrows=2, width_ratios=[1, 1, 1], wspace=0.5, bottom=0.56, top=0.75, hspace=0.5)  # [1, 2.2, 1.2]
    gs[2] = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1, 1, 1], wspace=0.5, bottom=0.25, top=0.46)  # [1, 2.2, 1.2]
    gs[3] = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1, 1, 1], wspace=0.5, bottom=0, top=0.22)  # [1, 2.2, 1.2]
    rnn = {}
    i_rnn = 0
    mean_ls_dict = {0: '-', 1: ':'}
    neuron_selection = {'low': [0, 19], 'med': [0, 5], 'high': [19, 2]}
    title_selection = {'low': ['Switch neuron (#0)', 'Switch neuron (#19)'],
                       'med': ['Stable neuron (#0)', 'Switch neuron (#5)'],
                       'high': ['Stable neuron (#19)', 'Stable neuron (#2)']}
    rnn_title_dict = {'low': 'C) Anti-correlated coding', 'med': 'B) Decorrelated coding',
                      'high': 'A) Correlated coding'}
    swap_x_dict = {'low': True, 'med': True, 'high': False}
    swap_y_dict = {'low': False, 'med': True, 'high': True}
    ax_rast, ax_single, ax_ss, ax_ss_arr, ax_ctmat, ol = {}, {}, {}, {}, {}, {}
    for key, rnn_name in rnn_name_dict.items():
        rnn[key] = ru.load_rnn(os.path.join(rnn_folder, rnn_name))
        _, __, forw  = bp.train_single_decoder_new_data(rnn=rnn[key], ratio_expected=0.5,
                                                        sparsity_c=0.1, bool_train_decoder=False)
        labels_use_1 = np.array([x[0] == '1' for x in forw['labels_test']])
        labels_use_2 = np.array([x[0] == '2' for x in forw['labels_test']])

        ## raster plot
        ax_rast[key] = fig.add_subplot(gs[0][i_rnn])
        ol[key] = plot_raster_trial_average(forw=forw, ax=ax_rast[key], reverse_order=(key == 'low'), c_bar=False)
        ax_rast[key].set_title(rnn_title_dict[key], weight='bold')
        ax_rast[key].set_xlabel('Time ' + r"$\longrightarrow$")

        ## single examples
        ax_single[key] = {}  # plt.subplot(n_panels, n_rnn, 2 + (i_rnn * n_panels))
        activity_1, activity_2 = {}, {}
        for i_plot, n_neuron in enumerate([ol[key][neuron_selection[key][xx]] for xx in range(2)]): # two pre selected neurons
            ax_single[key][i_plot] = fig.add_subplot(gs[1][i_plot, i_rnn])
            activity_1[n_neuron] = forw['test'][labels_use_1, :, :][:, :, n_neuron].mean(0)
            activity_2[n_neuron] = forw['test'][labels_use_2, :, :][:, :, n_neuron].mean(0)
            ax_single[key][i_plot] = plot_neuron_diff(ax_select=ax_single[key][i_plot],
                                                      act_1=activity_1[n_neuron],
                                                      act_2=activity_2[n_neuron],
                                                      mean_ls=mean_ls_dict[i_plot])
            ax_single[key][i_plot].set_title(f'{title_selection[key][i_plot]}')
            ax_single[key][i_plot].set_ylim([-1, 1.75])  # overwrite stretch to fit vspans
            ax_single[key][i_plot].set_ylabel('')
            ax_single[key][i_plot].set_xlabel('')
            if i_plot == 0:
                ax_single[key][i_plot].set_xticklabels(['' for x in range(len(double_time_labels_blank[:-1]))])
            elif i_plot == 1:
                ax_single[key][i_plot].set_xlabel('Time ' + r'$\longrightarrow$')

        ## state space
        ax_ss[key] = fig.add_subplot(gs[2][i_rnn])
        plot_two_neuron_state_space(activity_1=activity_1, activity_2=activity_2, font_size=fontsize,
                                   mean_ls_dict=mean_ls_dict, save_fig=False, ax=ax_ss[key],
                                   swap_x=swap_x_dict[key], swap_y=swap_y_dict[key],
                                   draw_time=key,
                                   x_name=title_selection[key][0][:6] + '\nneuron',
                                   y_name=title_selection[key][1][:6] + ' neuron')
        # ax_ss[key].set_xlim([-1, 1])
        # ax_ss[key].set_ylim([-1, 1])


        ## state space with arrows
        ax_ss_arr[key] = fig.add_subplot(gs[3][i_rnn])
        plot_two_neuron_state_space(activity_1=activity_1, activity_2=activity_2, font_size=fontsize,
                                    mean_ls_dict=mean_ls_dict, save_fig=False, ax=ax_ss_arr[key],
                                    swap_x=swap_x_dict[key], swap_y=swap_y_dict[key],
                                    draw_sens_mem=True, x_name=title_selection[key][0][:6] + '\nneuron',
                                    y_name=title_selection[key][1][:6] + ' neuron',
                                    draw_names_sens_mem=key)
        # ax_ss_arr[key].set_xlim([-0.9, 0.9])
        # ax_ss_arr[key].set_ylim([-0.9, 0.9])

        # ## CT matrix
        # ax_ctmat[key] = fig.add_subplot(gs_top[i_rnn])
        # _, hm = plot_decoder_crosstemp_perf(score_matrix=rnn[key].decoding_crosstemp_score['alpha'],
        #                        ax=ax_ctmat[key], c_bar=False,
        #                        ticklabels=double_time_labels_blank[:-1])
        # ax_ctmat[key].set_title('')


        if i_rnn == 0:
            pass
        #     ax_rast[key].text(s='i) raster plot\ncolour preference', x=-0.4, y=0.5,
        #                       va='top', ha='center', rotation='vertical', fontdict={'weight': 'bold'})
        #     ax_single[key][1].text(s='ii) Two representative\nexample neurons', x=-0.4, y=0.5,
        #                       va='top', ha='center', rotation='vertical', fontdict={'weight': 'bold'})
        #     ax_ss[key].text(s='iii) state space of \nexample neurons \n ', x=-0.4, y=0.5,
        #                       va='top', ha='center', rotation='vertical', fontdict={'weight': 'bold'})
        #     ax_ss_arr[key].text(s='iv) switch cells rotate \nmemory representation\n ', x=-0.4, y=0.5,
        #                       va='top', ha='center', rotation='vertical', fontdict={'weight': 'bold'})
        #
        #     # ax_ctmat[key].set_ylabel('v) cross-temporal \ndecoding accuracy', weight='bold')

        elif i_rnn == (n_rnn - 1):
            ## Color bar raster  matrix:
            divider = make_axes_locatable(ax_rast[key])
            cax_rast = divider.append_axes('right', size='5%', pad=0.15)
            mpl_colorbar(ax_rast[key].get_children()[0], cax=cax_rast)
            cax_rast.yaxis.set_ticks_position('right')
            cax_rast.set_ylabel(r'$\mathbf{r}_{green} - \mathbf{r}_{purple}$')
            # ## Color bar ct matrix:
            # divider = make_axes_locatable(ax_ctmat[key])
            # cax_ct = divider.append_axes('right', size='5%', pad=0.01)
            # mpl_colorbar(hm.get_children()[0], cax=cax_ct)
            # cax_ct.yaxis.set_ticks_position('right')

        i_rnn += 1
    # for ax in [ax_rast['high'], ax_ss['high'], ax_ss_arr['high'], ax_single['high'][1]]:
        # ax.yaxis.label.set_va('top')
        # ax.yaxis.set_label_coords(x=-0.3, y=0.5)
    for ind in ['high', 'med', 'low']:
        ax_rast[ind].yaxis.label.set_va('top')
        ax_rast[ind].yaxis.set_label_coords(x=-0.24, y=0.5)
        ax_rast[ind].title.set_ha('left')
        ax_rast[ind].title.set_va('bottom')
        ax_rast[ind].title.set_position((-0.24, 1.25))
        ax_single[ind][1].set_ylabel('Neural activity ' + r'$r_t$')
        ax_single[ind][1].yaxis.label.set_va('top')
        ax_single[ind][1].yaxis.set_label_coords(x=-0.24, y=1.2)

    fig.text(s='i) raster plot\ncolour preference', x=0.03, y=0.98,
                      va='top', ha='left', rotation='vertical', fontdict={'weight': 'bold'})
    fig.text(s='ii) Two representative\nexample neurons', x=0.03, y=0.74,
                      va='top', ha='left', rotation='vertical', fontdict={'weight': 'bold'})
    fig.text(s='iii) state space of \nexample neurons \n ', x=0.03, y=0.46,
                      va='top', ha='left', rotation='vertical', fontdict={'weight': 'bold'})
    fig.text(s='iv) switch cells rotate \nmemory representation\n ', x=0.03, y=0.24,
                      va='top', ha='left', rotation='vertical', fontdict={'weight': 'bold'})

    return None

def plot_prediction_example(rnn, verbose=1, plot_conv=True):

    ## Generate new test trials:
    if verbose:
        print(f'generating data with {rnn.info_dict["ratio_train"]} train ratio, {rnn.info_dict["ratio_exp"]} expected ratio')
    tmp0, tmp1 = bp.generate_synt_data(n_total=100,
                                   n_times=rnn.info_dict['n_times'],
                                   n_freq=rnn.info_dict['n_freq'],
                                   ratio_train=rnn.info_dict['ratio_train'],
                                   ratio_exp=rnn.info_dict['ratio_exp'],
                                   noise_scale=rnn.info_dict['noise_scale'],
                                   double_length=rnn.info_dict['doublesse'])
    x_train, y_train, x_test, y_test = tmp0
    labels_train, labels_test = tmp1
    _, __, forward_mat = bp.train_decoder(rnn_model=rnn, x_train=x_train, x_test=x_test,
                                       labels_train=labels_train, labels_test=labels_test,
                                       save_inplace=False, sparsity_c=0.1, label_name='alpha', bool_train_decoder=False)  # train decoder just to get forward matrix really
    forward_mat['labels_train'] = labels_train
    forward_mat['labels_test'] = labels_test


    fig = plt.figure(constrained_layout=False)
    if plot_conv is True:
        plot_conv = 1
        gs = fig.add_gridspec(ncols=3, nrows=2, width_ratios=[1, 1, 1], left=0.32, right=1, wspace=0.3)  # [1, 2.2, 1.2]
        gs_bottom = fig.add_gridspec(ncols=1, nrows=3, left=0, right=0.22, hspace=0.4)
    else:
        plot_conv = 0
        gs = fig.add_gridspec(ncols=3, nrows=2, width_ratios=[1, 1, 1], wspace=0.4)


    ax_gt, ax_act, ax_pred = {}, {}, {}
    ind_exp, true_exp, pred_exp, input_exp = {}, {}, {}, {}
    for i_ind, ind in enumerate(['11', '12']):
        ind_exp[ind] = np.where(labels_test == ind)[0][0]
        pred_exp[ind] = bp.compute_full_pred(x_test[ind_exp[ind],:,:], model=rnn)  # computed forward predictions
        true_exp[ind] = y_test[ind_exp[ind], :, :]
        input_exp[ind] = x_test[ind_exp[ind], :, :]
        pred_exp[ind] = pred_exp[ind].squeeze()
        input_exp[ind] = input_exp[ind].squeeze()
        true_exp[ind] = true_exp[ind].squeeze()
        assert pred_exp[ind].ndim == true_exp[ind].ndim and pred_exp[ind].ndim == 2, 'pred_exp or true_exp doesnt have dim 2, probably because it is mutliple trials'
        pred_exp[ind] = pred_exp[ind].detach().numpy()
        true_exp[ind] = true_exp[ind].detach().numpy()
        input_exp[ind] = input_exp[ind].detach().numpy()

        if i_ind == 0:  # sort neurons
            eval_times = rnn.info_dict['eval_times']
            non_eval_times = np.array(list(set(np.arange(eval_times[-1])).difference(set(eval_times))))
            ol = opt_leaf(forward_mat['test'][ind_exp[ind], :, :].T)  # optimal leaf sorting
            forward_mat['test'] = forward_mat['test'][:, :, ol]
        pred_exp[ind][non_eval_times, :] = 0  # set non-clamped time points to 0

        ax_gt[ind] = fig.add_subplot(gs[i_ind, 0])  # stimuli
        plot_example_trial(input_exp[ind], ax=ax_gt[ind], c_map='bone_r', print_labels=False,
                           c_bar=False, xticklabels=double_time_labels_blank[:-1])

        ax_act[ind] = fig.add_subplot(gs[i_ind, 1])  # activity
        plot_trial_activity(forw=forward_mat, ax=ax_act[ind], n_trial=ind_exp[ind], c_bar=False)
        ax_act[ind].invert_yaxis()
        ax_act[ind].set_yticklabels(rotation=0, labels=ax_act[ind].get_yticklabels())

        ax_pred[ind] = fig.add_subplot(gs[i_ind, 2])  # predictions
        plot_example_trial(pred_exp[ind], ax=ax_pred[ind], c_map='bone_r', print_labels=False, c_bar=False)

        ## C highlight:
        # for ax in [ax_pred[ind], ax_gt[ind]]:
        color_patch = '#ff7f00'
        ax_pred[ind].add_patch(patches.FancyBboxPatch((8.5, 4.85), width=2.9, height=2.45,
                                       fill=False, edgecolor=color_patch, lw=3))
        ax_gt[ind].add_patch(patches.FancyBboxPatch((9.5, 4.85), width=2.9, height=2.45,
                                      fill=False, edgecolor=color_patch, lw=3))

        ## Colorbars
        if i_ind == 0:
            divider = make_axes_locatable(ax_pred[ind])
            cax_top = divider.append_axes('right', size='5%', pad=0.1)
            mpl_colorbar(ax_act[ind].get_children()[0], cax=cax_top)
            cax_top.yaxis.set_ticks_position('right')
            cax_top.set_yticks([-1, -0.5, 0, 0.5, 1])
            for tick in cax_top.yaxis.get_major_ticks():
                tick.label.set_fontsize('x-small')
            cax_top.set_ylabel('Neural activity')
        elif i_ind == 1:
            divider = make_axes_locatable(ax_pred[ind])
            cax_bottom = divider.append_axes('right', size='5%', pad=0.1)
            mpl_colorbar(ax_pred[ind].get_children()[0], cax=cax_bottom)
            cax_bottom.yaxis.set_ticks_position('right')
            cax_bottom.set_yticks(np.linspace(0, 1, 6))
            for tick in cax_bottom.yaxis.get_major_ticks():
                tick.label.set_fontsize('x-small')
            cax_bottom.set_ylabel('Probability')

        ## Labels and such:
        ax_act[ind].set_ylabel('neuron #')
        if i_ind == 0:
            ax_act[ind].set_title('B) Neural activity r' + r"$_t$", weight='bold')
            ax_gt[ind].set_title('A) Noisy input stimuli x' + r"$_t$", weight='bold')
            ax_gt[ind].set_ylabel('Stimulus vector')
            ax_pred[ind].set_ylabel('Stimulus vector')
            ax_gt[ind].text(s=r"$\mathbf{\alpha = 1, \beta = 1}$" + '\n expected trial', va='top',
                              x=-8, y=1.3, fontdict={'weight': 'bold'}, rotation='vertical', ha='center')
            ax_pred[ind].set_title('C) Network predictions ' + r"$\hat{\mathbf{y}}_t$", weight='bold')
            ax_act[ind].set_xlabel('')
        else:
            ax_act[ind].set_title('')
            ax_gt[ind].set_ylabel('Stimulus vector')
            ax_pred[ind].set_ylabel('Stimulus vector')
            ax_gt[ind].text(s=r"$\mathbf{\alpha = 1, \beta = 2}$" + '\n unexpected trial', va='top',
                              x=-8, y=1, fontdict={'weight': 'bold'}, rotation='vertical', ha='center')
            ax_gt[ind].set_xlabel('Time ' + r"$\longrightarrow$")
            ax_act[ind].set_xlabel('Time ' + r"$\longrightarrow$")
            ax_pred[ind].set_xlabel('Time + 1 ' + r"$\longrightarrow$")

    ## Alignment:
    for ax in [ax_gt, ax_act, ax_pred]: # align labesl & titles
        for ind in ['11', '12']:
            ax[ind].yaxis.label.set_va('top')  # set ylabel alignment
            ax[ind].yaxis.set_label_coords(x=-0.25, y=0.5)
        ax['11'].title.set_ha('left')  # set title alignment
        ax['11'].title.set_position((-0.25, 1.25))
    cax_top.yaxis.set_label_coords(x=7, y=0.5)  # align color bars
    cax_bottom.yaxis.set_label_coords(x=7, y=0.5)
    if plot_conv == 1:
        ax_conv_top = fig.add_subplot(gs_bottom[0, 0])
        ax_conv_middle = fig.add_subplot(gs_bottom[1, 0])
        ax_conv_bottom = fig.add_subplot(gs_bottom[2, 0])

        ax_conv_top, ax_conv_middle, ax_conv_bottom = plot_convergence_stats(ax_left=ax_conv_top,
                                                            ax_middle=ax_conv_middle, ax_right=ax_conv_bottom)

    return (ax_gt, ax_act, ax_pred)

def plot_convergence_stats(ax_left=None, ax_middle=None, ax_right=None,
                           networksize_folder='/home/thijs/repos/rotation/models/75-25_ChangingNNodes_115models/',
                           splitloss_folder='/home/thijs/repos/rotation/models/75-25_SplitLoss_Xmodels/200epochs'):
    if ax_left is None and ax_middle is None and ax_right is None:
        fig = plt.figure(constrained_layout=False)
        gs = fig.add_gridspec(ncols=3, nrows=1, wspace=0.45)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_middle = fig.add_subplot(gs[0, 1])
        ax_right = fig.add_subplot(gs[0, 2])

    ax_left, _ = plot_network_size(rnn_folder=networksize_folder, ax=ax_left)
    _ = plot_split_perf(rnn_folder=splitloss_folder,
                        ax_top=ax_middle, ax_bottom=ax_right, plot_total=False)
    xtick_arr = np.arange(start=0, step=25, stop=101)
    for ax in [ax_middle, ax_right]:  # change xticklabels because we use 200 sample epochs instead of default 1000
        ax.set_xticks(xtick_arr)
        ax.set_xticklabels([str(int(x / 5)) for x in xtick_arr])

    for ax in [ax_left, ax_middle, ax_right]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return (ax_right, ax_middle, ax_left)

def plot_distr_networks(rnn_name_dict, rnn_folder='models/75-25_100models/', verbose=0,
                        fontsize=10):

    rnn_title_dict = {'low': 'Anti-correlated', 'med': 'Decorrelated',
                      'high': 'Correlated'}

    fig = plt.figure(constrained_layout=False)
    gs_top = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1, 1, 1.1], bottom=0.7, top=1, wspace=0.45, right=0.6)  # [1, 2.2, 1.2]
    gs_right_top = fig.add_gridspec(ncols=1, nrows=2, left=0.72, top=1, bottom=0.5, hspace=0.9)
    gs_right_bottom = fig.add_gridspec(ncols=1, nrows=1, left=0.72, top=0.27, bottom=0, hspace=0.9)
    gs_bottom = fig.add_gridspec(ncols=2, nrows=1, bottom=0, top=0.43, wspace=0.45, hspace=0.4, right=0.6)
    rnn, ax_ctmat = {}, {}
    i_rnn = 0
    for key, rnn_name in rnn_name_dict.items():
        with open(rnn_folder + rnn_name, 'rb') as f:
            rnn[key] = pickle.load(f)

        ## CT matrix
        ax_ctmat[key] = fig.add_subplot(gs_top[i_rnn])
        _, hm = plot_decoder_crosstemp_perf(score_matrix=rnn[key].decoding_crosstemp_score['alpha'],
                               ax=ax_ctmat[key], c_bar=False, fontsize_ticks=8,
                               ticklabels=double_time_labels_blank[:-1], v_max=1)
        ax_ctmat[key].set_title(rnn_title_dict[key])#, weight='bold')

        # ax_ctmat[key].set_ylabel('v) cross-temporal \ndecoding accuracy', weight='bold')
        i_rnn += 1

    ## Custom color bar:
    divider = make_axes_locatable(ax_ctmat['low'])
    cax_top = divider.append_axes('right', size='5%', pad=0.1)
    mpl_colorbar(ax_ctmat['low'].get_children()[0], cax=cax_top)
    cax_top.yaxis.set_ticks_position('right')
    # cax_mean.yaxis.set_ticks(np.linspace(0, 1, 6))
    for tick in cax_top.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')


    agg_score_alpha = bp.aggregate_score_mats(model_folder=rnn_folder, label='alpha')
    agg_score_beta = bp.aggregate_score_mats(model_folder=rnn_folder, label='beta')
    if verbose:
        print(f'shape agg: {agg_score_alpha.shape}')
    train_times, test_times = ru.get_train_test_diag()
    summ_accuracy = agg_score_alpha[:, train_times, test_times] # .mean((1, 2))  # average of patch
    if verbose:
        print(summ_accuracy.shape)
    summ_accuracy = np.mean(np.squeeze(summ_accuracy), 1)
    if verbose:
        print(summ_accuracy.shape)
    alpha_diag = np.diag(agg_score_alpha.mean(0))
    beta_diag = np.diag(agg_score_beta.mean(0))
    alpha_diag_err = np.diag(agg_score_alpha.std(0))
    beta_diag_err = np.diag(agg_score_beta.std(0))
    ## draw fig:
    ax_mean = fig.add_subplot(gs_bottom[:, 0])
    plot_decoder_crosstemp_perf(score_matrix=agg_score_alpha.mean(0), cmap_hm='BrBG', c_bar=False,
                                   ax=ax_mean, ticklabels=double_time_labels_blank[:-1], v_max=1)
    ax_mean.set_title('Average ' + r'$\alpha$' + ' accuracy')#, weight='bold')
    ## Custom color bar:
    divider = make_axes_locatable(ax_mean)
    cax_mean = divider.append_axes('right', size='5%', pad=0.1)
    mpl_colorbar(ax_mean.get_children()[0], cax=cax_mean)
    cax_mean.yaxis.set_ticks_position('right')
    # cax_mean.yaxis.set_ticks(np.linspace(0, 1, 6))
    for tick in cax_mean.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')

    ax_var = fig.add_subplot(gs_bottom[:, 1])  # variance matrix
    plot_decoder_crosstemp_perf(score_matrix=agg_score_alpha.var(0), cmap_hm='bone_r', c_bar=False,
                                   ax=ax_var, ticklabels=double_time_labels_blank[:-1], v_max=0.1)
    ax_var.set_title('Variance ' + r'$\alpha$' + ' accuracy')#, weight='bold')
    ## custom color bars:
    divider = make_axes_locatable(ax_var)
    cax_var = divider.append_axes('right', size='5%', pad=0.1)
    mpl_colorbar(ax_var.get_children()[0], cax=cax_var)
    cax_var.yaxis.set_ticks_position('right')
    # cax_var.yaxis.set_ticks(np.linspace(0, 1, 6))
    for tick in cax_var.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')

    ax_auto = {0: fig.add_subplot(gs_right_top[0, 0]), 1: fig.add_subplot(gs_right_top[1, 0])}  # alpha and beta auto-decoding
    _  = plot_alpha_beta_performance(alpha_perf=alpha_diag, beta_perf=None, ax=ax_auto[0])
    _  = plot_alpha_beta_performance(alpha_perf=None, beta_perf=beta_diag, ax=ax_auto[1])
    ax_auto[0].fill_between(x=np.arange(len(alpha_diag)), y1=alpha_diag - alpha_diag_err,
                         y2=alpha_diag + alpha_diag_err, color='#018571', alpha=0.3)
    ax_auto[1].fill_between(x=np.arange(len(beta_diag)), y1=beta_diag - beta_diag_err,
                         y2=beta_diag + beta_diag_err, color='grey', alpha=0.3)
    # ax_auto[0].set_title('Auto-temporal accuracy', weight='bold')


    ax_hist = fig.add_subplot(gs_right_bottom[0, 0])  # histogram
    ax_hist.set_xlabel('Average accuracy')
    ax_hist.set_ylabel('Frequency');
    # ax_hist.set_title('Histogram of patch', weight='bold')
    if verbose:
        print(summ_accuracy)
    n, bins, hist_patches = ax_hist.hist(summ_accuracy, color='k', bins=np.linspace(0, 1, 21),
                                         rwidth=0.9, alpha=0.9)
    ## Colour hist bars: https://stackoverflow.com/questions/23061657/plot-histogram-with-colors-taken-from-colormap
    cm = plt.cm.get_cmap('BrBG')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - np.min(bin_centers)      # scale values to interval [0,1]
    col /= np.max(col)
    for c, p in zip(col, hist_patches):
        plt.setp(p, 'facecolor', cm(c))

    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)

    ## Add patches
    color_patch  = '#8f0d1e'
    lw_patch = 3
    # ax_mean.add_patch(patches.FancyBboxPatch((test_times[0], train_times[0]),
    #                                     width=len(test_times), height=len(train_times),
    #                                    fill=False, edgecolor=color_patch, lw=lw_patch))  # patch in variance plot
    # ax_var.add_patch(patches.FancyBboxPatch((test_times[0], train_times[0]), zorder=1,
    #                                     width=len(test_times), height=len(train_times),
    #                                    fill=False, edgecolor=color_patch, lw=lw_patch))  # patch in variance plot
    # ax_hist.add_patch(patches.FancyBboxPatch((-0.18, -6),
    #                                 width=1.1, height=21.1, clip_on=False,
    #                                 fill=False, edgecolor=color_patch, lw=lw_patch)) # box around histogram
    # line_top = patches.Arc(xy=(test_times[0] + 2, train_times[0] + 1.58), width=29.8, height=3.77,
    #                         theta1=270, theta2=360, clip_on=False, linewidth=lw_patch, color=color_patch) # top connecting line
    # ax_var.add_patch(line_top)
    # line_bottom = patches.Arc(xy=(test_times[0] + 2, train_times[0] + 12.55), width=29.8, height=18.5,
    #                         theta1=270, theta2=360, clip_on=False, linewidth=lw_patch, color=color_patch)  # bottom connecting line
    # ax_var.add_patch(line_bottom)

    # ax_mean.text(s='A', x=-2, y=-1, fontdict={'weight': 'bold', 'size': 'xx-large'})
    # ax_mean.text(s='B', x=25, y=-1, fontdict={'weight': 'bold', 'size': 'xx-large'})
    # ax_mean.text(s='C', x=50, y=-1, fontdict={'weight': 'bold', 'size': 'xx-large'})
    # ax_mean.text(s='D', x=50, y=11.1, fontdict={'weight': 'bold', 'size': 'xx-large'}, zorder=2)

    # for ax in [ax_auto[0], ax_auto[1]]:
    #     _ = clip_axes_tick(ax=ax)
    # _ = clip_axes_tick(ax=ax_auto[0])
    # print(ax_auto[0].get_xticks(), ax_auto[0].get_yticks())



    fig.align_ylabels(axs=[ax_auto[0], ax_auto[1], ax_hist])
    fig.align_ylabels(axs=[ax_mean, ax_ctmat['high']])

    ax_ctmat['high'].text(s='A) Cross-temporal ' + r'$\mathbf{\alpha}$' + '-decoding accuracy', x=-5.75, y=20,
                          fontdict={'fontsize': fontsize,  'weight': 'bold'})
    ax_ctmat['high'].text(s='B) Distribution of cross-temporal '+ r'$\mathbf{\alpha}$' + '-decoding accuracy', x=-5.75, y=-8.8,
                          fontdict={'fontsize': fontsize,  'weight': 'bold'})
    ax_ctmat['high'].text(s='C) Auto-temporal accuracy', x=78.7, y=20,  # x=81
                          fontdict={'fontsize': fontsize,  'weight': 'bold'})
    ax_ctmat['high'].text(s='D) Histogram of red regime', x=78.7, y=-18.5,
                          fontdict={'fontsize': fontsize, 'weight': 'bold'})

    return None

def plot_network_size(rnn_folder, ax=None, plot_fun=None):
    if plot_fun is None:
        plot_fun = sns.pointplot
    if ax is None:
        ax = plt.subplot(111)
    df_data = ru.make_df_network_size(rnn_folder=rnn_folder)
    df_data = df_data[df_data['n_nodes'] != 2]
    n_nodes = np.sort(np.unique(df_data['n_nodes']))
    plot_fun(data=df_data, x='n_nodes', y='min_test_perf', ax=ax,
             linewidth=3, ci='sd', color='k', join=False)

    xticks = np.arange(start=2, stop=22, step=5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(n_nodes[xx])) for xx in xticks])
    ax.set_xlabel('Number of neurons ' + r'$N$')
    ax.set_ylabel('Cross entropy ' + r'$H$')
    ax.set_xlim([-1.5, 23])
    return ax, df_data

def plot_bar_switch_rnn_types(df_stable_switch, plottype='bar', ax=None, neuron_type='n_switch',
                              rnn_types_list=['prediction_only', 'mnm_acc', 'mnm_nonacc'],
                              label_dict={'prediction_only': 'Prediction only',
                                          'mnm_acc': 'Prediction + M/NM',
                                          'mnm_nonacc': 'Prediction + Match/Non-Match'}):
    if ax is None:
        ax = plt.subplot(111)

    n_unique_sw = int(df_stable_switch[neuron_type].max() + 1)
    df_switch_summary = pd.DataFrame({**{neuron_type: np.arange(n_unique_sw)},
                                      **{key: np.zeros(n_unique_sw) for key in rnn_types_list}})
    for n_sw in range(n_unique_sw):
        df_switch_summary[neuron_type].iat[n_sw] = n_sw
        for key in rnn_types_list:
            df_switch_summary[key].iat[n_sw] = len(np.where(df_stable_switch[df_stable_switch['rnn_type'] == key][neuron_type] == n_sw)[0])
    for key in rnn_types_list:
        df_switch_summary[key] /= df_switch_summary[key].sum()

    if neuron_type == 'n_switch':
        colour_types = {'prediction_only': 'k', 'mnm_acc': '#44bed4',
                        'mnm_nonacc': 'green'}
    elif neuron_type == 'n_stable':
        colour_types = {'prediction_only': 'k', 'mnm_acc': '#9c0624',
                        'mnm_nonacc': 'green'}
    n_rnns = len(rnn_types_list)
    width_dict = {2: 0.35, 3: 0.27}
    for i_type, rnn_type in enumerate(rnn_types_list):
        if plottype == 'bar':
            ax.bar(x=df_switch_summary[neuron_type] + i_type * width_dict[n_rnns] - 0.15, height=df_switch_summary[rnn_type],
                    width=width_dict[n_rnns], label=label_dict[rnn_type], color=colour_types[rnn_type])
        elif plottype == 'cumulative':
            ax.plot(df_switch_summary[rnn_type], linewidth=3, marker='o',
                    label=label_dict[rnn_type], color=colour_types[rnn_type])
            # ax.plot(np.concatenate((np.array([0]), np.cumsum(df_switch_summary[rnn_type]))), linewidth=3, marker='o',
            #         label=label_dict[rnn_type], color=colour_types[rnn_type])
    if neuron_type == 'n_switch':
        ax.legend(frameon=False, loc='upper right')#, bbox_to_anchor=(0.15, 1))
    elif neuron_type == 'n_stable':
        ax.legend(frameon=False,  bbox_to_anchor=(-0.04, 1.15), loc='upper left') #55box_to_anchor=(0.5, 1))
    if plottype == 'bar':
        ax.set_xticks(df_switch_summary[neuron_type])
    ax.set_xlabel(f'# {neuron_type[2].upper()}{neuron_type[3:]} neurons');
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plottype == 'cumulative':  # perform kilmogorov smirnov test
        # _, p_val = scipy.stats.ks_2samp(np.concatenate((np.array([0]), np.cumsum(df_switch_summary[rnn_types_list[0]]))),
        #                                 np.concatenate((np.array([0]), np.cumsum(df_switch_summary[rnn_types_list[1]]))))
        _, p_val = scipy.stats.mannwhitneyu(df_switch_summary[rnn_types_list[0]],
                                            df_switch_summary[rnn_types_list[1]],
                                            alternative='two-sided')
        print(p_val)
        ax.set_ylabel('Fraction of runs')
    else:
        ax.set_ylabel('Fraction of runs')
    return ax

def plot_mnm_stsw(df_stable_switch, ax=None, ax_nonmatch=None, rnn_type='mnm_acc'):
    df_use = df_stable_switch[df_stable_switch['rnn_type'] == rnn_type]
    if ax is None:
        ax = plt.subplot(111)

    mnm_names = {'match': 'm', 'nonmatch': 'nm'}
    mnm_axes = {'match': ax, 'nonmatch': ax_nonmatch}
    stsw_names = {'stable': 'st', 'switch': 'sw'}
    # for mnm_type, mnm_ax in mnm_axes.items():
    #     for stsw_long, stsw_short in stsw_names.items():
    #         mnm_ax.hist(df_use[f'n_{mnm_names[mnm_type]}_{stsw_short}'], histtype='step',
    #                       linewidth=2, label=stsw_long, density=True)
    #     mnm_ax.set_title(mnm_type, weight='bold')
    #     mnm_ax.spines['top'].set_visible(False)
    #     mnm_ax.spines['right'].set_visible(False)
    #     mnm_ax.set_xlabel(f'# {mnm_type} cells')
    #     mnm_ax.set_ylabel('Fraction of runs')
    # ax.legend(frameon=False)

    n_rows = 4 * len(df_use)  # split m and nm and st/sw
    df_plot = pd.DataFrame({**{x: np.zeros(n_rows, dtype='object') for x in ['mnm', 'stsw']},
                            **{x: np.zeros(n_rows) for x in ['count']}})
    save_arrs_dict = {x: np.zeros(len(df_use)) for x in ['m_st', 'nm_st', 'm_sw', 'nm_sw']}
    for i_rnn in range(len(df_use)):
        i_1 = 4 * i_rnn
        i_2 = 4 * i_rnn + 1
        i_3 = 4 * i_rnn + 2
        i_4 = 4 * i_rnn + 3
        df_plot['mnm'].iat[i_1] = 'Match'
        df_plot['mnm'].iat[i_2] = 'Non-match'
        df_plot['mnm'].iat[i_3] = 'Match'
        df_plot['mnm'].iat[i_4] = 'Non-match'
        df_plot['stsw'].iat[i_1] = 'Stable'
        df_plot['stsw'].iat[i_2] = 'Stable'
        df_plot['stsw'].iat[i_3] = 'Switch'
        df_plot['stsw'].iat[i_4] = 'Switch'
        m_total = np.maximum(df_use['n_m_st'].iat[i_rnn] + df_use['n_m_sw'].iat[i_rnn], 1)  # total number of projections going to match neuron
        nm_total = np.maximum(df_use['n_nm_st'].iat[i_rnn] + df_use['n_nm_sw'].iat[i_rnn], 1)
        st_total = np.maximum(df_use['n_stable'].iat[i_rnn], 1)  # total number of stable neurons (not necessarily going to either M or NM)
        sw_total = np.maximum(df_use['n_switch'].iat[i_rnn], 1)
        df_plot['count'].iat[i_1] = df_use['n_m_st'].iat[i_rnn] / (st_total * m_total)
        df_plot['count'].iat[i_2] = df_use['n_nm_st'].iat[i_rnn] / (st_total * nm_total)
        df_plot['count'].iat[i_3] = df_use['n_m_sw'].iat[i_rnn] / (sw_total * m_total)
        df_plot['count'].iat[i_4] = df_use['n_nm_sw'].iat[i_rnn] / (sw_total * nm_total)
        save_arrs_dict['m_st'][i_rnn] = df_plot['count'].iat[i_1]
        save_arrs_dict['nm_st'][i_rnn] = df_plot['count'].iat[i_2]
        save_arrs_dict['m_sw'][i_rnn] = df_plot['count'].iat[i_3]
        save_arrs_dict['nm_sw'][i_rnn] = df_plot['count'].iat[i_4]
    # sns.violinplot(data=df_plot, x='stsw', y='count', hue='mnm', ax=ax, split=True)
    # sns.pointplot(data=df_plot, x='stsw', y='count', hue='mnm', ax=ax, split=True)
    # p_val = {key: scipy.stats.wilcoxon(df_use[f'n_{short}_st'], df_use[f'n_{short}_sw'],
    #                                        alternative='two-sided')[1] for key, short in mnm_names.items()}
    # stsw_colours = sns.color_palette("Set1", n_colors=2)
    stsw_colours = [(156 / 256, 6 / 256, 36 / 256), # stable
                    (68 / 256, 190 / 256, 212 / 256)]  # switch
    sns.pointplot(data=df_plot, hue='stsw', y='count', x='mnm', ax=ax,
                  split=True, palette=stsw_colours)
    # p_val = {key: scipy.stats.wilcoxon(df_use[f'n_m_{short}'], df_use[f'n_nm_{short}'],
    #                                        alternative='two-sided')[1] for key, short in stsw_names.items()}
    p_val = {key: scipy.stats.wilcoxon(save_arrs_dict[f'm_{short}'], save_arrs_dict[f'nm_{short}'],
                                       alternative='two-sided')[1] for key, short in stsw_names.items()}
    p_val['nm'] = scipy.stats.wilcoxon(save_arrs_dict['nm_st'], save_arrs_dict['nm_sw'],
                                       alternative='two-sided')[1]
    p_val['m'] = scipy.stats.wilcoxon(save_arrs_dict['m_st'], save_arrs_dict['m_sw'],
                                       alternative='two-sided')[1]

    print(p_val)
    ax.text(s=f'P = {np.round(p_val["stable"], 3)}', x=0.2, y=0.064,
                  c=stsw_colours[0], fontdict={'weight': 'bold'})
    ax.text(s=f'P = {np.round(p_val["switch"], 3)}', x=0.24, y=0.095,
                  c=stsw_colours[1], fontdict={'weight': 'bold'})
    ax.text(s=f'P = {np.round(p_val["nm"], 3)}', x=1.07, y=0.078,
                  c='k', fontdict={'weight': 'bold'})
    # ax.plot([1.05, 1.05], [np.mean(save_arrs_dict['nm_st']), np.mean(save_arrs_dict['nm_sw'])], c='k', linewidth=3)
    ax.legend(frameon=False, bbox_to_anchor=(0.75, 1))
    # ax.set_title('Match neurons receive more Stable projections', weight='bold', y=1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('# projecting neurons')
    return (ax, ax_nonmatch)


def plot_figure_mnm(df_switch, rnn_mnm_folder, save_fig=False, train_times = np.arange(8, 11),
                    test_times = np.arange(6, 8), fontsize=10):


    fig = plt.figure(constrained_layout=False)
    gs_schem = fig.add_gridspec(ncols=1, nrows=1, bottom=0.7, top=1, wspace=0, right=0.5)  # [1, 2.2, 1.2]
    gs_right_top = fig.add_gridspec(ncols=1, nrows=2, left=0.54, top=0.95, bottom=0.75, hspace=0.7, right=0.79)
    gs_middle_mats = fig.add_gridspec(ncols=2, nrows=1, left=0, top=0.6, bottom=0.34, wspace=0.5, right=0.6)
    gs_middle_right = fig.add_gridspec(ncols=1, nrows=1, bottom=0.34, top=0.6, wspace=0, hspace=0, left=0.72, right=1)
    gs_bottom = fig.add_gridspec(ncols=2, nrows=1, bottom=0, top=0.2, wspace=0.5, hspace=0, right=0.6, left=0)
    gs_bottom_right = fig.add_gridspec(ncols=1, nrows=1, bottom=0, top=0.2, wspace=0.0, hspace=0, right=1, left=0.72)


    ## Schematic


    ## Convergence figs
    ax_conv_top = fig.add_subplot(gs_right_top[:, 0])
    # ax_conv_bottom = fig.add_subplot(gs_right_top[1, 0])
    # _ = plot_split_perf(rnn_folder=rnn_mnm_folder, list_top=['MNM', 'D', 'C'],
    #                     ax_top=ax_conv_top, ax_bottom=None, plot_bottom=False)
    #
    _ = plot_split_perf_custom(folder_pred='models/75-25_SplitLoss_Xmodels/1000epochs/',
                               folder_mnm='models/75-25_MNM_Xmodels/acc_mnm-only/',
                               folder_mnmpred='models/75-25_MNM_Xmodels/accumulate/',
                               ax=ax_conv_top)
    # ax_conv_top.set_xlim([0, 20])
    ax_conv_top.set_ylim([-0.05, 1.05])
    ax_conv_top.legend(frameon=False, bbox_to_anchor=(1, 1.1))# loc='upper right')
    ## MNM prediction ??


    ## Mean & var matrices #TODO: this is copy-pasted from plot_distr_networks(); make into function

    agg_score_alpha = bp.aggregate_score_mats(model_folder=rnn_mnm_folder, label='alpha')
    ax_mean = fig.add_subplot(gs_middle_mats[0, 0])
    plot_decoder_crosstemp_perf(score_matrix=agg_score_alpha.mean(0), cmap_hm='BrBG', c_bar=False,
                                   ax=ax_mean, ticklabels=double_time_labels_blank[:-1], v_max=1, fontsize_ticks=fontsize)
    ax_mean.set_title('Average ' + r'$\alpha$' + ' accuracy')#, weight='bold')
    ## Custom color bar:
    divider = make_axes_locatable(ax_mean)
    cax_mean = divider.append_axes('right', size='5%', pad=0.1)
    mpl_colorbar(ax_mean.get_children()[0], cax=cax_mean)
    cax_mean.yaxis.set_ticks_position('right')
    # cax_mean.yaxis.set_ticks(np.linspace(0, 1, 6))
    for tick in cax_mean.yaxis.get_major_ticks():
        tick.label.set_fontsize('x-small')

    ax_var = fig.add_subplot(gs_middle_mats[0, 1])  # variance matrix
    plot_decoder_crosstemp_perf(score_matrix=agg_score_alpha.var(0), cmap_hm='bone_r', c_bar=False,
                                   ax=ax_var, ticklabels=double_time_labels_blank[:-1], v_max=0.1, fontsize_ticks=fontsize)
    ax_var.set_title('Variance ' + r'$\alpha$' + ' accuracy')#, weight='bold')
    ## custom color bars:
    divider = make_axes_locatable(ax_var)
    cax_var = divider.append_axes('right', size='5%', pad=0.1)
    mpl_colorbar(ax_var.get_children()[0], cax=cax_var)
    cax_var.yaxis.set_ticks_position('right')
    # cax_var.yaxis.set_ticks(np.linspace(0, 1, 6))
    for tick in cax_var.yaxis.get_major_ticks():
        tick.label.set_fontsize('x-small')

    ## Histogram
    # summ_accuracy = agg_score_alpha[:, train_times, :][:, :, test_times].mean((1, 2))  # average of patch
    train_times, test_times = ru.get_train_test_diag()
    summ_accuracy = agg_score_alpha[:, train_times, test_times] # .mean((1, 2))  # average of patch
    print(summ_accuracy.shape)
    summ_accuracy = np.mean(np.squeeze(summ_accuracy), 1)
    print(summ_accuracy.shape)

    ax_hist = fig.add_subplot(gs_middle_right[0, 0])  # histogram
    ax_hist.set_xlabel('Average accuracy')
    ax_hist.set_ylabel('Frequency');
    # ax_hist.set_title('Histogram of patch', weight='bold')
    n, bins, hist_patches = ax_hist.hist(summ_accuracy, color='k', bins=np.linspace(0, 1, 21),
                                         rwidth=0.9, alpha=0.9)
    ## Colour hist bars: https://stackoverflow.com/questions/23061657/plot-histogram-with-colors-taken-from-colormap
    cm = plt.cm.get_cmap('BrBG')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - np.min(bin_centers)      # scale values to interval [0,1]
    col /= np.max(col)
    for c, p in zip(col, hist_patches):
        plt.setp(p, 'facecolor', cm(c))
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)

    ## Add patches
    color_patch  = '#8f0d1e'
    lw_patch = 3
    # ax_mean.add_patch(patches.FancyBboxPatch((test_times[0], train_times[0]),
    #                                     width=len(test_times), height=len(train_times),
    #                                    fill=False, edgecolor=color_patch, lw=lw_patch))  # patch in variance plot
    tmp_patch = np.zeros((16, 2))
    tmp_patch[::2, 0] = np.arange(8, 16) / 16
    tmp_patch[1::2, 0] = np.arange(8, 16) / 16
    tmp_patch[::2, 1] = np.arange(4, 12) / 16
    tmp_patch[1::2, 1] = np.arange(4, 12) / 16
    ax_mean.add_patch(patches.Polygon(xy=tmp_patch, fill=False, edgecolor=color_patch, lw=lw_patch, zorder=10))  # patch in variance plot
    # ax_var.add_patch(patches.FancyBboxPatch((test_times[0], train_times[0]), zorder=1,
    #                                     width=len(test_times), height=len(train_times),
    #                                    fill=False, edgecolor=color_patch, lw=lw_patch))  # patch in variance plot
    # ax_hist.add_patch(patches.FancyBboxPatch((0, -4),
    #                                 width=0.9, height=16, clip_on=False,
    #                                 fill=False, edgecolor=color_patch, lw=lw_patch)) # box around histogram

    ## Stable and switch cells
    ax_switch = fig.add_subplot(gs_bottom[0, 0])
    ax_stable = fig.add_subplot(gs_bottom[0, 1])
    ax_mnm = fig.add_subplot(gs_bottom_right[0, 0])
    _ = plot_bar_switch_rnn_types(df_stable_switch=df_switch, neuron_type='n_stable', plottype='cumulative',
                                     rnn_types_list=['prediction_only', 'mnm_acc'], ax=ax_stable,
                                     label_dict={'prediction_only': 'Pred-only',
                                                  'mnm_acc': 'Pred &\nM/NM'})
    _ = plot_bar_switch_rnn_types(df_stable_switch=df_switch, neuron_type='n_switch', plottype='cumulative',
                                     rnn_types_list=['prediction_only', 'mnm_acc'], ax=ax_switch,
                                     label_dict={'prediction_only': 'Pred-only',
                                                  'mnm_acc': 'Pred & M/NM'})
    _ = plot_mnm_stsw(df_stable_switch=df_switch, rnn_type='mnm_acc', ax=ax_mnm)
    ax_mnm.tick_params(axis='x', direction='out', pad=21)
    # print(*ax_mnm.get_xticklabels())
    ## Alignment:
    # for ax in [ax_mean, ax_var, ax_stable, ax_switch, ax_conv_top, ax_hist, ax_mnm]: # align labesl & titles
    #     ax.yaxis.label.set_va('top')  # set ylabel alignment
    #     ax.yaxis.label.set_ha('center')
    #     ax.yaxis.set_label_coords(x=-0.25, y=0.5)
    #     ax.title.set_ha('left')  # set title alignment
        # ax.title.set_position((-0.25, 1.25))
    # cax_top.yaxis.set_label_coords(x=7, y=0.5)  # align color bars
    # cax_bottom.yaxis.set_label_coords(x=7, y=0.5)


    fig.align_ylabels([ax_hist, ax_mnm])
    fig.align_ylabels([ax_mean, ax_switch])
    fig.align_ylabels([ax_var, ax_stable])
    fig.text(s='A) RNN prediction + Match / Non-match (M/NM) task', x=-0.065, y=1, fontdict={'weight': 'bold'})
    fig.text(s='B) ' + r'$\it{\bf{H}}$' + ' during training', x=0.48, y=1, fontdict={'weight': 'bold'})
    fig.text(s='C) Distribution of cross-temporal ' + r'$\mathbf{\alpha}}$' + ' decoding accuracy', x=-0.065, y=0.65, fontdict={'weight': 'bold'})
    fig.text(s='D) Histogram of red regime', x=0.655, y=0.65, fontdict={'weight': 'bold'})
    fig.text(s='E) Number of switch neurons', x=-0.065, y=0.24, fontdict={'weight': 'bold'})
    fig.text(s='F) Number of stable neurons', x=0.305, y=0.24, fontdict={'weight': 'bold'})
    fig.text(s='G) Match vs Non-match', x=0.655, y=0.24, fontdict={'weight': 'bold'})

    if save_fig:
        plt.savefig('figures/thesis/fig5_python.svg', bbox_to_inches='tight')
    return fig

def plot_sa_convergence(sim_an_folder, pred_folder, mnm_folder, figsize=(6, 4)):

    ratio_exp_array = None
    for i_rnn, rnn_name in enumerate(os.listdir(sim_an_folder)):
        rnn = ru.load_rnn(os.path.join(sim_an_folder, rnn_name))
        assert rnn.info_dict['simulated_annealing']
        if i_rnn == 0:
            ratio_exp_array = rnn.info_dict['ratio_exp_array']
        else:
            assert (ratio_exp_array == rnn.info_dict['ratio_exp_array']).all()
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs_conv = fig.add_gridspec(ncols=1, nrows=1, bottom=0, top=0.75, left=0, right=1)
    gs_ratio = fig.add_gridspec(ncols=1, nrows=1, bottom=0.85, top=1, left=0, right=1)
    ax_conv = fig.add_subplot(gs_conv[0])
    ax_ratio = fig.add_subplot(gs_ratio[0])
    plot_split_perf_custom(folder_pred=pred_folder,
                          folder_mnm=mnm_folder,
                          folder_mnmpred=sim_an_folder, ax=ax_conv,
                          legend_anchor=(1, 1))
    ax_ratio.plot(ratio_exp_array, linewidth=3, c='grey')
    ax_ratio.set_xticklabels([])
    for sp_name in ['top', 'right']:
        ax_ratio.spines[sp_name].set_visible(False)
    ax_ratio.set_ylabel(r'$P(\alpha = \beta)$');
    return fig
