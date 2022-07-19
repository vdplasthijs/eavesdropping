# @Author: Thijs L van der Plas <thijs>
# @Date:   2021-04-14
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Last modified by:   thijs
# @Last modified time: 2021-06-01

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.lines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar as mpl_colorbar
import seaborn as sns
import pickle, os, sys, copy
import scipy.stats
import sklearn.decomposition
import bptt_rnn_mtl as bpm
import rot_utilities as ru
import pandas as pd
from cycler import cycler
from tqdm import tqdm
## Create list with standard colors:
color_dict_stand = {}
for ii, x in enumerate(plt.rcParams['axes.prop_cycle']()):
    color_dict_stand[ii] = x['color']
    if ii > 8:
        break  # after 8 it repeats (for ever)
plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))

time_labels = ['0', '0', r'$S_1$', r'$S_1$', '0', '0', r'$S_2$', r'$S_2$', '0', '0', 'G', 'G', '0', '0']
time_labels_blank = ['' if x == '0' else x for x in time_labels]
input_vector_labels = ['0', r'$A_1$', r'$A_2$', r'$B_1$', r'$B_2$', 'G']
output_vector_labels = input_vector_labels + [r'$M_1$', r'$M_2$']

pred_only_colour = [67 / 255, 0, 0]
spec_only_colour = [207 / 255, 143 / 255, 23 / 255]
pred_spec_colour = [73 / 255, 154 / 255, 215 / 255]

def set_fontsize(font_size=12):
    """Change font size of all matplotlib items.

    Parameters
    ----------
    font_size : int/float
        new fontsie .
    """
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.autolimit_mode'] = 'data'  # default: 'data'
    params = {'legend.fontsize': font_size,
             'axes.labelsize': font_size,
             'axes.titlesize': font_size,
             'xtick.labelsize': font_size,
             'ytick.labelsize': font_size}
    plt.rcParams.update(params)

def despine(ax):
    """Despines axes

    Parameters
    ----------
    ax : ax
        ax to despine
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def naked(ax):
    for ax_name in ['top', 'bottom', 'right', 'left']:
        ax.spines[ax_name].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')


def opt_leaf(w_mat, dim=0):
    '''create optimal leaf order over dim, of matrix w_mat. if w_mat is not an
    np.array then its assumed to be a RNN layer. see also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html#scipy.cluster.hierarchy.optimal_leaf_ordering

    Parameters
    ----------
    w_mat : np array or pytorch network layer
        2D matrix to sort by
    dim: 0 or 1
        dimension to sort along

    Returns:
    ----------
    opt_leaves: np array
        sorted indices
    '''
    if type(w_mat) != np.ndarray:  # assume it's an rnn layer
        w_mat = [x for x in w_mat.parameters()][0].detach().numpy()
    assert w_mat.ndim == 2
    if dim == 1:  # transpose to get right dim in shape
        w_mat = w_mat.T
    dist = scipy.spatial.distance.pdist(w_mat, metric='euclidean')  # distanc ematrix
    link_mat = scipy.cluster.hierarchy.ward(dist)  # linkage matrix
    opt_leaves = scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.optimal_leaf_ordering(link_mat, dist))
    return opt_leaves

def plot_split_perf(rnn_name=None, rnn_folder=None, ax_top=None, ax_bottom=None,
                    normalise_start=False,
                    plot_top=True, plot_bottom=True, list_top=None, lw=3, plot_total=True,
                    label_dict_keys = {x: x for x in ['dmc', 'dms', 'pred', 'S2', 'G', 'G1', 'G2',
                                                            '0', '0_postS1', '0_postS2', '0_postG']},
                    linestyle_custom_dict={}, colour_custom_dict={},
                    plot_std=True, plot_indiv=False, max_date_bool=True):
    """Function that plots the performance (after convergence), split by loss function.
    Can take single RNN or folder of RNNs.

    Parameters
    ----------
    rnn_name : str
        filename.
    rnn_folder : str
        folder containing rnns.
    ax_top : ax
        main ax.
    ax_bottom : ax
        second ax, optional (for L1 regularisation loss i believe).
    normalise_start : bool
        whether to normalise start value.
    plot_top : bool
        whether to plot top .
    plot_bottom : bool
        same
    list_top : list
        list of split loss function names to plot
    lw : float
        linewidth
    plot_total : bool
        plot total loss function
    label_dict_keys : dict
        dictionary with legend labels .
    linestyle_custom_dict : dict
        dictionary with linestyles (per loss func).
    colour_custom_dict : dict
        same with line colors
    plot_std : bool
        plot confidence interval (shaded area)
    plot_indiv : bool
        plot individual rnn traces

    Returns:
    (ax_top, ax_bottom)
    """
    if normalise_start:
        print('Normalising loss functions')
        assert False, 'do not normalise start value'
    if ax_top is None and plot_top:
        ax_top = plt.subplot(211)
    if ax_bottom is None and plot_bottom:
        ax_bottom = plt.subplot(212)
    if rnn_folder is None:
        list_rnns = [rnn_name]
    else:
        if 'early_match' in rnn_folder:
            print('EARLY MATCH CORRECTED LIST DIR')
            list_rnns = os.listdir(rnn_folder)
        else:
            list_rnns = ru.get_list_rnns(rnn_folder=rnn_folder, max_date_bool=max_date_bool)

    ## Get loss function per RNN
    n_rnn = len(list_rnns)
    for i_rnn, rnn_name in enumerate(list_rnns):
        rnn = ru.load_rnn(rnn_name=os.path.join(rnn_folder, rnn_name))
        if i_rnn == 0:
            n_tp = rnn.info_dict['n_epochs']
            # if 'simulated_annealing' in list(rnn.info_dict.keys()) and rnn.info_dict['simulated_annealing']:
            #     pass
            # else:
            #     assert n_tp == rnn.info_dict['n_epochs']  # double check and  assume it is the same for all rnns in rnn_folder\
            conv_dict = {key: np.zeros((n_rnn, n_tp)) for key in rnn.test_loss_split.keys()}
            if plot_total:
                conv_dict['pred_sep'] = np.zeros((n_rnn, n_tp))
        else:
            assert rnn.info_dict['n_epochs'] == n_tp
        for key, arr in rnn.test_loss_split.items():
            conv_dict[key][i_rnn, :] = arr.copy()
        if plot_total:
            conv_dict['pred_sep'][i_rnn, :] = np.sum([conv_dict[key][i_rnn, :] for key in ['0', 'S2', 'G']], 0)

    ## Set stule
    i_plot_total = 0
    dict_keys = list(conv_dict.keys())[::-1]
    colour_dict_keys = {key: color_dict_stand[it] for it, key in enumerate(['S2', 'G', 'L1', 'dmc', '0', 'pred', 'pred_sep'])}
    colour_dict_keys['0'] = color_dict_stand[7]
    for key, val in colour_custom_dict.items():
        colour_dict_keys[key] = val
    linestyle_dict_keys = {x: '-' for x in label_dict_keys.keys()}
    for key, val in linestyle_custom_dict.items():
        linestyle_dict_keys[key] = val

    ## Plot
    for key in dict_keys:
        mat = conv_dict[key]
        if normalise_start:
            mat = mat / np.mean(mat[:, 0])#[:, np.newaxis]
        plot_arr = np.mean(mat, 0)
        if plot_top:
            if (list_top is not None and key in list_top) or (list_top is None and '_' not in key and 'L' not in key):
                ax_top.plot(plot_arr, label=label_dict_keys[key], linestyle=linestyle_dict_keys[key], linewidth=lw, color=colour_dict_keys[key])
                if plot_std:
                    ax_top.fill_between(x=np.arange(len(plot_arr)), y1=plot_arr - np.std(mat, 0),
                                        y2=plot_arr + np.std(mat, 0), alpha=0.2, color=colour_dict_keys[key])
                if plot_indiv:
                    for i_rnn in range(mat.shape[0]):
                        ax_top.plot(mat[i_rnn, :], label=None, linestyle=linestyle_dict_keys[key],
                                    linewidth=1, color=colour_dict_keys[key])

                i_plot_total += 1
        if plot_bottom:
            if key == 'L1':
                ax_bottom.plot(plot_arr, label=key, linestyle='-', linewidth=lw, color=colour_dict_keys[key])
                if plot_std:
                    ax_bottom.fill_between(x=np.arange(len(plot_arr)), y1=plot_arr - np.std(mat, 0),
                                        y2=plot_arr + np.std(mat, 0), alpha=0.2, color=colour_dict_keys[key])
                i_plot_total += 1
    if plot_top:
        ax_top.set_ylabel('Loss function ($H$)')
        ax_top.set_xlabel('Epoch'); #ax.set_ylabel('error relative')
        # ax_top.legend(frameon=False, bbox_to_anchor=(0.5, 0.2)); #ax.set_xlim([0, 10])
    if plot_bottom:
        ax_bottom.legend(frameon=False)
        ax_bottom.set_ylabel('L1 regularisation')
        ax_bottom.set_xlabel('Epoch'); #ax.set_ylabel('error relative')

    return (ax_top, ax_bottom)

def len_data_files(dir_path):
    """Returns number of rnns in folder

    Parameters
    ----------
    dir_path : str
        forlder path

    Returns
    -------
    int
        number of rnns
    """
    return len(ru.get_list_rnns(rnn_folder=dir_path))

def plot_split_perf_custom(folder_pred=None, folder_dmcpred=None, folder_dmc=None, ax=None,
                           plot_legend=True, legend_anchor=(1, 1), task_type='dmc',
                           plot_std=True, plot_indiv=False, plot_pred=True, plot_spec=True,
                           pred_only_colour=pred_only_colour, spec_only_colour=spec_only_colour,
                           pred_spec_colour=pred_spec_colour, linestyle_predspec_spec='-'):
    """Function that plots pred loss and spec loss for pred only, spec only and combined rnns .

    Parameters
    ----------
    folder_pred : str
        folder with pred only rnns
    folder_dmcpred : str
        folder with multitask rnns
    folder_dmc : str
        folder with spec only rnns .

    Returns
    -------
    ax

    """
    if ax is None:
        ax = plt.subplot(111)

    ## prediction only
    if folder_pred is not None and os.path.exists(folder_pred) and plot_pred:
        _ = plot_split_perf(rnn_folder=folder_pred, list_top=['pred'], lw=5,
                            linestyle_custom_dict={'pred': '-'}, colour_custom_dict={'pred': pred_only_colour},
                            plot_std=plot_std, plot_indiv=plot_indiv,
                            ax_top=ax, ax_bottom=None, plot_bottom=False,
                            label_dict_keys={'pred': f'Cat STL' + f' ({len_data_files(folder_pred)} networks)'})
                            # label_dict_keys={'pred': 'H Pred' + f'    (Pred-only, N={len_data_files(folder_pred)})'})

    ## dmc only
    if folder_dmc is not None and os.path.exists(folder_dmc) and plot_spec:
        _ = plot_split_perf(rnn_folder=folder_dmc, list_top=[task_type], lw=5, plot_total=False,
                            linestyle_custom_dict={task_type: '-'}, colour_custom_dict={task_type: spec_only_colour},
                            plot_std=plot_std, plot_indiv=plot_indiv,
                            ax_top=ax, ax_bottom=None, plot_bottom=False,
                            label_dict_keys={task_type: f'Cat STL' + f' ({len_data_files(folder_dmc)} networks)'})
                            # label_dict_keys={task_type: f'H {task_type}' + f'   ({task_type}-only, N={len_data_files(folder_dmc)})'})

    ## dmc+ prediction only
    if folder_dmcpred is not None and os.path.exists(folder_dmcpred):
        list_top = []
        if plot_pred:
            list_top.append('pred')
        if plot_spec:
            list_top.append(task_type)

        _ = plot_split_perf(rnn_folder=folder_dmcpred, list_top=list_top, lw=5,
                            linestyle_custom_dict={'pred': '--', task_type: linestyle_predspec_spec},
                            colour_custom_dict={'pred': pred_spec_colour, task_type: pred_spec_colour},
                            plot_std=plot_std, plot_indiv=plot_indiv,
                            ax_top=ax, ax_bottom=None, plot_bottom=False,
                            label_dict_keys={'pred': f'Cat MTL' + f' ({len_data_files(folder_dmcpred)} networks)',
                                             task_type: f'Cat MTL' + f' ({len_data_files(folder_dmcpred)} networks)'})
                            # label_dict_keys={'pred': f'H Pred' + f'    (Pred & {task_type},  N={len_data_files(folder_dmcpred)})',
                            #                  task_type: f'H {task_type}' + f'   (Pred & {task_type},  N={len_data_files(folder_dmcpred)})'})

    if plot_legend:
        ax.legend(frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if plot_pred:
        ax.set_ylim([-0.05, 3.5])
    else:
        ax.set_ylim([-0.05, 1.5])
    return ax

def plot_n_nodes_convergence(parent_folder='/home/tplas/repos/eavesdropping/models/sweep_n_nodes/7525/dmc_task/onehot/sparsity_5e-03/',
                   plot_legend=True, ax=None, plot_std=True, plot_indiv=False):
    """Function that plots convergence of rnns depending on n nodes
    """
    list_child_folders = os.listdir(parent_folder)
    if ax is None:
        ax = plt.subplot(111)
    for i_f, cfolder in enumerate(list_child_folders):
        n_nodes = int(cfolder.split('_')[0])
        full_folder = os.path.join(parent_folder, cfolder, 'pred_only')
        _ = plot_split_perf(rnn_folder=full_folder, list_top=['pred'], lw=5,
                            linestyle_custom_dict={'pred': '-'}, colour_custom_dict={'pred': color_dict_stand[i_f]},
                            plot_std=plot_std, plot_indiv=plot_indiv,
                            ax_top=ax, ax_bottom=None, plot_bottom=False,
                            label_dict_keys={'pred': f'N_nodes={n_nodes} N={len_data_files(full_folder)})'})
    if plot_legend:
        ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([-0.05, 1.05])

def plot_n_nodes_sweep(parent_folder='/home/tplas/repos/eavesdropping/models/sweep_n_nodes/7525/dmc_task/onehot/sparsity_1e-03/',
                  verbose=0, ax=None, method='integral', color='k', print_labels=True):
    """Function that plots performance per number of nodes

    Parameters
    ----------
    parent_folder : str
        folder containing folders with n_nodes ..
    """
    list_child_folders = os.listdir(parent_folder)
    if ax is None:
        ax = plt.subplot(111)
    learn_eff_dict = {}
    for i_f, cfolder in enumerate(list_child_folders):
        n_nodes = int(cfolder.split('_')[0])
        full_folder = os.path.join(parent_folder, cfolder, 'pred_only')
        tmp_dict = ru.compute_learning_index(rnn_folder=full_folder, list_loss=['pred'],
                                                   method=method)
        learn_eff_dict[n_nodes] = tmp_dict['pred']
    learn_eff_df = pd.DataFrame(learn_eff_dict)
    learn_eff_df = pd.melt(learn_eff_df, value_vars=[x for x in [5, 10, 15, 20, 25]])
    learn_eff_df.columns = ['n_nodes', 'learning_index']
    g = sns.pointplot(data=learn_eff_df, x='n_nodes', y='learning_index', ax=ax, color=color, alpha=0.7)
    plt.setp(g.collections, alpha=0.6)
    plt.setp(g.lines, alpha=0.6)
    if print_labels:
        ax.set_xlabel('Number of neurons')
        if method == 'integral':
            ax.set_ylabel('Speed of convergence\nof prediction task')
        elif method == 'final_loss':
            ax.set_ylabel('Final loss of\nprediction task')
        ax.set_title('Optimal network size\nfor various sparsity values', loc='left', fontdict={'weight': 'bold'})
        ax = despine(ax)

def plot_n_nodes_sweep_multiple(super_folder='/home/tplas/repos/eavesdropping/models/sweep_n_nodes/7525/dmc_task/onehot',
                                ax=None, method='integral'):
    """Function that plots n nodes sweep for multiple sparsity values

    Parameters
    ----------
    super_folder : str
        folder containing different sparsity values
    """
    if ax is None:
        ax = plt.subplot(111)
    spars_folders = os.listdir(super_folder)
    label_list = []
    for ii, spars_folder in enumerate(spars_folders):
        plot_n_nodes_sweep(parent_folder=os.path.join(super_folder, spars_folder), ax=ax,
                            method=method, print_labels=(ii == len(spars_folders) - 1),
                            color='#696969')
                            # color=color_dict_stand[ii])
        label_list.append(spars_folder.split('_')[1])

    if method == 'integral':
        ax.set_ylim([0.5, 1.05])

    ax.arrow(4.35, 0.75, 0, -0.1, head_width=0.3, head_length=0.03, linewidth=1,
              color='k', length_includes_head=True)
    ax.text(s='sparsity', x=4.55, y=0.63, rotation=90, fontsize=12)

def plot_late_s2_comparison(late_s2_folder='/home/tplas/repos/eavesdropping/models/late_s2/7525/dmc_task/onehot/sparsity_1e-03/pred_only',
                            early_s2_folder='/home/tplas/repos/eavesdropping/models/7525/dmc_task/onehot/sparsity_1e-03/pred_only',
                            method='integral', ax=None):
    """Function plotting pointplot of regular (early) s2 performance and late s2 performance.
    """
    if ax is None:
        ax = plt.subplot(111)
    learn_eff_dict = {}
    dict_early = ru.compute_learning_index(rnn_folder=early_s2_folder, list_loss=['pred'],
                                               method=method)
    learn_eff_dict['early'] = dict_early['pred']
    dict_late = ru.compute_learning_index(rnn_folder=late_s2_folder, list_loss=['pred'],
                                               method=method)
    learn_eff_dict['late'] = dict_late['pred']
    learn_eff_df = pd.DataFrame(learn_eff_dict)
    learn_eff_df = pd.melt(learn_eff_df, value_vars=['early', 'late'])
    learn_eff_df.columns = ['s2_timing', 'learning_index']
    sns.pointplot(data=learn_eff_df, x='s2_timing', y='learning_index', ax=ax, color='k', join=False)
    p_val = scipy.stats.wilcoxon(dict_early['pred'], dict_late['pred'],
                                       alternative='two-sided')[1]
    print(p_val, 'late s2')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([0.2, 0.8], [0.627, 0.627], c='k')
    if p_val < 0.01:
        ax.text(s=f'P < 10^-{str(int(ru.two_digit_sci_not(p_val)[-2:]) - 1)}', x=0.2, y=0.63)
    else:
        ax.text(s=f'n.s.', x=0.4, y=0.68)
    ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.set_ylim([-0.05, 1.6])
    ax.set_xlabel('Timing of stimulus 2')
    if method == 'integral':
        ax.set_ylabel('Speed of convergence of\nprediction task')
    elif method == 'final_loss':
        ax.set_ylabel('Final loss of\nprediction task')
    ax.set_title('Stimulus timing does not \ncause learning difference', loc='left', fontdict={'weight': 'bold'})
    ax = despine(ax)
    ax.set_ylim()

def plot_stl_mtl_comparison(dmc_only_folder='/home/tplas/repos/eavesdropping/models/7525/dmc_task/onehot/sparsity_1e-03/dmc_only/',
                            pred_dmc_folder='/home/tplas/repos/eavesdropping/models/7525/dmc_task/onehot/sparsity_1e-03/pred_dmc/',
                            method='integral', ax=None):
    """Function that quantifies eavesdropping effect (between STL and MTL networks )"""
    if ax is None:
        ax = plt.subplot(111)
    learn_eff_dict = {}
    dict_stl = ru.compute_learning_index(rnn_folder=dmc_only_folder, list_loss=['dmc'],
                                           method=method)
    learn_eff_dict['single'] = dict_stl['dmc']
    dict_mtl = ru.compute_learning_index(rnn_folder=pred_dmc_folder, list_loss=['dmc'],
                                               method=method)
    learn_eff_dict['dual'] = dict_mtl['dmc']
    learn_eff_df = pd.DataFrame(learn_eff_dict)
    learn_eff_df = pd.melt(learn_eff_df, value_vars=['single', 'dual'])
    learn_eff_df.columns = ['network_task', 'learning_index']
    sns.pointplot(data=learn_eff_df, x='network_task', y='learning_index', ax=ax, color='k', join=False)
    p_val = scipy.stats.wilcoxon(dict_stl['dmc'], dict_mtl['dmc'],
                                       alternative='two-sided')[1]
    print(p_val, 'mtl stl')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([0.2, 0.8], [0.6, 0.6], c='k')
    if p_val < 0.01:
        ax.text(s=f'P < 10^-{str(int(ru.two_digit_sci_not(p_val)[-2:]) - 1)}', x=0.2, y=0.63)
    else:
        ax.text(s=f'n.s.', x=0.4, y=0.63)
    ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.set_ylim([-0.05, 1.6])
    ax.set_xlabel('Learning paradigms')
    if method == 'integral':
        ax.set_ylabel('Speed of convergence of\nmatching task')
    elif method == 'final_loss':
        ax.set_ylabel('Final loss of\nmatching task')
    ax.set_title('Dual task networks eavesdrop\nto learn the matching task', loc='left', fontdict={'weight': 'bold'})
    ax = despine(ax)

def plot_7525_5050_comparison(folder_50='/home/tplas/repos/eavesdropping/models/5050/dmc_task/onehot/sparsity_1e-03/pred_dmc/',
                            folder_75='/home/tplas/repos/eavesdropping/models/7525/dmc_task/onehot/sparsity_1e-03/pred_dmc/',
                            method='integral', ax=None):
    """Quantify eavesdropping  difference between correlated and uncorrelated networks
    """
    if ax is None:
        ax = plt.subplot(111)
    learn_eff_dict = {}
    dict_50 = ru.compute_learning_index(rnn_folder=folder_50, list_loss=['dmc'],
                                           method=method)
    learn_eff_dict['0.50'] = dict_50['dmc']
    dict_75 = ru.compute_learning_index(rnn_folder=folder_75, list_loss=['dmc'],
                                               method=method)
    learn_eff_dict['0.75'] = dict_75['dmc']
    learn_eff_df = pd.DataFrame(learn_eff_dict)
    learn_eff_df = pd.melt(learn_eff_df, value_vars=['0.50', '0.75'])
    learn_eff_df.columns = ['ratio_alpha_beta', 'learning_index']
    sns.pointplot(data=learn_eff_df, x='ratio_alpha_beta', y='learning_index',
                  ax=ax, color='k', join=False)
    p_val = scipy.stats.wilcoxon(dict_50['dmc'], dict_75['dmc'],
                                       alternative='two-sided')[1]
    print(p_val, 'mtl stl')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([0.2, 0.8], [0.6, 0.6], c='k')
    if p_val < 0.01:
        ax.text(s=f'P < 10^-{str(int(ru.two_digit_sci_not(p_val)[-2:]) - 1)}', x=0.2, y=0.63)
    else:
        ax.text(s=f'n.s.', x=0.4, y=0.63)
    ax.set_xlim(xlim)
    ax.set_ylim([-0.05, 1.6])
    # ax.set_xlabel('Ratio ' + r"$\alpha$" + '/' + r"$\beta$")
    ax.set_xlabel(r'$P(\alpha = \beta)$')
    if method == 'integral':
        ax.set_ylabel('Speed of convergence of\nmatching task')
    elif method == 'final_loss':
        ax.set_ylabel('Final loss of\nmatching task')
    ax.set_title('Stimulus predictability is\nrequired for eavesdropping', loc='left', fontdict={'weight': 'bold'})
    ax = despine(ax)

def plot_example_trial(trial, ax=None, yticklabels=output_vector_labels,
                       xticklabels=time_labels_blank[1:], c_bar=True,
                       vmin=None, vmax=None, c_map='magma', print_labels=True):
    """Plot raster of one trial.
    """
    if ax is None:
        ax = plt.subplot(111)

    sns.heatmap(trial.T, yticklabels=yticklabels, cmap=c_map, cbar=c_bar,
    rasterized=True, linewidths=0,
            xticklabels=xticklabels, ax=ax, vmin=vmin, vmax=vmax, )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_yticklabels(labels=yticklabels, rotation=0)
    ax.set_xticklabels(labels=xticklabels, rotation=0)
    if print_labels:
        ax.set_xlabel('Time')
        ax.set_ylabel('Stimulus vector')
    return ax

def plot_effect_eavesdropping_learning(task='dmc', ratio_exp_str='7525', nature_stim='onehot',
                                       sparsity_str='1e-03', ax=None, plot_legend=True, verbose=0,
                                       plot_std=True, plot_indiv=False, plot_pred=True, plot_spec=True,
                                       plot_title=True):
    """Function that shows convergence of rnns. Parameters define the folder that are loaded
    and passed on to plot_split_perf()

    Parameters
    ----------
    task : str
        task type dmc dms dmrc dmrs
    ratio_exp_str : str
        stim correlation ratio
    nature_stim : str
        onehot or periodic
    sparsity_str : str
        sparsity value

    """
    base_folder = f'models/{ratio_exp_str}/{task}_task/{nature_stim}/sparsity_{sparsity_str}/'
    # print('USING SAVED STATE')
    folders_dict = {}
    folders_dict['pred_only'] = base_folder + 'pred_only/'
    folders_dict[f'{task}_only'] = base_folder + f'{task}_only/'
    folders_dict[f'pred_{task}'] = base_folder + f'pred_{task}/'
    # print(folders_dict)
    plot_split_perf_custom(folder_pred=folders_dict['pred_only'],
                          folder_dmc=folders_dict[f'{task}_only'],
                          folder_dmcpred=folders_dict[f'pred_{task}'],
                          plot_std=plot_std, plot_indiv=plot_indiv,
                          task_type=task, ax=ax, plot_legend=plot_legend,
                          plot_pred=plot_pred, plot_spec=plot_spec)
    if plot_title:
        plt.title(task + r'$\; P(\alpha = \beta) = $' + f'0.{ratio_exp_str[:2]},' + r'$ \; \; \lambda=$' + f'{sparsity_str}');

    if verbose > 0:

        for key, folder_rnns in folders_dict.items():
            if os.path.exists(folder_rnns):
                list_keys = key.split('_')
                if 'only' in list_keys:
                    list_keys.remove('only')
                learn_eff = ru.compute_learning_index(rnn_folder=folder_rnns,
                                                     list_loss=list_keys)
                print(key, {x: (np.round(np.mean(learn_eff[x]), 4), np.round(np.std(learn_eff[x]), 4)) for x in list_keys})

def plot_learning_efficiency(task_list=['dms', 'dmc'], plot_difference=False, indicate_sparsity=False,
                             method='integral', nature_stim_list=['periodic', 'onehot'], ax=None,
                             plot_custom_legend=False, plot_title=False, leg_anchor=(0, 1.05), leg_cols=2,
                             new_x_axis_df=None, use_gridsweep_rnns=False, gridsweep_n_nodes='n_nodes_20',
                             plot_pred_only=False):
    """Function that plots eavesdropping effect as a function of sparsity. Either plot
    both STL and MTl, or plot difference between these two. Can take multiple tasks and nature-stim


    Parameters
    ----------
    plot_difference : bool
        whether to plot difference
    indicate_sparsity : bool
        add clarification on x axis label of what is sparser
    new_x_axis_df : pd.df
        dataframe that contains the mapping of sparsity value (that would be on x axis) to number of non zero connections

    """
    df = ru.calculate_all_learning_eff_indices(method=method, task_list=task_list,
                                                nature_stim_list=nature_stim_list,
                                                use_gridsweep_rnns=use_gridsweep_rnns,
                                                gridsweep_n_nodes=gridsweep_n_nodes,
                                                eval_pred_loss_only=plot_pred_only)
    if plot_pred_only:
        assert plot_difference is False
        task_list = ['pred']  # change for plotting
        colour_dict = {'single': pred_only_colour,
                       'multi': pred_spec_colour}
    else:
        colour_dict = {'single': spec_only_colour,
                       'multi': pred_spec_colour}
    if ax is None:
        fig, ax = plt.subplots(1, len(nature_stim_list), figsize=(6 * len(nature_stim_list), 3), gridspec_kw={'wspace': 0.7})
    if len(nature_stim_list) == 1:
        ax = [ax]
    # if plot_pred_only:
    #     colour_dict = {'pred'}    
    # else:
    if use_gridsweep_rnns:
        alpha_line = np.power(int(gridsweep_n_nodes.split('_')[-1]) / 100, 0.67)
    else:
        alpha_line = 1
    i_plot = 0
    if plot_difference:
        tmp_df = df[[x[:4] != 'pred' for x in df['loss_comp']]].groupby(['task', 'nature_stim', 'setting','sparsity']).mean()  # compute mean for each set of conditions [ across simulations]
        multi_rows = [True if x[2] == 'multi' else False for x in tmp_df.index]  # select multitask settings
        tmp_df['learning_eff'][multi_rows] *= -1   # multiple effiency with -1 so the difference can be computed using condition-specific sum
        tmp_df = tmp_df.groupby(['task', 'nature_stim', 'sparsity']).sum()  # effectively comppute difference
        tmp_df.reset_index(inplace=True)  # bring multi indexes back to column values
        print(tmp_df['learning_eff'].sum(), task_list, nature_stim_list)
        if new_x_axis_df is not None:
            assert len(new_x_axis_df) == len(tmp_df)
            tmp_df = pd.merge(tmp_df, new_x_axis_df, on='sparsity')
            xaxis_name = 'number_nonzero'
        else:
            xaxis_name = 'sparsity'
        for i_nat, nat in enumerate(nature_stim_list):
            sns.lineplot(data=tmp_df[tmp_df['nature_stim'] == nat], x=xaxis_name, y='learning_eff',
                         style='task', ax=ax[i_plot], color='k', linewidth=3, 
                         markers=True,  err_kws={'alpha':0.1}, label='Difference',
                         **{'alpha': alpha_line})
            i_plot += 1
    else:
        spec_task_df = df[[x.split('_')[0] in task_list for x in df['loss_comp']]]
        if new_x_axis_df is not None:
            # assert len(new_x_axis_df) == len(tmp_df)
            spec_task_df = pd.merge(spec_task_df, new_x_axis_df, on='sparsity')
            xaxis_name = 'number_nonzero'
        else:
            xaxis_name = 'sparsity'
        for i_nat, nat in enumerate(nature_stim_list):
            # print(spec_task_df)
            sns.lineplot(data=spec_task_df[spec_task_df['nature_stim'] == nat], x=xaxis_name, y='learning_eff',
                         hue='setting', markers=True, ci=95, linewidth=1.5, style='loss_comp' if plot_pred_only else 'task',
                         ax=ax[i_plot], hue_order=['multi', 'single'], palette=colour_dict,
                         err_kws={'alpha': 0.1}, **{'alpha': alpha_line})
            i_plot += 1
    for i_plot in range(len(ax)):
        if len(nature_stim_list) > 1:
            ax[i_plot].legend(bbox_to_anchor=(1.4, 1), loc='upper right')
            if plot_title:
                ax[i_plot].set_title(nature_stim_list[i_plot], loc='left', fontdict={'weight': 'bold'})
        else:
            if plot_title:
                ax[i_plot].set_title('Sparsity-dependent eavesdropping\n', loc='left', fontdict={'weight': 'bold'})
            ax[i_plot].get_legend().remove()
        if new_x_axis_df is None:
            ax[i_plot].set_xlabel('Sparsity regularisation')
            ax[i_plot].set_xscale('symlog', linthreshx=2e-6)
        else:
            ax[i_plot].set_xlabel('Fraction of nonzero connections')

        if method == 'final_loss':
            ax[i_plot].set_ylim([-0.02, 1.2])
        elif method == 'integral':
            ax[i_plot].set_ylim([-0.02, 1.5])
        if indicate_sparsity and new_x_axis_df is None:

            ax[i_plot].arrow(0.015, -0.32, 0.05,0, head_width=0.07, head_length=0.02, linewidth=1,
                      color='k', length_includes_head=True, clip_on=False)
            ax[i_plot].text(s='sparser', x=0.014, y=-0.45, fontsize=12)

        despine(ax[i_plot])

        if method == 'integral':
            ax[i_plot].set_ylabel('Speed of convergence of\nmatching task')
        elif method == 'final_loss':
            ax[i_plot].set_ylabel('Final loss of\nmatching task')

        if plot_custom_legend:
            if plot_pred_only:
                custom_lines = [matplotlib.lines.Line2D([0], [0], color=colour_dict['single'], lw=1.5),
                                matplotlib.lines.Line2D([0], [0], color=colour_dict['multi'], lw=1.5, linestyle='--'),
                                matplotlib.lines.Line2D([0], [0], color='k', lw=1.5)]
                ax[i_plot].legend(custom_lines, ['single', 'dual', 'optimal'], frameon=False,
                                loc='upper left', bbox_to_anchor=leg_anchor, ncol=3)
            else:
                custom_lines = [matplotlib.lines.Line2D([0], [0], color=colour_dict['single'], lw=1.5),
                                matplotlib.lines.Line2D([0], [0], color=colour_dict['multi'], lw=1.5),
                                matplotlib.lines.Line2D([0], [0], color='k', lw=3)]
                ax[i_plot].legend(custom_lines, ['single', 'dual', 'difference'], frameon=False,
                                loc='upper left', bbox_to_anchor=leg_anchor, ncol=leg_cols)
    return df

def plot_learning_efficiency_matrix_sweep(learn_eff_df=None):
    if learn_eff_df is None:
        learn_eff_df = ru.calculate_all_learning_eff_indices_gridsweep()

    ## Single match task;
    tmp_df = learn_eff_df.drop(columns=['task', 'nature_stim', 'setting'])
    tmp_df = tmp_df[tmp_df['loss_comp'] == 'dmc_single']

    tmp_df = tmp_df.groupby(['loss_comp', 'sparsity', 'n_nodes']).mean()
    tmp_df = tmp_df.reset_index()
    tmp_mat_single = tmp_df.pivot(index='n_nodes', columns='sparsity', values='learning_eff')

    ## Dual task;
    tmp_df = learn_eff_df.drop(columns=['task', 'nature_stim', 'setting'])
    tmp_df = tmp_df[tmp_df['loss_comp'] == 'dmc_multi']

    tmp_df = tmp_df.groupby(['loss_comp', 'sparsity', 'n_nodes']).mean()
    tmp_df = tmp_df.reset_index()
    tmp_mat_multi = tmp_df.pivot(index='n_nodes', columns='sparsity', values='learning_eff')

    # print(f'Overall min: {np.minimum(tmp_mat_single.min(), tmp_mat_multi.min())}')
    tmp_mat_diff = tmp_mat_multi - tmp_mat_single

    fig, ax = plt.subplots(1, 3, figsize=(12, 2.5), gridspec_kw={'wspace': 0.5})

    vmin = 0.0
    vmax = 1.2

    sns.heatmap(data=tmp_mat_single, vmin=vmin, vmax=vmax, cmap='inferno',
                ax=ax[0], cbar_kws={'label': 'Match Loss'})
    ax[0].set_title('Single match task')


    sns.heatmap(data=tmp_mat_multi, vmin=vmin, vmax=vmax, cmap='inferno',
                ax=ax[1], cbar_kws={'label': 'Match Loss'})
    ax[1].set_title('Dual task')


    sns.heatmap(data=tmp_mat_diff, cbar_kws={'label': 'Loss Dual - Loss Single'},
                ax=ax[2], cmap='PiYG', vmin=-1, vmax=1)
    ax[2].set_title('Difference (dual - single)')


    for i_plot in range(3):
        bottom, top = ax[i_plot].get_ylim()
        ax[i_plot].set_ylim(bottom + 0.5, top - 0.5)
        ax[i_plot].invert_yaxis()
        ax[i_plot].set_ylabel('Number of neurons')
        ax[i_plot].set_xlabel('Sparsity parameter ' + r"$\lambda$")
        ax[i_plot].set_yticklabels(ax[i_plot].get_yticklabels(), rotation=0)
        

def plot_bar_plot_all_tasks(ax=None, method='final_loss', save_fig=False):
    """Function that plots the bar plots showing the cumulative eavesdropping
    effect across sparsity values

    """

    if ax is None:
        ax = plt.subplot(111)
    task_nat_comb = (['dmc', 'onehot'], ['dmc', 'periodic'], ['dmrc', 'periodic'],
                     ['dms', 'onehot'], ['dms', 'periodic'], ['dmrs', 'periodic'])
    cum_eavesdropping_effect = {}

    bar_width = 0.4
    bar_locs = np.arange(3)
    color_same = '#195190FF'
    color_different = '#A2A2A1FF'
    same_arr, diff_arr = np.zeros(3), np.zeros(3)
    i_same, i_diff = 0, 0
    for i_cond, cond in enumerate(task_nat_comb):
        task = cond[0]
        nat = cond[1]
        key = task + '_' + nat
        df = ru.calculate_all_learning_eff_indices(method=method, task_list=[task],
                                                    nature_stim_list=[nat])

        tmp_df = df[[x[:4] != 'pred' for x in df['loss_comp']]].groupby(['task', 'nature_stim', 'setting','sparsity']).mean()  # compute mean for each set of conditions [ across simulations]
        multi_rows = [True if x[2] == 'multi' else False for x in tmp_df.index]  # select multitask settings
        tmp_df['learning_eff'][multi_rows] *= -1   # multiple effiency with -1 so the difference can be computed using condition-specific sum
        tmp_df = tmp_df.groupby(['task', 'nature_stim', 'sparsity']).sum()  # effectively comppute difference
        tmp_df.reset_index(inplace=True)  # bring multi indexes back to column values

        cum_eavesdropping_effect[key] = tmp_df['learning_eff'].sum()

        if task in ['dmc', 'dmrc']:  # same
            same_arr[i_same] = cum_eavesdropping_effect[key]
            i_same += 1
        else:
            diff_arr[i_diff] = cum_eavesdropping_effect[key]
            i_diff += 1

    ax.bar(bar_locs - bar_width / 2, same_arr, width=bar_width,
           label='Same input domain', color=color_same)
    ax.bar(bar_locs + bar_width / 2, diff_arr, width=bar_width,
           label='Different input domain', color=color_different)
    ax.set_xlabel('Task complexity ' + r'$\to$', fontdict={'weight': 'bold'})
    ax.set_ylabel('Cumulative\neavesdropping effect')
    ax.set_xticks(bar_locs)
    ax.set_xticklabels(['2 samples\nMatch task', '4 samples\nMatch task', '4 samples\nRotated match task'])
    ax.legend(frameon=False, loc='upper right')
    ax.set_title('Eavesdropping effect decreases with\nincreasing match task complexity',
                 fontdict={'weight': 'bold'}, loc='left')
    despine(ax)
    if save_fig:
        plt.savefig('figures/fig3_other-tasks_v2.pdf', bbox_inches='tight')


def plot_bar_plot_all_tasks_splitup(ax=None, method='final_loss', save_fig=False):
    """Function that plots the bar plots showing the cumulative eavesdropping
    effect across sparsity values

    """

    if ax is None:
        ax = plt.subplot(111)
    task_nat_comb = (['dmc', 'onehot'], ['dms', 'onehot'], ['dmc', 'periodic'], 
                     ['dms', 'periodic'], ['dmrc', 'periodic'], ['dmrs', 'periodic'])

    bar_width = 0.4
    bar_locs = np.arange(6)
    single_arr, multi_arr = np.zeros(6), np.zeros(6)
    i_single, i_multi = 0, 0
    for i_cond, cond in enumerate(task_nat_comb):
        task = cond[0]
        nat = cond[1]
        key = task + '_' + nat
        df = ru.calculate_all_learning_eff_indices(method=method, task_list=[task],
                                                    nature_stim_list=[nat])
        tmp_df = df[[x[:4] != 'pred' for x in df['loss_comp']]].groupby(['task', 'nature_stim', 'setting','sparsity']).mean()  # compute mean for each set of conditions [ across simulations]
        tmp_df = tmp_df.groupby(['setting']).sum()  # sum by network type 
        tmp_df = tmp_df.reset_index()

        single_arr[i_single] = tmp_df[tmp_df['setting'] == 'single']['learning_eff']
        i_single += 1
        multi_arr[i_multi] = tmp_df[tmp_df['setting'] == 'multi']['learning_eff']
        i_multi += 1
    print(multi_arr, single_arr)
    ax.bar(bar_locs - bar_width / 2, single_arr, width=bar_width,
           label='single', color=spec_only_colour)
    ax.bar(bar_locs + bar_width / 2, multi_arr, width=bar_width,
           label='dual', color=pred_spec_colour)
    ax.set_xlabel('Task complexity ' + r'$\to$', fontdict={'weight': 'bold'})
    ax.set_ylabel('Cumulative loss across\nsparsity values')
    ax.set_xticks(bar_locs )
    ax.set_xticklabels(['DMC', 'DMS', '4 sample DMC', '4 sample DMS', '4 sample rotated DMC', '4 sample rotated DMS'], 
                        rotation=30, ha='right')
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.15, 1.25))
    ax.set_title('Eavesdropping effect decreases with\nincreasing match task complexity',
                 fontdict={'weight': 'bold'}, loc='left')
    ax.arrow(0.1, 7.365, 0, 2.465, head_width=0.1, head_length=0.4, linewidth=1.5,
              color='k', length_includes_head=True)  
    ax.arrow(0.1, 7.365, 0, -2.665, head_width=0.1, head_length=0.4, linewidth=1.5,
              color='k', length_includes_head=True)  
    ax.text(s='eavesdropping\neffect', x=0.2, y=7.4, fontdict={'rotation': 90, 'va': 'center', 'size': 10})
    despine(ax)
    if save_fig:
        plt.savefig('figures/fig3_other-tasks_v4.svg', bbox_inches='tight')

def plot_sa_convergence(sa_folder_list=['/home/tplas/repos/eavesdropping/models/simulated_annealing/7525/dmc_task/onehot/sparsity_1e-03/pred_dmc'],
                        figsize=None, plot_std=True, plot_indiv=False):
    """Function plotting convergence of simulated annealing networks by plotting both
    ratio_expected-array and loss function, for the foldres given in the list .
    """
    if figsize is None:
        figsize = (5 * len(sa_folder_list), 3)
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs_conv = fig.add_gridspec(ncols=len(sa_folder_list), nrows=1, bottom=0, top=0.75, left=0, right=1, wspace=0.3)
    gs_ratio = fig.add_gridspec(ncols=len(sa_folder_list), nrows=1, bottom=0.85, top=1, left=0, right=1, wspace=0.3)

    ax_conv, ax_ratio, ratio_exp_array = {}, {}, {}
    letters = ['A', 'B', 'C', 'D']
    for i_col, sa_folder in enumerate(sa_folder_list):

        ax_conv[i_col] = fig.add_subplot(gs_conv[i_col])
        ax_ratio[i_col] = fig.add_subplot(gs_ratio[i_col])
        list_rnns = ru.get_list_rnns(rnn_folder=sa_folder)
        for i_rnn, rnn_name in enumerate(list_rnns):
            rnn = ru.load_rnn(os.path.join(sa_folder, rnn_name))
            assert rnn.info_dict['simulated_annealing']
            if i_rnn == 0:
                ratio_exp_array[i_col] = rnn.info_dict['ratio_exp_array']
            else:
                assert (ratio_exp_array[i_col] == rnn.info_dict['ratio_exp_array']).all()


        plot_split_perf_custom(folder_pred=None,
                               folder_dmc=None,
                               folder_dmcpred=sa_folder,
                               plot_std=plot_std, plot_indiv=plot_indiv,
                               task_type='dmc', ax=ax_conv[i_col], plot_legend=False,
                               plot_pred=False, plot_spec=True)
        ax_ratio[i_col].plot(ratio_exp_array[i_col], linewidth=3, c='grey')
        ax_ratio[i_col].set_xticklabels([])
        ax_ratio[i_col].set_ylim([0.45, 0.85])
        ax_conv[i_col].set_ylabel('Final loss')
        despine(ax_ratio[i_col])
        ax_ratio[i_col].text(s=letters[i_col], x=-40, y=1.2, fontdict={'weight': 'bold'})
        ax_ratio[i_col].set_ylabel(r'$P(\alpha = \beta)$');
        fig.align_ylabels(axs=[ax_ratio[i_col], ax_conv[i_col]])
    ax_ratio[0].set_title('Simulated annealing of stimulus predictability\n' + r'$\mathbf{P(\alpha = \beta)}$' + ' enables RNNs to learn the matching task',# with ' + r'$\mathbf{P(\alpha = \beta) = 0.5}$',
                            fontdict={'weight': 'bold'}, loc='left')
    if len(sa_folder_list) == 2:
        ax_ratio[1].set_title('Whereas tasks with a constant ' + r'$\mathbf{P(\alpha = \beta) = 0.5}$' + ' do not\nlearn to solve the task',
                                fontdict={'weight': 'bold'}, loc='left')
    return fig


def plot_sa_convergence_small(sa_folder_list=['/home/tplas/repos/eavesdropping/models/simulated_annealing/7525/dmc_task/onehot/sparsity_1e-03/pred_dmc'],
                        figsize=None, plot_std=True, plot_indiv=False, color_list=[pred_spec_colour, '#6ecba6ff']):
    """Function plotting convergence of simulated annealing networks by plotting both
    ratio_expected-array and loss function, for the foldres given in the list .
    """
    if figsize is None:
        figsize = (4, 3)
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs_conv = fig.add_gridspec(ncols=1, nrows=1, bottom=0, top=0.75, left=0, right=1, wspace=0.3)
    gs_ratio = fig.add_gridspec(ncols=1, nrows=1, bottom=0.85, top=1, left=0, right=1, wspace=0.3)

    ax_conv = fig.add_subplot(gs_conv[0])
    ax_ratio = fig.add_subplot(gs_ratio[0])
    ratio_exp_array = {}
    for i_col, sa_folder in enumerate(sa_folder_list):
        list_rnns = ru.get_list_rnns(rnn_folder=sa_folder)
        for i_rnn, rnn_name in enumerate(list_rnns):
            rnn = ru.load_rnn(os.path.join(sa_folder, rnn_name))
            assert rnn.info_dict['simulated_annealing']
            if i_rnn == 0:
                ratio_exp_array[i_col] = rnn.info_dict['ratio_exp_array']
            else:
                assert (ratio_exp_array[i_col] == rnn.info_dict['ratio_exp_array']).all()

        plot_split_perf_custom(folder_pred=None,
                               folder_dmc=None,
                               folder_dmcpred=sa_folder,
                               plot_std=plot_std, plot_indiv=plot_indiv,
                               task_type='dmc', ax=ax_conv, plot_legend=False,
                               plot_pred=False, plot_spec=True,
                               pred_spec_colour=color_list[i_col],
                               linestyle_predspec_spec=('-' if i_col == 0 else '--'))
        ax_ratio.plot(ratio_exp_array[i_col], linewidth=3, linestyle=('-' if i_col == 0 else '--'), c=color_list[i_col])
    ax_ratio.set_xticklabels([])
    ax_ratio.set_ylim([0.45, 0.85])
    ax_conv.set_ylabel('Final loss')
    despine(ax_ratio)
    ax_ratio.set_ylabel(r'$P(\alpha = \beta)$');
    fig.align_ylabels(axs=[ax_ratio, ax_conv])
    ax_ratio.set_title('Simulated annealing of stimulus predictability\n' + r'$\mathbf{P(\alpha = \beta)}$' + ' enables RNNs to learn the matching task',# with ' + r'$\mathbf{P(\alpha = \beta) = 0.5}$',
                            fontdict={'weight': 'bold'}, loc='left')
    return fig

def plot_autotemp_s1_decoding(parent_folder='/home/tplas/repos/eavesdropping/models/7525/dmc_task/onehot/sparsity_1e-03/',
                              ax=None, plot_legend=False):
    """Plot autotemporal decoding accuracy for pred only, spec only and pred spec networks .
    """

    if ax is None:
        ax = plt.subplot(111)

    child_folders = os.listdir(parent_folder)

    for cf in child_folders:
        rnn_folder = os.path.join(parent_folder, cf + '/')
        bpm.train_multiple_decoders(rnn_folder=rnn_folder, ratio_expected=0.5,
                                    n_samples=None, ratio_train=0.8, label='s1',
                                    reset_decoders=True, skip_if_already_decoded=True)  # check if decoding has been done before
    autotemp_dec_dict = {}
    n_tp = 13  # for t_stim = 2 and t_dleay = 2

    colour_dict = {'dmc_only': spec_only_colour, 'pred_only': pred_only_colour,
                  'pred_dmc': pred_spec_colour}
    label_dict = {'dmc_only': 'single cat', 'pred_only': 'single pred',
                  'pred_dmc': 'dual task'}
    for cf in child_folders:
        rnn_folder = os.path.join(parent_folder, cf + '/')
        list_rnns =  ru.get_list_rnns(rnn_folder=rnn_folder)
        autotemp_dec_dict[cf] = np.zeros((len(list_rnns), n_tp))
        for i_rnn, rnn_name in enumerate(list_rnns):
            rnn = ru.load_rnn(os.path.join(rnn_folder, rnn_name))
            autotemp_score = rnn.decoding_crosstemp_score['s1'].diagonal()
            autotemp_dec_dict[cf][i_rnn, :] = autotemp_score
        mean_dec = autotemp_dec_dict[cf].mean(0)
        std_dec = autotemp_dec_dict[cf].std(0)
        ax.plot(mean_dec, linewidth=3, label=label_dict[cf], c=colour_dict[cf])
        ax.fill_between(x=np.arange(n_tp), y1=mean_dec - std_dec, y2=mean_dec + std_dec, alpha=0.3, facecolor=colour_dict[cf])
    if plot_legend:
        ax.legend(frameon=False)
    ax.set_xticks(np.arange(n_tp))
    ax.set_xticklabels(time_labels_blank[:-1])
    ax.set_ylim([0.45, 1.05])
    ax.set_xlabel('Time')
    ax.set_ylabel('S1 decoding\naccuracy')
    # ax.set_title('S1 memory')
    despine(ax)


def plot_autotemp_all_reps_decoding(rnn_folder='/home/tplas/repos/eavesdropping/models/7525/dms_task/onehot/sparsity_1e-04/pred_dms/',
                              ax=None, plot_legend=True, reset_decoders=True, skip_if_already_decoded=False):
    """Plot autotemporal decoding for S1, S2 and MNM """
    if ax is None:
        ax = plt.subplot(111)


    for i_rep, rep in enumerate(['s1', 's2', 'go']):
        print(rep)
        if reset_decoders:
            res = (i_rep == 0)
        else:
            res = False
        bpm.train_multiple_decoders(rnn_folder=rnn_folder, ratio_expected=0.5,
                                    n_samples=None, ratio_train=0.8, label=rep,
                                    reset_decoders=res, skip_if_already_decoded=skip_if_already_decoded)  # check if decoding has been done before
    autotemp_dec_dict = {}
    n_tp = 13  # for t_stim = 2 and t_dleay = 2
    colour_dict = {'s1': pred_spec_colour, 's2': pred_spec_colour,
                  'go': pred_spec_colour}
    linestyle_dict = {'s1': '-', 's2': '--', 'go': ':'}
    label_dict = {'s1': 'S1', 's2': 'S2', 'go': 'M/NM'}
    for i_rep, rep in enumerate(['go', 's1', 's2']):
        list_rnns =  ru.get_list_rnns(rnn_folder=rnn_folder)
        autotemp_dec_dict[rep] = np.zeros((len(list_rnns), n_tp))
        for i_rnn, rnn_name in enumerate(list_rnns):
            rnn = ru.load_rnn(os.path.join(rnn_folder, rnn_name))
            autotemp_score = rnn.decoding_crosstemp_score[rep].diagonal()
            autotemp_dec_dict[rep][i_rnn, :] = autotemp_score
        mean_dec = autotemp_dec_dict[rep].mean(0)
        std_dec = autotemp_dec_dict[rep].std(0)
        ax.plot(mean_dec, linewidth=3, label=label_dict[rep], c=colour_dict[rep], linestyle=linestyle_dict[rep])
        ax.fill_between(x=np.arange(n_tp), y1=mean_dec - std_dec, y2=mean_dec + std_dec,
                        alpha=0.3, facecolor=colour_dict[rep])
    if plot_legend:
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(-0.015, 1.2), ncol=2)
    ax.set_xticks(np.arange(n_tp))
    ax.set_xticklabels(time_labels_blank[:-1])
    ax.set_ylim([0.45, 1.05])
    ax.set_xlabel('Time')
    ax.set_ylabel('Decoding\naccuracy')
    ax.set_title('Memory of S1, S2 and M/NM\n', loc='left', fontdict={'weight': 'bold'})
    despine(ax)

def plot_correlation_matrix(rnn, representation='s1', ax=None, hard_reset=False,
                            plot_mat=True, alpha=1, plot_diag=True, plot_cbar=True):
    """Plot cross correlation matrix of rnn, of given representation. If plot_mat is False ,
    then plot the neural code (S1 cross corr over time). if hard_reset, recompute cross corr.
    """
    if ax is None:
        ax = plt.subplot(111)

    if hard_reset:
        bpm.save_pearson_corr(rnn=rnn, representation=representation)
    else:
        ru.ensure_corr_mat_exists(rnn=rnn, representation=representation)

    n_tp = 13
    if plot_mat:
        full_mat = copy.deepcopy(rnn.rep_corr_mat_dict[representation])
        plot_mat = full_mat[2:-4, :][:, 2:-4]
        mask = np.zeros_like(plot_mat)
        if plot_diag:
            mask[np.tril_indices_from(plot_mat, k=-1)] = True
        else:
            mask[np.tril_indices_from(plot_mat, k=0)] = True

        ## Plot full cross correlation matrix:
        # sns.heatmap(full_mat, cmap='BrBG',
        #             ax=ax, xticklabels=time_labels_blank[:-1], yticklabels=time_labels_blank[:-1],
        #             cbar='BrBG', vmin=-1, vmax=1)

        ## Plot triangular matrix:
        sns.heatmap(plot_mat, cmap='BrBG', mask=mask, rasterized=True, linewidths=0,
                    ax=ax, xticklabels=time_labels_blank[2:-5], yticklabels=time_labels_blank[2:-5],
                    cbar=plot_cbar, vmin=-1, vmax=1, square=True)
        ax.set_yticklabels(rotation=0, labels=ax.get_yticklabels())
        ax.set_ylabel('Time')
        ax.set_xlabel('Time')
        ax.invert_yaxis()
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom - 0.5, top + 1.5)
    else:
        assert representation == 's1', 'other rep time indexing not implemented'

        if 's1' not in rnn.decoding_crosstemp_score.keys():  # if not pre-trained, do it now
            score_mat, _, __ = bpm.train_single_decoder_new_data(rnn=rnn, save_inplace=True,
                                                                label=representation)          ## calculate autotemp score
        autotemp = rnn.decoding_crosstemp_score['s1'].diagonal()
        code = rnn.rep_corr_mat_dict[representation][np.array([2, 3]), :].mean(0)
        # print(autotemp, code, len(autotemp), len(code))
        ax.plot([0, 13], [0, 0], ':', c='grey')
        ax.plot([0, 13], [0.5, 0.5], ':', c='grey')
        ax.plot([0, 13], [-0.5, -0.5], ':', c='grey')

        for i_tp in range(n_tp - 1):
            ax.plot([i_tp, i_tp + 1], code[i_tp:(i_tp + 2)], linewidth=2, c='k',
                    alpha=np.clip((autotemp[i_tp] + autotemp[i_tp + 1]- 1) * 1, a_min=0, a_max=1))
        ax.set_ylim([-1, 1])
        ax.set_xlabel('Time');
        ax.set_ylabel('Cross correlation')
        ax.set_xticks(np.arange(n_tp))
        ax.set_xticklabels(time_labels_blank[:-1])

def plot_decoding_matrix(rnn, representation='s1', ax=None):
    """Plot cross temporal decoding accuracy matrix """
    if ax is None:
        ax = plt.subplot(111)

    score_mat, _, __ = bpm.train_single_decoder_new_data(rnn=rnn, save_inplace=False,
                                            label=representation)          ## calculate autotemp score

    sns.heatmap(copy.deepcopy(score_mat), cmap='BrBG',rasterized=True, linewidths=0,
                ax=ax, xticklabels=time_labels_blank[:-1], yticklabels=time_labels_blank[:-1],
                cbar='BrBG', vmin=0, vmax=1)
    ax.set_yticklabels(rotation=90, labels=ax.get_yticklabels())
    ax.set_ylabel('Time')
    ax.set_xlabel('Time');
    ax.invert_yaxis()
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom - 0.5, top + 1.5)

def plot_hist_rot_indices(rnn_folder, representation='s1', ax=None, verbose=0):
    """Plot histogram of rotation indeces of rnns in folder. Index determined as
    average S1-S2 cross correlation"""
    if ax is None:
        ax = plt.subplot(111)

    list_rnns = ru.get_list_rnns(rnn_folder=rnn_folder)
    rot_ind_arr = np.zeros(len(list_rnns))
    print(len(list_rnns))
    for i_rnn, rnn_name in tqdm(enumerate(list_rnns)):
        rnn = ru.load_rnn(os.path.join(rnn_folder, rnn_name))
        ru.ensure_corr_mat_exists(rnn=rnn, representation=representation)
        corr_mat = rnn.rep_corr_mat_dict[representation]
        corr_s1s2_block = corr_mat[np.array([2, 3]), :][:, np.array([6, 7])]  # extract s1-s2 cross correlation.
        assert corr_s1s2_block.shape == (2, 2)
        rot_ind_arr[i_rnn] = np.mean(corr_s1s2_block)
        if rnn.info_dict['task'] == 'pred_dmc' and verbose > 0:
            print(rot_ind_arr[i_rnn], np.mean(rnn.test_loss_split['dmc'][-10:]))
    n, bins, hist_patches = ax.hist(rot_ind_arr, bins=np.linspace(-1, 1, 11),
                                    linewidth=1, color='k', rwidth=0.9, alpha=0.9)
    ## Colour hist bars: https://stackoverflow.com/questions/23061657/plot-histogram-with-colors-taken-from-colormap
    cm = plt.cm.get_cmap('BrBG')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - np.min(bin_centers)      # scale values to interval [0,1]
    col /= np.max(col)
    for c, p in zip(col, hist_patches):
        plt.setp(p, 'facecolor', cm(c))

    ax.set_xlabel('S1 - S2 cross-correlation')
    ax.set_ylabel('Frequency')
    despine(ax)
    return rot_ind_arr

def plot_autotemp_s1_different_epochs(rnn_folder='/home/tplas/repos/eavesdropping/models/save_state/7525/dmc_task/onehot/sparsity_1e-03/pred_dmc/',
                                      # rnn_name='rnn-mnm_2021-05-13-2134.data',
                                      add_labels=False,
                                      epoch_list=[1, 2, 4, 6, 8, 10, 12, 15, 18, 20, 21, 25, 40],
                                      ax=None, plot_legend=True, autotemp_dec_mat_dict=None):
    """Plot autotemp corr of S1 represention for different epochs. Averaged over
    rnns in the rnn_folder. This is saved in autotemp_dec_mat_dict, which is returned.
    It can also be passed as arg, bypassing the calculation (and saving a lot of time).
    """
    if ax is None:
        ax = plt.subplot(111)
    n_tp = 13

    rnn_list = ru.get_list_rnns(rnn_folder=rnn_folder)
    n_rnns = len(rnn_list)
    if autotemp_dec_mat_dict is None:
        autotemp_dec_mat_dict = {x: np.zeros((n_rnns, n_tp)) for x in epoch_list}
        for i_rnn, rnn_name in enumerate(rnn_list):
            print(f'RNN {i_rnn + 1}/{len(rnn_list)}')
            rnn = ru.load_rnn(os.path.join(rnn_folder, rnn_name))
            tmp_autotemp_dec_dict = ru.calculate_autotemp_different_epochs(rnn=rnn, epoch_list=epoch_list,
                                                                      autotemp_dec_dict=None)
            for k, v in tmp_autotemp_dec_dict.items():
                autotemp_dec_mat_dict[k][i_rnn, :] = tmp_autotemp_dec_dict[k]

    alpha_list = [0.86 ** ii for ii in range(len(epoch_list))]
    for i_epoch, epoch in tqdm(enumerate(epoch_list)):
        ax.plot(autotemp_dec_mat_dict[epoch].mean(0), linewidth=3, color=pred_spec_colour, # color=('red' if epoch == 21 else pred_spec_colour), #'#000087',
                alpha=alpha_list[::-1][i_epoch], label=f'epoch {epoch}')

    ax.set_xticks(np.arange(n_tp))
    ax.set_xticklabels(time_labels_blank[:-1])
    ax.set_xlabel('Time')
    ax.set_ylim([0.45, 1.05])
    ax.set_ylabel('S1 decoding\naccuracy ')
    despine(ax)
    if add_labels:
        ax.text(s='1', x=5, y=0.54, c='k', fontdict={'weight': 'bold'})
        ax.text(s='8', x=5.95, y=0.75, c='k', fontdict={'weight': 'bold'})
        ax.text(s='18', x=7.5, y=0.76, c='k', fontdict={'weight': 'bold'})
        ax.text(s='21', x=7.6, y=0.84, c='k', fontdict={'weight': 'bold'})
    if plot_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False)
    # ax.set_title('S1 memory over epochs')

    return autotemp_dec_mat_dict

def plot_raster_trial_average(plot_diff, ax=None, reverse_order=False,
                              c_bar=True, ol=None, th=None, plot_title=True,
                              representation='s1'):
    """Plot raster plot [of trial av activity] of plot_diff. It is sorted unless
    a sorted array ol is passed as arg.
    """
    if representation == 'go':
        plot_cmap = 'RdGy'
    elif representation == 's1' or representation == 's2':
        plot_cmap = 'PiYG'
    if ax is None:
        ax = plt.subplot(111)
    assert plot_diff.shape == (20, 13)
    if ol is None:
        ol = opt_leaf(plot_diff, dim=0)  # optimal leaf sorting
        if reverse_order:
            ol = ol[::-1]
    # rev_ol = np.zeros_like(ol) # make reverse mapping of OL
    # for i_ol, el_ol in enumerate(ol):
    #     rev_ol[el_ol] = i_ol
    plot_diff = plot_diff[ol, :]
    if th is None:
        th = np.max(np.abs(plot_diff)) # threshold for visualisation

    sns.heatmap(plot_diff, cmap=plot_cmap, vmin=-1 * th, vmax=th, ax=ax,
                    rasterized=True, linewidths=0,
                    xticklabels=time_labels_blank[:-1], cbar=c_bar)
    ax.set_yticklabels(rotation=0, labels=ax.get_yticklabels())
    ax.set_ylabel('neuron #')
    ax.invert_yaxis()
    ax.set_xticklabels(rotation=0, labels=ax.get_xticklabels())
    bottom, top = ax.get_ylim()
    ax.set_ylim([0, plot_diff.shape[0]])
    if plot_title:
        ax.set_title(f'Activity difference dependent on {representation}', loc='left', weight='bold')
    ax.set_xlabel('Time');

    return ol

def plot_weights(rnn_layer, ax=None, title='weights', xlabel='',
                 ylabel='', xticklabels=None, yticklabels=None,
                 weight_order=None, th=None):
    '''Plot a weight matrix; given a RNN layer, with zero-symmetric clipping.'''
    if ax is None:
        ax = plt.subplot(111)
    weights = [x for x in rnn_layer.parameters()][0].detach().numpy()
    if weight_order is not None and weights.shape[0] == len(weight_order):
        weights = weights[weight_order, :]
    elif weight_order is not None and weights.shape[1] == len(weight_order):
        weights = weights[:, weight_order]
    else:
        if weight_order is not None:
            print(f'weight order not implemented because the size is different. weights: {weights.shape}, order: {weight_order.shape}')
        else:
            print('weight sorting not defined')
    if th is None:
        th = np.percentile(np.abs(weights), 95)
    sns.heatmap(weights, ax=ax, cmap='PuOr', vmax=th, vmin=-1 * th)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title(title, weight='bold');
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    return ax

def plot_all_UWVT(rnn_model, freq_labels='', weight_order=None, ax_w=None, th=None):
    '''Plot the 3 weight matrices  U, W and V.'''
    if ax_w is None:
        fig, ax_w = plt.subplots(1, 3, figsize=(14, 3))
    else:
        if type(ax_w) is dict:
            assert ax_w.keys() == [0, 1, 2]
        elif type(ax_w) is np.ndarray:
            assert len(ax_w) == 3
        fig = None

    plot_weights(ax=ax_w[0], rnn_layer=rnn_model.lin_input,
                title='U - Input-neuron weights', th=th,
                xticklabels=input_vector_labels, ylabel='Neuron',
                weight_order=weight_order, xlabel='Input')


    plot_weights(ax=ax_w[1], rnn_layer=rnn_model.lin_feedback,
                 title='W - Feedback neuron-neuron weights',
                 ylabel='Neuron', xlabel='Neuron', th=th,
                 weight_order=weight_order)

    plot_weights(ax=ax_w[2], rnn_layer=rnn_model.lin_output,
                 title='V & T - Neuron-output weights', th=th,
                 yticklabels=output_vector_labels, xlabel='Neuron',
                 ylabel='Output', weight_order=weight_order)


    return (fig, ax_w)

def plot_example_codes(one_ax=True, specify_irnn_list=None, sorting_rnns=None):
    """Plot neural memory code of rnns in rnn_folder. Either plot the 1D code (mean S1 cross corr)
    of all rnns on 1 ax, or plot 1D and 2D cross corr for each rnn on a new row."""
    rnn_folder = '/home/tplas/repos/eavesdropping/models/7525/dmc_task/onehot/sparsity_1e-03/pred_dmc/'
    rnn_list = ru.get_list_rnns(rnn_folder=rnn_folder)

    # fig, ax = plt.subplots(5, 5, figsize=(25, 20))
    if one_ax:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    else:
        fig, ax = plt.subplots(20, 2, figsize=(6, 60), gridspec_kw={'hspace': 0.7})
    i_row, i_col = 0, 0
    # epoch_list = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 19, 20, 21, 22, 23, 25, 30, 40, 50]

    rot_ind_arr = np.zeros(len(rnn_list))
    if sorting_rnns is not None:
        assert len(sorting_rnns) == len(rnn_list)
        rnn_list = [rnn_list[x] for x in sorting_rnns]
    for i_rnn, rnn_name in tqdm(enumerate(rnn_list)):
        if specify_irnn_list is not None:
            if i_rnn not in specify_irnn_list:
                continue
        rnn = ru.load_rnn(os.path.join(rnn_folder, rnn_name))
        rnn.eval()

        if one_ax:
            plot_correlation_matrix(rnn=rnn, representation='s1', ax=ax,
                                        hard_reset=True, plot_mat=False, alpha=1)
        else:
            plot_correlation_matrix(rnn=rnn, representation='s1', ax=ax[i_rnn, 0],
                                        hard_reset=True, plot_mat=False, alpha=1)
            ax[i_rnn, 0].set_xticks(np.arange(13))
            ax[i_rnn, 0].set_xticklabels(time_labels_blank[:-1])
            ax[i_rnn, 0].set_title(f'{i_rnn}, {rnn_name}')
            plot_correlation_matrix(rnn=rnn, representation='s1', ax=ax[i_rnn, 1])

        corr_mat = rnn.rep_corr_mat_dict['s1']
        corr_s1s2_block = corr_mat[np.array([2, 3]), :][:, np.array([6, 7])]
        assert corr_s1s2_block.shape == (2, 2)
        rot_ind_arr[i_rnn] = np.mean(corr_s1s2_block)
    print(np.argsort(rot_ind_arr))
    if one_ax:
        ax.set_xticks(np.arange(13))
        ax.set_xticklabels(time_labels_blank[:-1])
