# @Author: Thijs L van der Plas <thijs>
# @Date:   2021-04-14
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Last modified by:   thijs
# @Last modified time: 2021-04-14

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar as mpl_colorbar
import seaborn as sns
import pickle, os, sys
import scipy.cluster, scipy.spatial
import sklearn.decomposition
import bptt_rnn_mtl as bpm
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


def plot_split_perf(rnn_name=None, rnn_folder=None, ax_top=None, ax_bottom=None,
                    normalise_start=True,
                    plot_top=True, plot_bottom=True, list_top=None, lw=3, plot_total=True,
                    label_dict_keys = {x: x for x in ['dmc', 'dms', 'pred', 'S2', 'G', 'G1', 'G2',
                                                            '0', '0_postS1', '0_postS2', '0_postG']},
                    linestyle_custom_dict={}, colour_custom_dict={}):
    if ax_top is None and plot_top:
        ax_top = plt.subplot(211)
    if ax_bottom is None and plot_bottom:
        ax_bottom = plt.subplot(212)
    if rnn_folder is None:
        list_rnns = [rnn_name]
    else:
        list_rnns = [x for x in os.listdir(rnn_folder) if x[-5:] == '.data']
    # print(label_dict_keys)
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
            # if key != 'pred':
            conv_dict[key][i_rnn, :] = arr.copy()

        if plot_total:
            conv_dict['pred_sep'][i_rnn, :] = np.sum([conv_dict[key][i_rnn, :] for key in ['0', 'S2', 'G']], 0)

    i_plot_total = 0
    dict_keys = list(conv_dict.keys())[::-1]
    colour_dict_keys = {key: color_dict_stand[it] for it, key in enumerate(['S2', 'G', 'L1', 'dmc', '0', 'pred', 'pred_sep'])}
    colour_dict_keys['0'] = color_dict_stand[7]
    for key, val in colour_custom_dict.items():
        colour_dict_keys[key] = val
    linestyle_dict_keys = {x: '-' for x in label_dict_keys.keys()}
    for key, val in linestyle_custom_dict.items():
        linestyle_dict_keys[key] = val

    for key in dict_keys:
        mat = conv_dict[key]
        if normalise_start:
            mat = mat / np.mean(mat[:, 0])#[:, np.newaxis]
        plot_arr = np.mean(mat, 0)
        # if normalise_start:
        #     plot_arr = plot_arr / plot_arr[0]
        if plot_top:
            if (list_top is not None and key in list_top) or (list_top is None and '_' not in key and 'L' not in key):
                # print(key)
                # print(label_dict_keys.keys(), linestyle_dict_keys.keys(), colour_dict_keys.keys())
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
        # ax_top.legend(frameon=False, bbox_to_anchor=(0.5, 0.2)); #ax.set_xlim([0, 10])
    if plot_bottom:
        ax_bottom.legend(frameon=False)
        ax_bottom.set_ylabel('L1 regularisation')
        ax_bottom.set_xlabel('Epoch'); #ax.set_ylabel('error relative')

    return (ax_top, ax_bottom)

def len_data_files(dir_path):
    return len([x for x in os.listdir(dir_path) if x[-5:] == '.data'])

def plot_split_perf_custom(folder_pred=None, folder_dmcpred=None, folder_dmc=None, ax=None,
                           plot_legend=True, legend_anchor=(1, 1), task_type='dmc'):
    if ax is None:
        ax = plt.subplot(111)

    ## prediction only
    if folder_pred is not None:
        _ = plot_split_perf(rnn_folder=folder_pred, list_top=['pred'], lw=5,
                            linestyle_custom_dict={'pred': '-'}, colour_custom_dict={'pred': [67 / 255, 0, 0]},
                            ax_top=ax, ax_bottom=None, plot_bottom=False, label_dict_keys={'pred': 'H Pred' + f'    (Pred-only, N={len_data_files(folder_pred)})'})

    ## dmc only
    if folder_dmc is not None:
        _ = plot_split_perf(rnn_folder=folder_dmc, list_top=[task_type], lw=5, plot_total=False,
                            linestyle_custom_dict={task_type: '-'}, colour_custom_dict={task_type: [207 / 255, 143 / 255, 23 / 255]},
                            ax_top=ax, ax_bottom=None, plot_bottom=False, label_dict_keys={task_type: f'H {task_type}' + f'   ({task_type}-only, N={len_data_files(folder_dmc)})'})

    ## dmc+ prediction only
    if folder_dmcpred is not None:
        colour_comb = [73 / 255, 154 / 255, 215 / 255]
        _ = plot_split_perf(rnn_folder=folder_dmcpred, list_top=['pred', task_type], lw=5,
                            linestyle_custom_dict={'pred': ':', task_type: '-'},
                            colour_custom_dict={'pred': colour_comb, task_type: colour_comb},
                            ax_top=ax, ax_bottom=None, plot_bottom=False, label_dict_keys={'pred': f'H Pred' + f'    (Pred & {task_type},  N={len_data_files(folder_dmcpred)})',
                                                                                           task_type: f'H {task_type}' + f'   (Pred & {task_type},  N={len_data_files(folder_dmcpred)})'})

    if plot_legend:
        ax.legend(frameon=False, bbox_to_anchor=legend_anchor)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([-0.2, 1.6])
    return ax

def plot_effect_eavesdropping_learning(task='dmc', ratio_exp_str='7525', nature_stim='onehot',
                                       sparsity_str='5e-03', ax=None, plot_legend=True):
   base_folder = f'models/{ratio_exp_str}/{task}_task/{nature_stim}/sparsity_{sparsity_str}/'
   folders_dict = {}
   folders_dict['pred_only'] = base_folder + 'pred_only/'
   folders_dict[f'{task}_only'] = base_folder + f'{task}_only/'
   folders_dict[f'pred_{task}'] = base_folder + f'pred_{task}/'
   # print(folders_dict)
   plot_split_perf_custom(folder_pred=folders_dict['pred_only'],
                          folder_dmc=folders_dict[f'{task}_only'],
                          folder_dmcpred=folders_dict[f'pred_{task}'],
                          task_type=task, ax=ax, plot_legend=plot_legend)
   plt.title(task + r'$\; P(\alpha = \beta) = $' + f'0.{ratio_exp_str[:2]},' + r'$ \; \; \lambda=$' + f'{sparsity_str}');

   for key, folder_rnns in folders_dict.items():
       list_keys = key.split('_')
       if 'only' in list_keys:
           list_keys.remove('only')
       learn_eff = ru.compute_learning_index(rnn_folder=folder_rnns,
                                             list_loss=list_keys)
       print(key, {x: (np.round(np.mean(learn_eff[x]), 4), np.round(np.std(learn_eff[x]), 4)) for x in list_keys})
