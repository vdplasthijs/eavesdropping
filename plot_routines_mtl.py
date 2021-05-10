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

time_labels = ['0', '0', r'$S_1$', r'$S_1$', '0', '0', r'$S_2$', r'$S_2$', '0', '0', 'G', 'G', '0', '0']
time_labels_blank = ['' if x == '0' else x for x in time_labels]
input_vector_labels = ['0', r'$A_1$', r'$A_2$', r'$B_1$', r'$B_2$', 'G']
output_vector_labels = input_vector_labels + [r'$M_1$', r'$M_2$']


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
                    normalise_start=False,
                    plot_top=True, plot_bottom=True, list_top=None, lw=3, plot_total=True,
                    label_dict_keys = {x: x for x in ['dmc', 'dms', 'pred', 'S2', 'G', 'G1', 'G2',
                                                            '0', '0_postS1', '0_postS2', '0_postG']},
                    linestyle_custom_dict={}, colour_custom_dict={},
                    plot_std=True, plot_indiv=False):
    if normalise_start:
        print('Normalising loss functions')
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
        # if 'dmc' in rnn.test_loss_split.keys():
            # print(rnn.test_loss_split['dmc'][-3:])
        if i_rnn == 0:
            # print(rnn.info_dict['pred_loss_function'])
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
                           plot_legend=True, legend_anchor=(1, 1), task_type='dmc',
                           plot_std=True, plot_indiv=False, plot_pred=True, plot_spec=True):
    if ax is None:
        ax = plt.subplot(111)

    ## prediction only
    if folder_pred is not None and os.path.exists(folder_pred) and plot_pred:
        _ = plot_split_perf(rnn_folder=folder_pred, list_top=['pred'], lw=5,
                            linestyle_custom_dict={'pred': '-'}, colour_custom_dict={'pred': [67 / 255, 0, 0]},
                            plot_std=plot_std, plot_indiv=plot_indiv,
                            ax_top=ax, ax_bottom=None, plot_bottom=False, label_dict_keys={'pred': 'H Pred' + f'    (Pred-only, N={len_data_files(folder_pred)})'})

    ## dmc only
    if folder_dmc is not None and os.path.exists(folder_dmc) and plot_spec:
        _ = plot_split_perf(rnn_folder=folder_dmc, list_top=[task_type], lw=5, plot_total=False,
                            linestyle_custom_dict={task_type: '-'}, colour_custom_dict={task_type: [207 / 255, 143 / 255, 23 / 255]},
                            plot_std=plot_std, plot_indiv=plot_indiv,
                            ax_top=ax, ax_bottom=None, plot_bottom=False, label_dict_keys={task_type: f'H {task_type}' + f'   ({task_type}-only, N={len_data_files(folder_dmc)})'})

    ## dmc+ prediction only
    if folder_dmcpred is not None and os.path.exists(folder_dmcpred):
        list_top = []
        if plot_pred:
            list_top.append('pred')
        if plot_spec:
            list_top.append(task_type)
        colour_comb = [73 / 255, 154 / 255, 215 / 255]
        _ = plot_split_perf(rnn_folder=folder_dmcpred, list_top=list_top, lw=5,
                            linestyle_custom_dict={'pred': ':', task_type: '-'},
                            colour_custom_dict={'pred': colour_comb, task_type: colour_comb},
                            plot_std=plot_std, plot_indiv=plot_indiv,
                            ax_top=ax, ax_bottom=None, plot_bottom=False, label_dict_keys={'pred': f'H Pred' + f'    (Pred & {task_type},  N={len_data_files(folder_dmcpred)})',
                                                                                           task_type: f'H {task_type}' + f'   (Pred & {task_type},  N={len_data_files(folder_dmcpred)})'})

    if plot_legend:
        ax.legend(frameon=False, bbox_to_anchor=legend_anchor)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if plot_pred:
        ax.set_ylim([-0.05, 3.5])
    else:
        ax.set_ylim([-0.05, 1.5])
    return ax

def plot_n_nodes_sweep(parent_folder='/home/thijs/repos/rotation/models/sweep_n_nodes/7525/dmc_task/onehot/sparsity_5e-03/',
                   plot_legend=True, ax=None, plot_std=True, plot_indiv=False):
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

def plot_example_trial(trial, ax=None, yticklabels=output_vector_labels,
                       xticklabels=time_labels_blank[1:], c_bar=True,
                       vmin=None, vmax=None, c_map='magma', print_labels=True):
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

def plot_effect_eavesdropping_learning(task='dmc', ratio_exp_str='7525', nature_stim='onehot',
                                       sparsity_str='5e-03', ax=None, plot_legend=True, verbose=0,
                                       plot_std=True, plot_indiv=False, plot_pred=True, plot_spec=True):
   base_folder = f'models/{ratio_exp_str}/{task}_task/{nature_stim}/sparsity_{sparsity_str}/'
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

def plot_learning_efficiency(task_list_tuple=(['dms', 'dmc'],), plot_difference=False):
    df = ru.calculate_all_learning_eff_indices()
    # return df
    fig, ax = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'wspace': 0.7})
    nature_stim_list = ['periodic', 'onehot']
    i_plot = 0
    if plot_difference:
        tmp_df = df[[x[:4] != 'pred' for x in df['loss_comp']]].groupby(['task', 'nature_stim', 'setting','sparsity']).mean()  # compute mean for each set of conditions [ across simulations]
        multi_rows = [True if x[2] == 'multi' else False for x in tmp_df.index]  # select multitask settings
        tmp_df['learning_eff'][multi_rows] *= -1   # multiple effiency with -1 so the difference can be computed using condition-specific sum
        tmp_df = tmp_df.groupby(['task', 'nature_stim', 'sparsity']).sum()  # effectively comppute difference
        tmp_df.reset_index(inplace=True)  # bring multi indexes back to column values
        for i_nat, nat in enumerate(nature_stim_list):
            sns.lineplot(data=tmp_df[tmp_df['nature_stim'] == nat], x='sparsity', y='learning_eff',
                         style='task', ax=ax[i_plot], color='k', linewidth=3, markers=True)
            ax[i_plot].plot([0, 0.01], [0, 0], c='grey')
            ax[i_plot].set_ylim([-0.5, 0.8])
            ax[i_plot].set_ylabel('Difference in \nlearning efficiency index')
            i_plot += 1
    else:
        for task_list in task_list_tuple:
            spec_task_df = df[[x[:3] in task_list for x in df['loss_comp']]]
            for i_nat, nat in enumerate(nature_stim_list):
                sns.lineplot(data=spec_task_df[spec_task_df['nature_stim'] == nat], x='sparsity', y='learning_eff',
                             hue='setting', style='task', markers=True, ci=95, ax=ax[i_plot], hue_order=['multi', 'single'])
                # ax[i_plot].set_ylim([0, 1.1])
                ax[i_plot].set_ylabel('Learning efficiency index\n(= integral loss function)')
                i_plot += 1
    for i_plot in range(len(ax)):
        # ax[i_plot].set_xscale('log', nonposx='clip')
        ax[i_plot].set_xscale('symlog', linthreshx=2e-6)
        ax[i_plot].legend(bbox_to_anchor=(1.4, 1), loc='upper right')
        ax[i_plot].set_title(nature_stim_list[i_plot], fontdict={'weight': 'bold'})
        ax[i_plot].set_xlabel('Sparsity regularisation')

    return df
