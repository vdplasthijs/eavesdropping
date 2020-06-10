# @Author: Thijs van der Plas <TL>
# @Date:   2020-05-15
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Filename: plot_routines.py
# @Last modified by:   thijs
# @Last modified time: 2020-05-21



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster, scipy.spatial
import bptt_rnn as bp
import rot_utilities as ru

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

def plot_decoder_crosstemp_perf(score_matrix, ax=None, ticklabels='',
                                save_fig=False, fig_name='figures/example_low_crosstempmat.pdf'):
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
    ax.set_title('Cross-temporal decoding score\nCorrelated single example', weight='bold');
    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    return ax

def plot_raster_trial_average(forw, ax=plt.subplot(111), save_fig=False,
                              fig_name='figures/example_high_forward_difference.pdf'):
    labels_use_1 = np.array([x[0] == '1' for x in forw['labels_train']])
    labels_use_2 = np.array([x[0] == '2' for x in forw['labels_train']])

    plot_diff = (forw['train'][labels_use_1, :, :].mean(0) - forw['train'][labels_use_2, :, :].mean(0))
    ol = opt_leaf(plot_diff, dim=1)  # optimal leaf sorting
    # ol = np.argsort(plot_diff.sum(0))
    # rev_ol = np.zeros_like(ol) # make reverse mapping of OL
    # for i_ol, el_ol in enumerate(ol):
    #     rev_ol[el_ol] = i_ol
    plot_diff = plot_diff[:, ol]
    th = np.max(np.abs(plot_diff)) # threshold for visualisation
    hm = sns.heatmap(plot_diff.T, cmap='PiYG', vmin=-1 * th, vmax=th, ax=ax,
                     xticklabels=double_time_labels_blank[:-1])
    bottom, top = hm.get_ylim()
    hm.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Activity difference between green and pink trials', weight='bold')
    plt.xlabel('Time'); plt.ylabel('neuron id');
    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    return hm, ol

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
    plt.legend(); ax_abs_w.set_xticks(np.arange(agg_weights[kk].shape[1]));
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

def plot_alpha_beta_performance(alpha_perf, beta_perf, ax=plt.subplot(111),
                                time_labels=double_time_labels_blank[:-1],
                                save_fig=False, fig_name='figures/alpha_beta_decoding_75.pdf'):
    '''Plot two lines - alpha_perf & beta_perf'''
    ax.plot(alpha_perf, alpha=0.9, color='k',
                  marker='', linewidth=6, markersize=18, label=r"colour$\mathregular{_{AB}}$")
    ax.plot(beta_perf, alpha=0.9, color='k', linestyle='--',
                  marker='', linewidth=6, markersize=18, label=r'colour$\mathregular{_C}$')
    ax.set_xticks(np.arange(len(time_labels)))
    ax.set_xticklabels(time_labels);
    ax.set_yticks([0.5, 0.75, 1])
    ax.set_xlabel('Time'); ax.set_ylabel('Decoding accuracy');
    ax.legend(bbox_to_anchor=(1,0 , 0, 1), fontsize=20)
    # ax.set_title(r'$\alpha$ decoding performance', weight='bold', fontsize=15);
    # ax.axvspan(xmin=10, xmax=17, alpha=0.15)
    ax.set_xlim([0, 17]); sns.despine()
    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    return ax

def plot_stable_switch_bar_diagram(stable_list, switch_list, ax=plt.subplot(111), bar_width=0.35,
                                   save_fig=False, fig_name='figures/stable_switch_correlated.pdf'):
    '''Plot bar diagram of number of stable & switch neurons '''
    assert len(stable_list) == len(switch_list)
    bar_locs = np.arange(len(stable_list))
    bar_stable = ax.bar(bar_locs - bar_width / 2, stable_list,
                         width=bar_width, label='stable', color='#6699FF')  # plot bar
    bar_switch = ax.bar(bar_locs + bar_width / 2, switch_list,
                         width=bar_width, label='switch', color='#660033')
    ax.set_xticks(bar_locs)
    ax.set_xticklabels(['anti-correlated', 'decorrelated', 'correlated'], rotation=0) # ax_bar.set_xticklabels(inds_sel.keys())
    ax.legend(); ax.set_ylabel('Fraction of neurons');
    ax.set_title('Distribution of stable & switch neurons', weight='bold')
    sns.despine()
    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    return ax

def plot_neuron_diff(ax_select, act_1, act_2, mean_ls='-',
                     time_labels=double_time_labels_blank[:-1]):
    ax_select.axvspan(xmin=1.5, xmax=3.5, color='grey', alpha=0.3)
    ax_select.axvspan(xmin=5.5, xmax=7.5, color='grey', alpha=0.3)
    ax_select.axvspan(xmin=9.5, xmax=11.5, color='grey', alpha=0.3)
    ax_select.axvspan(xmin=13.5, xmax=15.5, color='grey', alpha=0.3)
    mean_act = (act_1 + act_2) / 2
    c_mat_green = ru.create_color_mat(x=act_1, c='green')
    c_mat_mag = ru.create_color_mat(x=act_1, c='m')
    c_mat_k = ru.create_color_mat(x=act_1, c='k')
    for ii in range(len(act_1) - 1):
        ax_select.plot(np.arange(ii, (ii + 2)), mean_act[ii:(ii + 2)],
                       linewidth=3, linestyle=mean_ls, c='k', alpha=0.8)#c=c_mat_k[ii, :])
        ax_select.plot(np.arange(ii, (ii + 2)), act_2[ii:(ii + 2)],
                       linewidth=5, c=c_mat_mag[ii, :]);
        ax_select.plot(np.arange(ii, (ii + 2)), act_1[ii:(ii + 2)],
                       c=c_mat_green[ii, :], linewidth=5);
    ax_select.set_xticks(np.arange(len(time_labels)));
    ax_select.set_xlabel('Time'); ax_select.set_ylabel('Activity')
    ax_select.set_xticklabels(time_labels);
    ax_select.set_ylim([-1, 1.3])
    for i_letter, letter in enumerate(['A', 'B', 'C', 'D']):
        ax_select.text(s=letter, x=2.02 + 4 * i_letter, y=1.01,
                       fontdict={'weight': 'bold', 'fontsize': 18})
    sns.despine()
    return ax_select

def plot_arrow_line(x, y, ax=None, c='blue',
                    sens_times=np.arange(2, 8), mem_times=np.arange(8, 11)):
    if ax is None:
        ax = plt.subplot(111)
    c_mat = ru.create_color_mat(x=x, c=c)
#     ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], width=.02,
#               scale_units='xy', angles='xy', scale=1, color=c_mat)
    y = -1 * y
    x = -1 * x
    x_sens = np.mean(x[sens_times])
    y_sens = np.mean(y[sens_times])
    x_mem = np.mean(x[mem_times])
    y_mem = np.mean(y[mem_times])
    print('sens', x_sens.round(2), y_sens.round(2))
    print('mem', x_mem.round(2), y_mem.round(2))
    for ii in range(len(x) - 1):
        ax.plot(x[ii:(ii + 2)], y[ii:(ii + 2)], c=c_mat[ii, :], linewidth=7)
#     ax.plot(x, y, color=c_mat)
    plt.scatter(x_sens, y_sens, marker='x', s=50, c=c_mat[-1, :][np.newaxis, :])
    plt.scatter(x_mem, y_mem, marker='o', s=50, c=c_mat[-1, :][np.newaxis, :])
    ax.set_axis_off()
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
    ax.text(s='-1', x=0.96, y=-0.1, fontdict={'fontsize': 12})
    ax.text(s='-1', x=-0.11, y=0.96, fontdict={'fontsize': 12})
    ax.text(s='1', x=-1, y=-0.1, fontdict={'fontsize': 12})
    ax.text(s='1', x=-0.08, y=-1.04, fontdict={'fontsize': 12})
    return ax

def plot_two_neuron_state_space(activity_1, activity_2, mean_ls_dict,
                                max_tp=17, ax=plt.subplot(111), save_fig=False,
                                fig_name='figures/example_med_statespace_stable-switch.pdf'):
    n1 = list(activity_1.keys())[0]
    n2 = list(activity_1.keys())[1]
    mean_n1 = (activity_1[n1] + activity_2[n1]) / 2
    mean_n2 = (activity_1[n2] + activity_2[n2]) / 2
    ax.plot([-1, 1], [0, 0], c='k', linewidth=2, linestyle=mean_ls_dict[1])  # x axis - so correpsonding to neuron on y axis
    ax.plot([0, 0], [-1, 1], c='k', linewidth=2, linestyle=mean_ls_dict[0])  # y axis - so correspond to neuron on x axis

    _ = plot_arrow_line(activity_1[n1][:max_tp] - mean_n1[:max_tp],
             activity_1[n2][:max_tp] - mean_n2[:max_tp], c='green', ax=ax)
    _ = plot_arrow_line(activity_2[n1][:max_tp] - mean_n1[:max_tp],
             activity_2[n2][:max_tp] - mean_n2[:max_tp], c='m', ax=ax)
    # ax.set_title('State space', weight='bold')
    ax.text(s='Stable neuron', x=0.65, y=-0.25,
               fontdict={'fontsize': 16})
    ax.text(s='Switch neuron', x=-0.4, y=1.1,
               fontdict={'fontsize': 16})

    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
    return ax
