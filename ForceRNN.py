## FORCE RNN
import numpy as np

def create_sparse_mask(matrix, frac_sparse=0.1):
    n_total = matrix.size
    n_nz = int(np.round(frac_sparse * n_total))
    inds_nonzero_array = np.random.choice(a=n_total, size=n_nz, replace=False)
    tmp = np.zeros(n_total)
    tmp[inds_nonzero_array] = 1
    tmp = tmp.reshape(matrix.shape)
    inds_nonzero = np.where(tmp == 1)
    return inds_nonzero, n_nz

class FRNN:
    def __init__(self, n_gen=20, p_gg=0.1, p_zg=1, p_gz=1,
                 tau=0.01, interval_w_update=10, alpha=1.0,
                 g_gg=1.5, g_gz=1, fix_seed=None):
        if fix_seed is not None:
            np.random.seed(fix_seed)
        self.n_gen = int(n_gen)

        self.p_sparsity = {}
        assert (p_gg >= 0) & (p_gg <= 1)
        self.p_sparsity['gg'] = p_gg
        assert (p_gz >= 0) & (p_gz <= 1)
        self.p_sparsity['gz'] = p_gz
        assert (p_zg >= 0) & (p_zg <= 1)
        self.p_sparsity['zg'] = p_zg

        self.tau = tau
        assert interval_w_update > tau
        self.interval_w_update = interval_w_update
        self.alpha = alpha

        self.g_const = {}
        self.g_const['gg'] = g_gg
        self.g_const['gz'] = g_gz
        self.scale_init = 0.3


        self.initiate_weights()
        print('FRNN initiated')



    def __str__(self):
        """Define name"""
        return self.name

    def __repr__(self):
        """define representation"""
        return f'instance {self.name} of Class FRNN'

    def initiate_weights(self, read_out_init_zero=False):
        """Notation: weights_yx means from x to y"""
        self.weights, self.mask_w, self.n_nz = {}, {}, {}
        self.weights['gg'] = np.zeros((self.n_gen, self.n_gen))  # interaction weights in generator network
        self.weights['gz'] = np.zeros(self.n_gen)  # feedback from z to generator
        self.weights['zg'] = np.zeros(self.n_gen) # read out from generator to z
        self.w_keys = list(self.weights.keys())
        for kk in self.w_keys:
            self.mask_w[kk], self.n_nz[kk] = create_sparse_mask(matrix=self.weights[kk],
                                                                frac_sparse=self.p_sparsity[kk])
        self.weights['gg'][self.mask_w['gg']] = np.random.randn(self.n_nz['gg']) / np.sqrt((self.p_sparsity['gg'] * self.n_gen))
        self.weights['gz'][self.mask_w['gz']] = np.random.uniform(low=-1, high=1, size=self.n_nz['gz'])
        if read_out_init_zero:
            self.weights['zg'][self.mask_w['zg']] = 0
        elif read_out_init_zero is False:
            self.weights['zg'][self.mask_w['zg']] = np.random.randn(self.n_nz['zg']) / np.sqrt((self.p_sparsity['zg'] * self.n_gen))

        self.p_lr = np.eye(self.n_gen) / self.alpha

    def forward(self, f_est, x_init=None,
                time_pre=True, time_post=True):
        assert (f_est.ndim == 1)
        if x_init is not None:
            assert (x_init.ndim == 1) and (len(x_init) == self.n_gen), 'x_init is not of size (n_gen,)'
        else:
            x_init = np.random.randn(self.n_gen) * self.scale_init
        time_f = len(f_est) * self.tau
        if time_pre:
            self.t_start = -1 * time_f
            self.ind_train_start = len(f_est)
            self.ind_train_stop = 2 * len(f_est)
        else:
            self.t_start = 0
            self.ind_train_start = 0
            self.ind_train_stop = len(f_est)
        if time_post:
            self.t_end = 2 * time_f
        else:
            self.t_end = time_f
        self.t_train_start = 0
        self.t_train_end = time_f
        self.time_array = np.arange(start=self.t_start, stop=self.t_end, step=self.tau)
        self.time_array = np.round(self.time_array, 6)  # get rid of float precision error
        n_timepoints = len(self.time_array)
        self.f_full = np.zeros(n_timepoints) + np.nan
        self.f_full[self.ind_train_start:self.ind_train_stop] = f_est
        n_updates = int(np.floor(len(f_est) / self.interval_w_update))
        self.x_forw = np.zeros((self.n_gen, n_timepoints))
        self.r_forw = np.zeros((self.n_gen, n_timepoints))
        self.z_forw = np.zeros(n_timepoints)
        self.f_forw = f_est
        self.x_forw[:, 0] = x_init  # initiate
        self.r_forw[:, 0] = np.tanh(self.x_forw[:, 0])
        self.z_forw[0] = np.random.randn() * self.scale_init
        self.error = {xx: np.zeros(n_updates) for xx in ['minus', 'plus', 'time']}
        self.euler_integration()

    def rnn_diff_eq(self, time=1):
        """time is an index"""
        assert type(time) == int
        delta_x = 0
        delta_x += -1 * self.x_forw[:, time]
        delta_x += self.g_const['gg'] * np.dot(self.weights['gg'], self.r_forw[:, time])  # checked ,multiplies (R, C) x (C) = (R)
        delta_x += self.g_const['gz'] * np.dot(self.weights['gz'], self.z_forw[time])  # checked
        # GF
        # GI
        delta_x = delta_x / self.tau
        return delta_x

    def euler_integration(self):
        """All parameters defined earlier"""
        assert (self.z_forw[1:] == 0).all()  # not yet run
        i_update = 0
        count_between_intervals = 0
        for i_t, t in enumerate(self.time_array):
            if i_t == 0:  # skip first because of manual init
                continue

            delta_x = self.rnn_diff_eq(time=i_t - 1)
            self.x_forw[:, i_t] = self.x_forw[:, i_t - 1] + self.tau * delta_x
            self.r_forw[:, i_t] = np.tanh(self.x_forw[:, i_t])
            self.z_forw[i_t] = np.dot(self.weights['zg'], self.r_forw[:, i_t])

            if t >= self.t_train_start and t < self.t_train_end:  # during train interval
                self.t = t
                count_between_intervals += 1
                if self.interval_w_update == count_between_intervals:  # time for w update
                    relative_zero = np.where(np.isclose(self.time_array, 0, atol=1e-8))[0][0]  # use isclose() because float error
                    f_time = i_t - relative_zero

                    self.error['time'][i_update] = t
                    self.error['minus'][i_update] = (self.z_forw[i_t] - self.f_forw[f_time]).copy()

                    tmp_div = 1.0 / (1.0 + self.r_forw[:, i_t].T * self.p_lr * self.r_forw[:, i_t])
                    self.p_lr = self.p_lr - (self.p_lr * self.r_forw[:, i_t] * self.r_forw[:, i_t].T * self.p_lr * tmp_div)
                    # self.p_lr = np.linalg.inv(np.matmul(self.r_forw[:, :i_t],
                    #                                  self.r_forw[:, :i_t].T) + self.alpha * np.eye(self.n_gen))
                    self.weights['zg'] = self.weights['zg'] - self.error['minus'][i_update] * np.dot(self.p_lr, self.r_forw[:, i_t])

                    self.error['plus'][i_update] =  np.dot(self.weights['zg'], self.r_forw[:, i_t]) - self.f_forw[f_time]
                    count_between_intervals = 0
                    i_update += 1
