import os
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.metrics
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn') # pretty matplotlib plots

import seaborn as sns
sns.set('notebook', style='whitegrid', font_scale=1.25)

def calc_mean_binary_cross_entropy_from_probas(ytrue_N, yproba_N):
    return sklearn.metrics.log_loss(ytrue_N, yproba_N, labels=[0, 1]) / np.log(2.0)

def porblem1B():
    DATA_DIR = os.path.join('.', 'data_digits_8_vs_9_noisy')

    x_tr_A = np.loadtxt(os.path.join(DATA_DIR, 'x_train.csv'), delimiter=',', skiprows=1)
    x_va_B = np.loadtxt(os.path.join(DATA_DIR, 'x_valid.csv'), delimiter=',', skiprows=1)
    x_te_B = np.loadtxt(os.path.join(DATA_DIR, 'x_test.csv'), delimiter=',', skiprows=1)

    y_tr_A = np.loadtxt(os.path.join(DATA_DIR, 'y_train.csv'), delimiter=',', skiprows=1)
    y_va_B = np.loadtxt(os.path.join(DATA_DIR, 'y_valid.csv'), delimiter=',', skiprows=1)

    tr_err_list = list()
    va_err_list = list()
    tr_bce_list = list()
    va_bce_list = list()

    C = 1e6
    for i in range(1, 41):
        lr = sklearn.linear_model.LogisticRegression(solver='lbfgs', C=C, max_iter=i).fit(x_tr_A, y_tr_A)
        yproba_tr_N = lr.predict_proba(x_tr_A)[:, 1]
        yproba_va_N = lr.predict_proba(x_va_B)[:, 1]

        tr_bce = calc_mean_binary_cross_entropy_from_probas(y_tr_A, yproba_tr_N)
        va_bce = calc_mean_binary_cross_entropy_from_probas(y_va_B, yproba_va_N)
        tr_bce_list.append(tr_bce)
        va_bce_list.append(va_bce)

        tr_error = sklearn.metrics.zero_one_loss(y_tr_A, yproba_tr_N >= 0.5)
        va_error = sklearn.metrics.zero_one_loss(y_va_B, yproba_va_N >= 0.5)
        tr_err_list.append(tr_error)
        va_err_list.append(va_error)
    return tr_bce_list,va_bce_list,tr_err_list,va_err_list

def porblem1C():
    DATA_DIR = os.path.join('.', 'data_digits_8_vs_9_noisy')

    x_tr_A = np.loadtxt(os.path.join(DATA_DIR, 'x_train.csv'), delimiter=',', skiprows=1)
    x_va_B = np.loadtxt(os.path.join(DATA_DIR, 'x_valid.csv'), delimiter=',', skiprows=1)
    x_te_B = np.loadtxt(os.path.join(DATA_DIR, 'x_test.csv'), delimiter=',', skiprows=1)

    y_tr_A = np.loadtxt(os.path.join(DATA_DIR, 'y_train.csv'), delimiter=',', skiprows=1)
    y_va_B = np.loadtxt(os.path.join(DATA_DIR, 'y_valid.csv'), delimiter=',', skiprows=1)

    tr_err_list = list()
    va_err_list = list()
    tr_bce_list = list()
    va_bce_list = list()

    i = 1000
    C_grid = np.logspace(-9, 6, 31)
    for C in C_grid:
        lr = sklearn.linear_model.LogisticRegression(solver='lbfgs', C=C, max_iter=i).fit(x_tr_A, y_tr_A)
        yproba_tr_N = lr.predict_proba(x_tr_A)[:, 1]
        yproba_va_N = lr.predict_proba(x_va_B)[:, 1]

        tr_bce = calc_mean_binary_cross_entropy_from_probas(y_tr_A, yproba_tr_N)
        va_bce = calc_mean_binary_cross_entropy_from_probas(y_va_B, yproba_va_N)
        tr_bce_list.append(tr_bce)
        va_bce_list.append(va_bce)

        tr_error = sklearn.metrics.zero_one_loss(y_tr_A, yproba_tr_N >= 0.5)
        va_error = sklearn.metrics.zero_one_loss(y_va_B, yproba_va_N >= 0.5)
        tr_err_list.append(tr_error)
        va_err_list.append(va_error)
    return tr_bce_list,va_bce_list,tr_err_list,va_err_list


def plot(tr_bce_list,va_bce_list,tr_err_list,va_err_list):
    fig, ax_grid = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(16, 5.5))
    fig.tight_layout()
    cur_ax = ax_grid[0]
    plt.sca(cur_ax)
    cur_ax.set_title('binary cross entropy')
    plt.plot(range(1, 41), tr_bce_list, 'b.-', label='train binary cross entropy')
    plt.plot(range(1, 41), va_bce_list, 'r.-', label='valid binary cross entropy')
    plt.ylabel('0-1 error rate')
    plt.xlabel("binary cross entropy")
    plt.legend(bbox_to_anchor=(0.7, 1))
    plt.ylim([0, 1])

    cur_ax = ax_grid[1]
    plt.sca(cur_ax)
    cur_ax.set_title('error rate')
    plt.plot(range(1, 41), tr_err_list, 'b.-', label='train err')
    plt.plot(range(1, 41), va_err_list, 'r.-', label='valid err')
    plt.ylabel('0-1 error rate')
    plt.xlabel("iteration")
    plt.legend(bbox_to_anchor=(0.7, 1))
    plt.ylim([0, 1])
    plt.show()


if __name__ == '__main__':
    tr_bce_list,va_bce_list,tr_err_list,va_err_list = porblem1B()
    plot(tr_bce_list,va_bce_list,tr_err_list,va_err_list)
