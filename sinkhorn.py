import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial.distance import cdist

# Calculate P & W using POT (https://pythonot.github.io/)
import ot


def calc_(a, b, cost: np.ndarray, gamma: float):
    P = ot.sinkhorn(a, b, cost, gamma)
    W = ot.sinkhorn2(a, b, cost, gamma)

    return P, W


def calc(a, b, cost: np.ndarray, gamma: float):
    # The Sinkhorn's algorithm

    K = np.exp(-cost / gamma)  # (n, m), given by the problem
    u = np.random.random(cost.shape[0])  # (n,), to be optimized
    v = np.random.random(cost.shape[1])  # (m,), to be optimized

    for _ in range(10):
        u = a / np.matmul(K, v)  # (n,), a / (K x v)
        v = b / np.matmul(K.T, u)  # (m,), b / (K^T x u)

    P = np.matmul(np.matmul(np.diag(u), K), np.diag(v))  # (n, m), diag(u) x K x diag(v)
    W = np.sum(P * cost)
    return P, W


def show(x, y, P, W):
    print(f'Wasserstein distance: {W:.2f}')

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.set_title('OT matrix with samples')
    # plot2D_samples_mat(x, y, P, c=[.5, .5, 1.])
    ax.plot(x[:, 0], x[:, 1], '+r', label='Source samples')
    ax.plot(y[:, 0], y[:, 1], 'xb', label='Target samples')
    ax.legend(loc=0)

    mx = P.max()
    thr = 1e-8
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            alpha = P[i, j] / mx
            if alpha > thr:
                ax.plot([x[i, 0], y[j, 0]], [x[i, 1], y[j, 1]], alpha=alpha, c=[0.5, 0.5, 1])

    plt.show()


def main():
    # Empirical measures
    n = 5  # number of source points
    m = 10  # number of target points
    gamma = 0.1  # coefficient of the regularization term
    # \mu = \sum a_i * \delta_{x_i}
    mu_x = np.array([0., 0.])
    cov_x = np.array([[1., 0.], [0., 1.]])
    x = np.stack([np.random.multivariate_normal(mu_x, cov_x) for _ in range(n)])  # (n, d)
    a = np.ones(n) / n  # (n,). same mass at each position

    # \nu = \sum b_j * \delta_{y_j}
    mu_y = np.array([4., 4.])
    cov_y = np.array([[1., 0.], [0., 1.]])
    y = np.stack([np.random.multivariate_normal(mu_y, cov_y) for _ in range(m)])  # (m, d)
    b = np.ones(m) / m  # (m,). same mass at each position

    # plt.figure()
    # plt.plot(x[:, 0], x[:, 1], '+r', label='Source samples')  # '+' colored by red
    # plt.plot(y[:, 0], y[:, 1], 'xb', label='Target samples')  # 'x' colored by blue
    # plt.legend(loc=0)
    # plt.title('Source and target distributions.')

    # Transportation costs
    dist: np.ndarray = cdist(x, y, metric='euclidean')  # (n, m)

    P, W = calc(a, b, cost=dist, gamma=gamma)

    show(x, y, P, W)


if __name__ == '__main__':
    main()
