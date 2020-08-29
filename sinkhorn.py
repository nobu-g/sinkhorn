import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist

# Calculate P & W using POT (https://pythonot.github.io/)
import ot


def calc(a, b, cost: np.ndarray, gamma: float):
    # The Sinkhorn's algorithm

    K = np.exp(-cost / gamma)  # (n, m), given by the problem
    u = np.random.random(cost.shape[0])  # (n,), to be optimized
    v = np.random.random(cost.shape[1])  # (m,), to be optimized

    for _ in range(20):
        u = a / np.matmul(K, v)  # (n,), a / (K x v)
        v = b / np.matmul(K.T, u)  # (m,), b / (K^T x u)

    P = np.matmul(np.matmul(np.diag(u), K), np.diag(v))  # (n, m), diag(u) x K x diag(v)
    return P


def calc_(a, b, cost: np.ndarray, gamma: float):
    return ot.sinkhorn(a, b, cost, gamma)


artist = None
idx = None
lines = []


def show(x, y, P, W):
    print(f'Wasserstein distance: {W:.2f}')

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.set_title('OT matrix with samples')
    line1, = ax.plot(x[:, 0], x[:, 1], 'o', label='Source samples', picker=True)
    line2, = ax.plot(y[:, 0], y[:, 1], 'o', label='Target samples', picker=True)
    ax.legend(loc=0)

    ax.xaxis.set_pickradius(10)
    ax.yaxis.set_pickradius(10)

    global lines
    lines = draw_lines(x, y, P, ax)

    def motion(event):
        global artist, idx
        if artist is None:
            return
        assert isinstance(artist, Line2D)

        if event.xdata is None or event.ydata is None:
            artist = None
            return

        xdata = artist.get_xdata()
        ydata = artist.get_ydata()
        xdata[idx] = event.xdata
        ydata[idx] = event.ydata
        print(f'({event.xdata:.3}, {event.ydata:.3})')
        artist.set_data(xdata, ydata)
        fig.canvas.draw()

    def onpick(event):
        global artist, idx
        if event.artist is line1:
            artist = event.artist
            idx = event.ind
            print(f'picked: {idx}')
        if event.artist is line2:
            artist = event.artist
            idx = event.ind
            print(f'picked: {idx}')

    def release(_):
        global artist, idx, lines
        print(f'released: {idx}')
        artist = None
        idx = None

        for line in lines:
            line.remove()
        lines = draw_lines(x, y, P, ax)
        fig.canvas.draw()

    fig.canvas.mpl_connect('motion_notify_event', motion)
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.mpl_connect('button_release_event', release)

    plt.show()


def draw_lines(x, y, P, ax):
    mx = P.max()
    thr = 1e-3
    lines_ = []
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            alpha = P[i, j] / mx
            if alpha < thr:
                continue
            line, = ax.plot([x[i, 0], y[j, 0]], [x[i, 1], y[j, 1]], alpha=alpha, c=[0.5, 0.5, 1])
            lines_.append(line)
    return lines_


def main():
    # Empirical measures
    n = 5  # number of source points
    m = 5  # number of target points
    gamma = 0.01  # coefficient of the regularization term
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

    # Transportation costs
    dist: np.ndarray = cdist(x, y, metric='euclidean')  # (n, m)

    P = calc(a, b, cost=dist, gamma=gamma)
    W = np.sum(P * dist)

    show(x, y, P, W)


if __name__ == '__main__':
    main()
