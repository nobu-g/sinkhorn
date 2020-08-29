import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from scipy.spatial.distance import cdist

# Calculate P & W using POT (https://pythonot.github.io/)
import ot

artist = None
idx = None
lines = []


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


class OptimalTransport:
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.gamma = 0.1
        self.d = 2

        # \mu = \sum a_i * \delta_{x_i}
        mu_src = np.array([0., 0.])
        cov_src = np.array([[1., 0.], [0., 1.]])
        self.src = np.stack([np.random.multivariate_normal(mu_src, cov_src) for _ in range(n)])  # (n, d)
        self.a = np.ones(n) / n  # (n,). same mass at each position

        # \nu = \sum b_j * \delta_{y_j}
        mu_tgt = np.array([4., 4.])
        cov_tgt = np.array([[1., 0.], [0., 1.]])
        self.tgt = np.stack([np.random.multivariate_normal(mu_tgt, cov_tgt) for _ in range(m)])  # (m, d)
        self.b = np.ones(m) / m  # (m,). same mass at each position

        self.costs = self.calc_costs()

    def calc(self):
        """The Sinkhorn's algorithm"""

        cost = self.calc_costs()
        K = np.exp(-cost / self.gamma)  # (n, m), given by the problem
        u = np.random.random(self.n)  # (n,), to be optimized
        v = np.random.random(self.m)  # (m,), to be optimized

        for _ in range(10):
            u = self.a / np.matmul(K, v)  # (n,), a / (K x v)
            v = self.b / np.matmul(K.T, u)  # (m,), b / (K^T x u)

        P = np.matmul(np.matmul(np.diag(u), K), np.diag(v))  # (n, m), diag(u) x K x diag(v)
        return P

    def calc_(self):
        cost = self.calc_costs()
        return ot.sinkhorn(self.a, self.b, cost, self.gamma)

    def calc_costs(self):
        """Transportation costs"""
        return cdist(self.src, self.tgt, metric='euclidean')  # (n, m)

    def wasserstein(self):
        return np.sum(self.calc() * self.costs)

    def show(self):
        # print(f'Wasserstein distance: {W:.2f}')

        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
        plt.subplots_adjust(bottom=0.25)
        ax.set_title('OT matrix with samples')
        line_src, = ax.plot(self.src[:, 0], self.src[:, 1], 'o', label='Source samples', picker=True)
        line_tgt, = ax.plot(self.tgt[:, 0], self.tgt[:, 1], 'o', label='Target samples', picker=True)
        ax.legend(loc=0)

        ax.xaxis.set_pickradius(10)
        ax.yaxis.set_pickradius(10)

        global lines
        lines = draw_lines(self.src, self.tgt, self.calc(), ax)

        axcolor = 'lightgoldenrodyellow'
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        slider = Slider(ax_slider, 'log10 (gamma)', valmin=-2, valmax=2, valinit=np.log10(self.gamma), valstep=0.1)

        def update(_):
            global lines
            self.gamma = float(np.power(10.0, slider.val))
            for line in lines:
                line.remove()
            lines = draw_lines(self.src, self.tgt, self.calc(), ax)
            fig.canvas.draw()

        slider.on_changed(update)

        def motion(event):
            global artist, idx
            if artist is None:
                return
            assert isinstance(artist, Line2D)

            if event.xdata is None or event.ydata is None:
                release(None)
                return

            print(f'({event.xdata:.3}, {event.ydata:.3})')

            xdata = artist.get_xdata()  # (n or m,)
            ydata = artist.get_ydata()  # (n or m,)
            xdata[idx] = event.xdata
            ydata[idx] = event.ydata
            if artist is line_src:
                self.src = np.stack([xdata, ydata], axis=1)
            if artist is line_tgt:
                self.tgt = np.stack([xdata, ydata], axis=1)

            artist.set_data(xdata, ydata)
            fig.canvas.draw()

        def onpick(event):
            global artist, idx
            if event.artist is line_src:
                artist = event.artist
                idx = event.ind
                print(f'picked: {idx}')
            if event.artist is line_tgt:
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
            lines = draw_lines(self.src, self.tgt, self.calc(), ax)
            fig.canvas.draw()

        fig.canvas.mpl_connect('motion_notify_event', motion)
        fig.canvas.mpl_connect('pick_event', onpick)
        fig.canvas.mpl_connect('button_release_event', release)

        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '--src', '-s', type=int, default=5,
                        help='number of source points')
    parser.add_argument('--target', '--tgt', '-t', type=int, default=5,
                        help='number of target points')
    args = parser.parse_args()

    opt = OptimalTransport(args.source, args.target)

    opt.show()


if __name__ == '__main__':
    main()
