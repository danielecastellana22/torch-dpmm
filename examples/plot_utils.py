import numpy as np
import matplotlib.pyplot as plt


def plot_Gauss2D_contour(
        mu, Sigma,
        color='r',
        radiusLengths=[1.0, 3.0],
        markersize=3.0,
        ax=None,
        **kwargs):
    ''' Plot elliptical contours for provided mean mu, covariance Sigma.
    Uses only the first 2 dimensions.
    Post Condition
    --------------
    Plot created on current axes
    '''
    if ax is None:
        ax = plt.gca()

    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)

    mu = mu[:2]
    Sigma = Sigma[:2, :2]
    D, V = np.linalg.eig(Sigma)
    sqrtSigma = np.dot(V, np.sqrt(np.diag(D)))

    # Prep for plotting elliptical contours
    # by creating grid of (x,y) points along perfect circle
    ts = np.arange(-np.pi, np.pi, 0.03)
    x = np.sin(ts)
    y = np.cos(ts)
    Zcirc = np.vstack([x, y])

    # Warp circle into ellipse defined by Sigma's eigenvectors
    Zellipse = np.dot(sqrtSigma, Zcirc)

    # plot contour lines across several radius lengths
    plotted_lines = []
    for r in radiusLengths:
        Z = r * Zellipse + mu[:, np.newaxis]
        p = ax.plot(Z[0], Z[1], '.',
                     markersize=markersize,
                     markerfacecolor=color,
                     markeredgecolor=color, **kwargs)
        plotted_lines += p

    return plotted_lines