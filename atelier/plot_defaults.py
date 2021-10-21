
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from cycler import cycler

matplotlib.font_manager._rebuild()

import tol_colors

# matplotlib.rc('font', weight='bold')
# matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# matplotlib.rc('font',**{'family':'serif','serif':['Times']})
# matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('font',**{'family': 'sans-serif', 'sans-serif': ['Source Sans '
                                                               'Pro']})
# matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Roboto']})


matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True

def test_plot():
    x = np.arange(0, 200, 0.1)
    y1 = 0.25*(x-100)**2
    y2 = 0.4*(x-100)**2
    y3 = 0.7*(x-100)**2

    cset = tol_colors.tol_cset(colorset='high-contrast')

    cmap = tol_colors.tol_cmap(colormap='rainbow_discrete')

    fig = plt.figure(num=None, figsize=(5.5, 6), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.16, bottom=0.1, right=0.81, top=0.98)

    ax.plot(x, y1, label='curve 1', color=cset[0], lw=2)
    ax.plot(x, y2, label='curve 1', color=cset[1], lw=2)
    ax.plot(x, y3, label='curve 1', color=cset[2], lw=2)
    #
    # ax.plot(x+30, y1, label='curve 1', color=cmap(0.2), lw=2)
    # ax.plot(x+30, y2, label='curve 1', color=cmap(0.5), lw=2)
    # ax.plot(x+30, y3, label='curve 1', color=cmap(0.8), lw=2)

    ax.set_xlabel(r'Redshift $z$',
                  fontweight='heavy', fontsize=15)
    ax.set_ylabel(r'Redshift $\Phi(M_{1450},z)$', fontsize=15,
                  fontweight='bold')

    plt.legend()

    plt.show()



test_plot()



