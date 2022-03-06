
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from cycler import cycler

matplotlib.font_manager._rebuild()




def set_paper_defaults():
    # Defining the paper plotting style

    matplotlib.rc('text', usetex=True)

    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amssymb}"]
    matplotlib.rcParams['xtick.major.size'] = 4.5
    matplotlib.rcParams['xtick.minor.size'] = 3
    matplotlib.rcParams['ytick.major.size'] = 4.5
    matplotlib.rcParams['ytick.minor.size'] = 3
    matplotlib.rcParams['axes.linewidth'] = 1.2
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['xtick.minor.visible'] = True
    matplotlib.rcParams['ytick.minor.visible'] = True
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['ytick.right'] = True



def set_presentation_defaults():
    # Defining the particular plotting style
    matplotlib.rc('font', weight=500)
    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Roboto']})
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.it'] = 'Roboto'
    matplotlib.rcParams['mathtext.rm'] = 'Roboto'

    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amssymb}"]
    matplotlib.rcParams['xtick.major.size'] = 4.5
    matplotlib.rcParams['xtick.minor.size'] = 3
    matplotlib.rcParams['ytick.major.size'] = 4.5
    matplotlib.rcParams['ytick.minor.size'] = 3
    matplotlib.rcParams['axes.linewidth'] = 1.2
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['xtick.minor.visible'] = True
    matplotlib.rcParams['ytick.minor.visible'] = True
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['ytick.right'] = True



