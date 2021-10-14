#!/usr/bin/env python

import numpy as np
import pandas as pd

from matplotlib import rc
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline
from scipy.stats import binned_statistic_2d


# Tex font
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


class ClippedFunction(object):
    """A helper class to clip output between two values.

    """
    def __init__(self,fun,minval,maxval):
        self.fun = fun
        self.minval = minval
        self.maxval = maxval
    def clip(self,arr):
        return arr.clip(self.minval,self.maxval)
    def __call__(self,*args,**kwargs):
        return self.clip(self.fun(*args,**kwargs))
    def ev(self,*args,**kwargs):
        return self.clip(self.fun.ev(*args,**kwargs))


class QsoSelectionFunction(object):
    """Quasar selection function class.

    This class provides the basic capabilities for quasar selection functions.

    Attributes
    ----------
    mag_range : (tuple)
        Magnitude range over which the quasar selection function is evaluated.
    redsh_range: (tuple)
        Redshift range over which the quasar selection function is evaluated.

    """

    def __init__(self, mag_range=None, redsh_range=None):
        """Initialize the quasar selection function class.

        :param mag_range: Magnitude range over which the quasar selection
         function is evaluated. (default = None)
        :type mag_range: tuple
        :param redsh_range: Redshift range over which the quasar selection
         function is evaluated. (default = None)
        :type redsh_range: tuple
        """
        self.mag_range = mag_range
        self.redsh_range = redsh_range

    def evaluate(self, mag, redsh):
        """Evaluate selection function at magnitude and redshift.

        :param mag: Magnitude at which the quasar selection function is
         evaluated.
        :type mag: float or np.ndarray
        :param redsh: Redshift at which the quasar selection function is
          evaluated.
        :type redsh: float or np.ndarray
        :return: Value of selection function at given redshift and magnitude.
        :rtype: float or np.ndarray
        """
        raise NotImplementedError

    def __call__(self, mag, redsh):
        """Evaluate selection function at magnitude and redshift.

        :type mag: float or np.ndarray
        :param redsh: Redshift at which the quasar selection function is
          evaluated.
        :type redsh: float or np.ndarray
        :return: Value of selection function at given redshift and magnitude.
        :return: Value of selection function at given redshift and magnitude.
        :rtype: float or np.ndarray
        """
        return self.evaluate(mag, redsh)

    def plot_selfun(self, mag_res=0.01, redsh_res=0.01,
                    mag_range=None, redsh_range=None, title=None):
        """Plot the selection function on a grid of redshift and magnitude.

        To calculate the map of the selection function the selection function is
        evaluated using the .evaluate method.

        If no magnitude and redshift range is provided the class attributes
        are used to determine the ranges.

        :param mag_res: Magnitude resolution (default = 0.01)
        :type mag_res: float
        :param redsh_res: Redshift resolution (default = 0.01)
        :type redsh_res: float
        :param mag_range: Magnitude range over which the quasar selection
         function is evaluated. (default = None)
        :type mag_range: tuple
        :param redsh_range: Redshift range over which the quasar selection
         function is evaluated. (default = None)
        :type redsh_range: tuple
        :param title: Title of the plot (default = None)
        :type: string
        :return:
        """
        # Set up figure
        fig = plt.figure(num=None, figsize=(5.5, 6), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.16, bottom=0.1, right=0.81, top=0.98)

        # Set up variables
        if mag_range is None and self.mag_range is not None:
            mag_range = self.mag_range
        elif mag_range is None and self.mag_range is None:
            raise ValueError('[ERROR] Magnitude range is not defined. Please '
                             'set the mag_range keyword argument.')
        if redsh_range is None and self.mag_range is not None:
            redsh_range = self.redsh_range
        elif redsh_range is None and self.mag_range is None:
            raise ValueError('[ERROR] Redshift range ist not defined. Please '
                             'set the redsh_range keyword argument.')

        mag = np.arange(mag_range[0], mag_range[1], mag_res)
        # mag = mag[::-1]
        redsh = np.arange(redsh_range[0], redsh_range[1], redsh_res)

        magmesh, redshmesh = np.meshgrid(mag, redsh)

        # Evaluate selection function
        selfun_arr = self.evaluate(magmesh.ravel(), redshmesh.ravel())
        selfun_arr = np.reshape(selfun_arr, (redsh.shape[0], mag.shape[0])).T

        cax = ax.imshow(selfun_arr, vmin=0, vmax=1,
                        extent=[redsh_range[0],
                                redsh_range[1],
                                mag_range[0],
                                mag_range[1]],
                        aspect='auto',
                        cmap=cm.viridis,
                        origin='lower')

        levels = [0.2, 0.5, 0.70]
        CS = ax.contour(redsh, mag, selfun_arr, levels,
                        colors='k')

        if title is not None:
            cbar_ax = fig.add_axes([0.83, 0.095, 0.04, 0.83])
        else:
            cbar_ax = fig.add_axes([0.83, 0.095, 0.04, 0.89])
        cbar_ax.tick_params(labelsize=15)
        fig.colorbar(cax, cax=cbar_ax,
                     ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                            1.0], ).set_label(label=r'$\rm{Completeness}$',
                                              size=20)

        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.clabel(CS, fontsize=9, inline=1)
        ax.set_ylabel(r'$M_{1450}\,[\rm{mag}]$', fontsize=20)
        ax.set_xlabel(r'$\rm{Redshift}\ z$', fontsize=20)

        if title is not None:
            fig.suptitle(r'$\textrm{'+title+'}$', fontsize=18)
            fig.subplots_adjust(top=0.93)

        plt.show()


class CompositeQsoSelectionFunction(QsoSelectionFunction):
    """Composite quasar selection function class

    This class provides the capabilities for composite quasar selection
    functions and updates the base class evaluate and call methods accordingly.

    Attributes
    ----------
    selfun_list : list(atelier.selfun.QsoSelectionFunction)
        List of quasar selection functions
    mag_range : (tuple)
        Magnitude range over which the quasar selection function is evaluated.
    redsh_range: (tuple)
        Redshift range over which the quasar selection function is evaluated.

    """

    def __init__(self, selfun_list, mag_range=None, redsh_range=None):
        """

        :param selfun_list: List of quasar selection functions
        :type selfun_list: list
        :param mag_range: Magnitude range over which the quasar selection
         function is evaluated. (default = None)
        :type mag_range: tuple
        :param redsh_range: Redshift range over which the quasar selection
         function is evaluated. (default = None)
        :type redsh_range: tuple
        """

        super(CompositeQsoSelectionFunction, self).__init__(mag_range=mag_range,
                                                        redsh_range=redsh_range)

        self.selfun_list = selfun_list

    def evaluate(self, mag, redsh):
        """Evaluate the composite selection function at magnitude and redshift.

        :param mag: Magnitude at which the quasar selection function is
         evaluated.
        :type mag: float or np.ndarray
        :param redsh: Redshift at which the quasar selection function is
          evaluated.
        :type redsh: float or np.ndarray
        :return: Value of selection function at given redshift and magnitude.
        :rtype: float or np.ndarray
        """

        selfun_values = [selfun.evaluate(mag, redsh) for selfun in
                         self.selfun_list]

        return np.prod(np.array(selfun_values, dtype='float32'), axis=0)

    def __call__(self, mag, redsh):
        """Evaluate the composite selection function at magnitude and redshift.

        :type mag: float or np.ndarray
        :param redsh: Redshift at which the quasar selection function is
          evaluated.
        :type redsh: float or np.ndarray
        :return: Value of selection function at given redshift and magnitude.
        :return: Value of selection function at given redshift and magnitude.
        :rtype: float or np.ndarray
        """

        return self.evaluate(mag, redsh)


class QsoSelectionFunctionConst(QsoSelectionFunction):
    """Composite quasar selection function class

    This class provides the capabilities for the constant quasar selection
    function and updates the base class evaluate and call methods accordingly.

    Attributes
    ----------
    value : float
        Value of the constant quasar selection function

    """

    def __init__(self, value):
        """Initialize the constant quasar selection function

        :param value:
        """
        super(QsoSelectionFunctionConst, self).__init__()

        self.value = value

    def evaluate(self, mag, redsh):
        """Evaluate the constant selection function at magnitude and redshift.

        :type mag: float or np.ndarray
        :param redsh: Redshift at which the quasar selection function is
          evaluated.
        :type redsh: float or np.ndarray
        :return: Value of selection function at given redshift and magnitude.
        :return: Value of selection function at given redshift and magnitude.
        :rtype: float or np.ndarray
        """

        if (isinstance(mag, float) or isinstance(mag, int)) and (isinstance(
                redsh, float) or isinstance(redsh, int)):

            return self.value

        elif isinstance(mag,np.ndarray) and isinstance(redsh, np.ndarray) and \
            mag.shape[0] == redsh.shape[0]:

            return np.array([self.value]*mag.shape[0], dtype='float32')

        else:
            raise ValueError('[ERROR] Input values are neither floats or '
                             'arrays of same length')

    def __call__(self, mag, redsh):
        """Evaluate the constant selection function at magnitude and redshift.

        :type mag: float or np.ndarray
        :param redsh: Redshift at which the quasar selection function is
          evaluated.
        :type redsh: float or np.ndarray
        :return: Value of selection function at given redshift and magnitude.
        :return: Value of selection function at given redshift and magnitude.
        :rtype: float or np.ndarray
        """

        return self.evaluate(mag, redsh)


class QsoSelectionFunctionGrid(QsoSelectionFunction):
    """Gridded quasar selection function class.

    This class provides the capabilities for a quasar selection function
    calculated on a magnitude redshift grid of sources defined by a query of
    the source properties. It updates the base class evaluate and call methods
    accordingly.

    Attributes
    ----------
    mag_range : (tuple)
        Magnitude range over which the quasar selection function is evaluated.
    redsh_range: (tuple)
        Redshift range over which the quasar selection function is evaluated.
    mag_bins : (int)
        Number of bins in magnitude
    redsh_bins : (int)
        Number of bins in redshift
    selfungrid : (np.ndarray)
        Grid with selection function values
    clip : (bool)
        Boolean to indicate whether to clip the output value of the selection
        function between 0 and 1.

    """

    def __init__(self, mag_range, redsh_range, mag_bins,
                 redsh_bins, selfungrid=None, clip=True):
        """Initialize the gridded quasar selection function.

        :param mag_range: Magnitude range over which the quasar selection
         function is evaluated.
        :type mag_range: tuple
        :param redsh_range: Redshift range over which the quasar selection
         function is evaluated.
        :type redsh_range: tuple
        :param mag_bins: Number of bins in magnitude
        :type mag_bins: int
        :param redsh_bins: Number of bins in redshift
        :type redsh_bins: int
        :param selfungrid: Grid with selection function values
        :type selfungrid: np.ndarray
        :param clip: Boolean to indicate whether to clip the output value of
         the selection function between 0 and 1.
        :type clip: bool
        """

        super(QsoSelectionFunctionGrid, self).__init__(mag_range=mag_range,
                                                       redsh_range=redsh_range)

        self.splineKwargs = dict(kx=3, ky=3, s=0)

        self.n_per_bin = None
        self.selection = None

        self.mag_bins = mag_bins
        self.redsh_bins = redsh_bins
        self.redsh_range = redsh_range
        self.clip = clip

        self.redsh_edges = np.linspace(redsh_range[0],
                                          redsh_range[1],
                                          redsh_bins+1, dtype='float32')
        self.mag_edges = np.linspace(mag_range[0],
                                        mag_range[1],
                                        mag_bins+1, dtype='float32')

        self.redsh_mid = self.redsh_edges[:-1]+np.diff(self.redsh_edges)/2.
        self.mag_mid = self.mag_edges[:-1] + np.diff(self.mag_edges)/2.

        if selfungrid:
            self.selfungrid = selfungrid
            self.get_selfun_from_grid()

    def get_selfun_from_grid(self):
        """Calculate an interpolation (RectBivariateSpline) of the selection
        function grid to allow evaluations of the selection function at any
        point within the magnitude and redshift range.

        This methods need to be called before the selection function can be
        evaluated.

        :return: None
        """
        f = RectBivariateSpline(self.mag_mid, self.redsh_mid, self.selfungrid,
                                **self.splineKwargs)
        if self.clip:
            f = ClippedFunction(f, 0, 1)
        self.selfun = f

    def calc_selfungrid_from_df(self, df, query, mag_col_name,
                                redshift_col_name,
                          n_per_bin=None, verbose=1):
        """Calculate the selection function grid from a pandas DataFrame

        :param df: Data frame to calculate the selection function from
        :type df: pandas.DataFrame
        :param query: A query describing the subsample that is compared to
         the full data set in order to calculate the selection function.
        :type query: string
        :param mag_col_name: Name of the magnitude column in the data frame.
        :type mag_col_name: string
        :param redshift_col_name:  Name of the redshift column in the data
         frame.
        :type redshift_col_name: string
        :param n_per_bin: Number of sources per bin (default = None). This
         keyword argument is used to check whether the redshift and magnitude
         ranges and the number of bins per dimension produce a uniform
         distribution of sources per bin.
        :type n_per_bin: int
        :param verbose: Verbosity
        :type verbose: int

        :return: None
        """

        # Get the number of sources per bin if not supplied
        if n_per_bin is None:
            ret = binned_statistic_2d(df[redshift_col_name].values,
                                      df[mag_col_name].values,
                                      None,
                                      'count',
                                      bins=[self.redsh_edges,
                                            self.mag_edges])

            if verbose > 0:
                print('[INFO] Number of sources per bin will be calculated from '
                      'grid.')
            if np.all(ret.statistic.flatten() == ret.statistic.flatten()[0]):
                self.n_per_bin = ret.statistic.flatten()[0]
                if verbose > 0:
                    print('[INFO] Found {} sources per grid bin.'.format(
                        self.n_per_bin))
            else:
                print('[ERROR] Different number of sources found per grid '
                      'cell.')
                print(pd.Series(ret.statistic.flatten()).value_counts())
                raise ValueError('[ERROR] The current grid does not have the'
                                 ' same number of sources in every bin. '
                                 'Calculation will be aborted.')
        else:
            self.n_per_bin = n_per_bin

        # Set up the "selected" column
        df['selected'] = False
        sel_idx = df.query(query).index
        df.loc[sel_idx, 'selected'] = True

        # Reshape into the grid
        selected = df['selected'].values.reshape(int(self.mag_bins),
                                                 int(self.redsh_bins),
                                                 int(self.n_per_bin))
        self.selfungrid = np.sum(selected, axis=-1) / float(self.n_per_bin)

        self.get_selfun_from_grid()

    def save(self):
        pass

    def load(self):
        pass

    def plot_grid(self, title=None):
        """Plot the selection function grid as a map of magnitude and redshift.

        :param title: Title of the plot
        :type title: string
        :return: None
        """

        # Set up figure
        fig = plt.figure(num=None, figsize=(5.5, 6), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.16, bottom=0.1, right=0.81, top=0.98)

        cax = plt.pcolormesh(self.redsh_mid, self.mag_mid, self.selfungrid)

        if title is not None:
            cbar_ax = fig.add_axes([0.83, 0.095, 0.04, 0.83])
        else:
            cbar_ax = fig.add_axes([0.83, 0.095, 0.04, 0.89])
        cbar_ax.tick_params(labelsize=15)
        fig.colorbar(cax, cax=cbar_ax,
                     ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                            1.0], ).set_label(label=r'$\rm{Completeness}$',
                                              size=20)

        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_ylabel(r'$M_{1450}\,[\rm{mag}]$', fontsize=20)
        ax.set_xlabel(r'$\rm{Redshift}\ z$', fontsize=20)

        if title is not None:
            fig.suptitle(r'$\textrm{' + title + '}$', fontsize=18)
            fig.subplots_adjust(top=0.93)

        plt.show()

    def evaluate(self, mag, redsh):
        """Evaluate the interpolation of the selection function grid at
        magnitude and redshift.

        :type mag: float or np.ndarray
        :param redsh: Redshift at which the quasar selection function is
          evaluated.
        :type redsh: float or np.ndarray
        :return: Value of selection function at given redshift and magnitude.
        :return: Value of selection function at given redshift and magnitude.
        :rtype: float or np.ndarray
        """

        return self.selfun.ev(mag, redsh)

    def __call__(self, mag, redsh):
        """Evaluate the interpolation of the selection function grid at
        magnitude and redshift.

        :type mag: float or np.ndarray
        :param redsh: Redshift at which the quasar selection function is
          evaluated.
        :type redsh: float or np.ndarray
        :return: Value of selection function at given redshift and magnitude.
        :return: Value of selection function at given redshift and magnitude.
        :rtype: float or np.ndarray
        """

        return self.selfun(mag, redsh)

