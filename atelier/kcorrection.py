#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy import stats

from matplotlib import rc
import matplotlib.pyplot as plt

from scipy.stats import binned_statistic_2d
from scipy.interpolate import RectBivariateSpline, griddata, SmoothBivariateSpline


# Tex font
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Adopted from simqso.sqanalysis (Ian McGreer)
# class Interpolator(object):
#     def __init__(self,x,y,z):
#         self.points = np.array([x,y]).T
#         self.values = z
#     def ev(self,x,y):
#         x = np.asarray(x)
#         y = np.asarray(y)
#         xi = np.array([x.ravel(),y.ravel()]).T
#         rv = griddata(self.points,self.values, xi, method='linear')
#         return rv.reshape(x.shape)
#     def __call__(self,x,y):
#         xx,yy = np.meshgrid(x,y,indexing='ij')
#         return self.ev(xx.ravel(),yy.ravel()).reshape(len(x),len(y))

class Interpolator(object):
    def __init__(self, x, y, z):
        self.points = np.array([x, y]).T
        self.values = z
    def ev(self,x,y):
        rv = griddata(self.points, self.values, (x, y), method='linear')
        return rv
    def __call__(self,x,y):
        return self.ev(x, y)



class KCorrection(object):
    """Base class for an astronomical K-correction.

    Attributes
    ----------
    cosmology : astropy.cosmology.Cosmology
        Cosmology

    """

    def __init__(self, cosmology):
        """Initialize the KCorrection base class

        Note that an astronomical  K-correction is specific to the filter
        bands for which it was determined. The current implementation relies
        on the user's understanding of the K-correction in order to apply it
        properly.

        :param cosmology:
        """
        self.cosmology = cosmology

    def evaluate(self, mag, redsh):
        """Evaluate the K-correction for an object of given apparent
        magnitude and redshift.

        This method is not implemented and will be overwritten by children
        methods.

        :param mag: Magnitude
        :type mag: float
        :param redsh: Redshift
        :type redsh: float
        :return: None
        """
        raise NotImplementedError

    def m2M(self, mag, redsh):
        """Calculate the rest-frame absolute magnitude based on the apparent
        (observed) source magnitude and redshift using the K-correction and
        specified cosmology.

        :param mag: Magnitude
        :type mag: float
        :param redsh: Redshift
        :type redsh: float
        :return: Absolute magnitude
        """

        mag = np.array([mag])
        redsh = np.array([redsh])
        # Argument is apparent magnitude, hence inverse=True
        kcorrection = np.array(self.evaluate(mag, redsh, inverse=True))
        distmod = self.cosmology.distmod(redsh).value

        return mag - distmod - kcorrection


    def M2m(self, mag, redsh):
        """Calculate the apparent (observed) magnitude based on the
        rest-frame absolute magnitude and redshift using the K-correction and
        specified cosmology.

        :param mag: Absolute magnitude
        :type mag: float
        :param redsh: Redshift
        :type redsh: float
        :return: Apparent magnitude
        """

        mag = np.array([mag])
        redsh = np.array([redsh])
        # Argument is absolute magnitude, hence inverse=False (default)
        kcorrection = np.array(self.evaluate(mag, redsh))
        distmod = self.cosmology.distmod(redsh).value

        return mag + distmod + kcorrection

class KCorrectionPL(KCorrection):
    """K-correction for a power law spectrum source with a specified spectral
    slope per unit frequency.

    Note that an astronomical  K-correction is specific to the filter
    bands for which it was determined. The current implementation relies
    on the user's understanding of the K-correction in order to apply it
    properly.

    Only for the special case of a power law spectrum source (this case) this is
    independent of the astronomical filter bands.

    Attributes
    ----------
    cosmology : astropy.cosmology.Cosmology
        Cosmology
    slope : float
        Slope of the power law (in frequency)

    """

    def __init__(self, slope, cosmology):
        """Initialize the power law spectrum source K-correction.

        :param slope: Slope of the power law spectrum source
        :type slope: float
        :param cosmology: Cosmology
        :type cosmology: astropy.cosmology.Cosmology
        """

        super(KCorrectionPL, self).__init__(cosmology)

        self.slope = slope

# TODO inverse needs to be correctly implemented here!

    def evaluate(self, mag, redsh, inverse=False):
        """Evaluate the K-correction for an object of given apparent
        magnitude and redshift.

        The power law K-correction does not depend on magnitude. Hence
        the argument 'amg' and the keyword argument 'inverse' play no role.

        :param mag: Magnitude
        :type mag: float
        :param redsh: Redshift
        :type redsh: float
        :return: K-correction
        :rtype: float
        """

        return -2.5 * (1 + self.slope) * np.log10(1+redsh)

    def __call__(self, mag, redsh, inverse=False):
        """Return the K-correction for an object of given apparent
        magnitude and redshift.

        The power law K-correction does not depend on magnitude. Hence
        the argument 'amg' and the keyword argument 'inverse' play no role.

        :param mag: Magnitude
        :type mag: float
        :param redsh: Redshift
        :type redsh: float
        :return: K-correction
        :rtype: float
        """

        return self.evaluate(mag, redsh)

class KCorrectionGrid(KCorrection):
    """K-correction between the observed filter band and the rest-frame
    filter band determined on a grid of data.

    Note that an astronomical  K-correction is specific to the filter
    bands for which it was determined. The current implementation relies
    on the user's understanding of the K-correction in order to apply it
    properly.

    The K-correction is calculated in redshift and rest-frame magnitude bins
    on a grid of data for which redshift, rest-frame magnitudes (either from
    the continuum or through a filter band) and observed frame apparent
    magnitudes through a filter band are provided. The bins are then
    interpolated to provide a smooth K-correction as a function of apparent
    magnitude and redshift.

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
    cosmology : astropy.cosmology.Cosmology
        Cosmology
    slope : float
        Slope of the power law (in frequency)
    kcorrgird : np.ndarray
        Grid of K-correction values

    """


    def __init__(self, mag_range, redsh_range, mag_bins, redsh_bins,
                 cosmology, kcorrgrid=None, min_n_per_bin=11):
        """Initialize the gridded K-correction class.

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
        :param cosmology: Cosmology object
        :type cosmology: astropy.cosmology.Cosmology
        :param kcorrgrid: Grid of K-correction values
        :type kcorrgrid: np.ndarray
        :param min_n_per_bin: Minimum number of objects per bin for the
        inverse k-correction calculation
        :type min_n_per_bin: int
        """

        super(KCorrectionGrid, self).__init__(cosmology)

        self.splineKwargs = dict(kx=3, ky=3, s=0)

        self.min_n_per_bin = min_n_per_bin

        self.mag_bins = mag_bins
        self.redsh_bins = redsh_bins
        self.mag_range = mag_range
        self.redsh_range = redsh_range

        self.redsh_edges = np.linspace(redsh_range[0],
                                       redsh_range[1],
                                       redsh_bins + 1, dtype='float32')
        self.mag_edges = np.linspace(mag_range[0],
                                     mag_range[1],
                                     mag_bins + 1, dtype='float32')

        self.redsh_mid = self.redsh_edges[:-1]+np.diff(self.redsh_edges)/2.
        self.mag_mid = self.mag_edges[:-1] + np.diff(self.mag_edges)/2.
        self.appmag_mid = None

        self.n_per_bin = None

        if kcorrgrid:
            self.kcorrgrid = kcorrgrid
            self.get_kcorrection_from_grid()


    def get_kcorrection_from_grid(self):
        """Interpolate the K-correction grid (absolute magnitude,
        redshift) using a linear interpolation over a rectangular mesh

        :return: None
        """

        # Works but interpolation is not optimal.
        ii = np.where(np.isfinite(self.kcorrgrid))
        mm, zz = np.meshgrid(self.mag_mid, self.redsh_mid, indexing='ij')
        f = Interpolator(mm[ii], zz[ii], self.kcorrgrid[ii])

        self.kcorrection = f

    def get_inv_kcorrection_from_grid(self):
        """Interpolate the inverse K-correction grid (apparent magnitude,
        redshift) using a linear interpolation over a rectangular mesh

        :return: None
        """

        # Works but interpolation is not optimal.
        ii = np.where(np.isfinite(self.inv_kcorrgrid))
        mm, zz = np.meshgrid(self.appmag_mid, self.redsh_mid, indexing='ij')
        f = Interpolator(mm[ii], zz[ii], self.inv_kcorrgrid[ii])

        self.inv_kcorrection = f

    def calc_kcorrection_from_df(self, df, redshift_col_name,
                                 mag_col_name, appmag_col_name,
                                 n_per_bin=None, statistic='median',
                                 inverse=True, verbose=1):
        """Calculate the K-correction grid from a data frame

        :param df: Data frame with redshift, absolute reference (filter band)
         magnitude, and apparent (filter band) magnitude
        :type df: pandas.DataFrame
        :param redshift_col_name:  Name of the redshift column in the data
         frame.
        :type redshift_col_name: string
        :param mag_col_name: Name of the absolute reference (filter band)
         magnitude column in the data frame.
        :type mag_col_name: string
        :param appmag_col_name: Name of the apparent (filter band) magnitude
         column in the data frame
        :type appmag_col_name: string
        :param n_per_bin: Number of sources per bin (default = None). This
         keyword argument is used to check whether the redshift and magnitude
         ranges and the number of bins per dimension produce a uniform
         distribution of sources per bin.
        :type n_per_bin: int
        :param statistic: A string indicating whether to use the 'median' or
         'mean' of the K-correction value in the bin when calculating the grid.
        :type statistic: string (default = 'median')
        :param inverse: Boolean to indicate whether inverse kcorrection will
        be calculated (default=True).
        :type inverse: bool
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
                print(
                    '[INFO] Number of sources per bin will be calculated from '
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

        print('[INFO] Calculating approximate K-correction (absolute '
              'magnitude, redshift).')

        # Calculate the distance modulus for all data frame entries
        distmod = self.cosmology.distmod(
            df[redshift_col_name].values)
        # Calculate the k-correction term
        kcorrection = (df[appmag_col_name] - distmod) - \
                                  df[mag_col_name]

        # Reshape into grid
        kcorrarray = kcorrection.values.reshape(int(self.mag_bins),
                                                       int(self.redsh_bins),
                                                       int(self.n_per_bin))

        if statistic == 'median':
            self.kcorrgrid = np.nanmedian(kcorrarray, axis=-1)
        elif statistic == 'mean':
            self.kcorrgrid = np.nanmean(kcorrarray, axis=-1)

        self.kcorrgrid_mad = stats.median_absolute_deviation(kcorrarray,
                                                             axis=-1,
                                                             nan_policy='omit')

        self.kcorrgrid_mad_all = stats.median_absolute_deviation(kcorrarray,
                                                             axis=None,
                                                             nan_policy='omit')

        self.get_kcorrection_from_grid()

        if inverse:
            print('[INFO] Calculating approximate inverse K-correction ('
                  'apparent magnitude, redshift).')

            # Set up the apparent magnitude grid
            appmag_step = np.median(np.diff(self.mag_mid)) / 2
            m = ((np.isfinite(df[appmag_col_name].values)) &
                 (df[appmag_col_name].values > 0))

            appmag_min = np.nanmin(df[appmag_col_name].values[m]) - \
                          appmag_step
            appmag_max = np.nanmax(df[appmag_col_name].values[m]) + appmag_step
            self.appmag_edges = np.arange(appmag_min, appmag_max, appmag_step)


            self.inv_kcorrgrid = binned_statistic_2d(
                df[appmag_col_name].values[m],
                df[redshift_col_name].values[m],
                kcorrection.values[m],
                'median', [self.appmag_edges, self.redsh_edges])[0]

            n = binned_statistic_2d(
                df[appmag_col_name].values[m],
                df[redshift_col_name].values[m],
                kcorrection.values[m],
                'count', [self.appmag_edges, self.redsh_edges])[0]

            self.inv_kcorrgrid[n < self.min_n_per_bin] = np.nan

            self.appmag_mid = self.appmag_edges[:-1] + \
                              np.diff(self.appmag_edges) / 2

            self.get_inv_kcorrection_from_grid()


    def save(self):
        pass

    def load(self):
        pass

    def plot_inv_grid(self, title=None, ylabel=None):
        # Set up figure
        fig = plt.figure(num=None, figsize=(5.5, 6), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.16, bottom=0.1, right=0.81, top=0.98)

        clabel = r'$\rm{K-correction}$'

        cax = plt.pcolormesh(self.redsh_mid, self.appmag_mid, self.inv_kcorrgrid)

        if title is not None:
            cbar_ax = fig.add_axes([0.83, 0.095, 0.04, 0.83])
        else:
            cbar_ax = fig.add_axes([0.83, 0.095, 0.04, 0.89])
        cbar_ax.tick_params(labelsize=15)

        fig.colorbar(cax, cax=cbar_ax).set_label(label=clabel,
                                                 size=20)

        ax.tick_params(axis='both', which='major', labelsize=15)

        if ylabel is None:
            ylabel = r'$\rm{Apparent\ magnitude}$'

        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_xlabel(r'$\rm{Redshift}\ z$', fontsize=20)

        if title is not None:
            fig.suptitle(r'$\textrm{' + title + '}$', fontsize=18)
            fig.subplots_adjust(top=0.93)

        plt.show()

    def plot_grid(self, title=None, ylabel=None, show_mad=False):
        """Plot the K-correction grid values using matplotlib.

        :param title: Title of the plot
        :type title: string
        :param ylabel: String for the y-label of the plot.
        :type ylabel: string
        :param show_mad: Boolean to indicate whether to show the median
         absolute deviation of the K-correction instead of the mean/median
         values.
        :type show_mad: bool (default = False)
        :return: None

        """

        # Set up figure
        fig = plt.figure(num=None, figsize=(5.5, 6), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.16, bottom=0.1, right=0.81, top=0.98)

        if show_mad:
            grid = self.kcorrgrid_mad
            clabel = r'$\rm{K-correction\ MAD}$'
        else:
            grid = self.kcorrgrid
            clabel = r'$\rm{K-correction}$'

        cax = plt.pcolormesh(self.redsh_mid, self.mag_mid, grid)

        if title is not None:
            cbar_ax = fig.add_axes([0.83, 0.095, 0.04, 0.83])
        else:
            cbar_ax = fig.add_axes([0.83, 0.095, 0.04, 0.89])
        cbar_ax.tick_params(labelsize=15)
        fig.colorbar(cax, cax=cbar_ax).set_label(label=clabel,
                                              size=20)

        ax.tick_params(axis='both', which='major', labelsize=15)

        if ylabel is None:
            ylabel = r'$M_{1450}\,[\rm{mag}]$'

        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_xlabel(r'$\rm{Redshift}\ z$', fontsize=20)

        if title is not None:
            fig.suptitle(r'$\textrm{' + title + '}$', fontsize=18)
            fig.subplots_adjust(top=0.93)

        plt.show()

    def plot_kcorrection(self, redsh_arr=None, mag_arr=None, ylabel=None,
                         save=False, save_filename=None,
                         mag_label=r'$M_{{1450}}={}$'):
        """Plot the K-correction (interpolation) for a range of magnitudes
        over the given redshifts.

        :param redsh_arr: Array of redshifts for which the K-correction is
         evaluated.
        :type redsh_arr: np.ndarray
        :param mag_arr: Array of rest-frame (filter band) magnitudes for
         which the K-correction is evaluated over the given redshift array.
        :param ylabel: String for the y-label of the plot.
        :type ylabel: string
        :param save: Boolean to indicate whether to save the plot.
        :type save: bool
        :param save_filename: Path and name to save the plot.
        :type save_filename: string
        :param mag_label: Label for the magnitude
        :type mag_label: string
        :return: None

        """

        # If no redshift array is given use the redshift range
        if redsh_arr is None:
            redsh_arr = np.linspace(self.redsh_range[0],
                                    self.redsh_range[1],
                                    200)

        # If no magnitude array is given use the mean of the magnitude range
        if mag_arr is None:
            mag_arr = np.array([np.mean(self.mag_range)])

        # Set up figure
        fig = plt.figure(num=None, figsize=(6, 4), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.95)

        # Set up colors
        n = len(mag_arr)
        colors = plt.cm.viridis(np.linspace(0, 0.9, n))

        for idx, mag in enumerate(mag_arr):
            ax.plot(redsh_arr, self.evaluate(np.array([mag]), redsh_arr).ravel(),
                    color=colors[idx],
                    label=mag_label.format(mag))

        if ylabel is None:
            ylabel = r'$\rm{K-correction}$'

        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_xlabel(r'$\rm{Redshift}\ z$', fontsize=20)

        ax.set_xlim(redsh_arr[0], redsh_arr[-1])

        plt.legend()

        if save and save_filename is not None:
            plt.savefig(save_filename)
        else:
            plt.show()

    def evaluate(self, mag, redsh, inverse=False):
        """Evaluate the K-correction for an object of given apparent
        magnitude and redshift.

        :param mag: Magnitude
        :type mag: float
        :param redsh: Redshift
        :type redsh: float
        :return: K-correction
        :rtype: float
        """
        if inverse:
            return self.inv_kcorrection(mag, redsh)
        else:
            return self.kcorrection(mag, redsh)

    def __call__(self, mag, redsh, inverse=False):
        """Return the K-correction for an object of given apparent
        magnitude and redshift.

        :param mag: Magnitude
        :type mag: float
        :param redsh: Redshift
        :type redsh: float
        :return: K-correction
        :rtype: float
        """
        if inverse:
            return self.inv_kcorrection(mag, redsh)
        else:
            return self.kcorrection(mag, redsh)