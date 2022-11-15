#!/usr/bin/env python

import scipy
import numpy as np
import pandas as pd

from astropy import stats as ap_stats
from scipy import integrate
from scipy import stats

from atelier import lumfun


def return_poisson_confidence(n, bound_low=0.15865, bound_upp=0.84135):
    """
    Return the Poisson confidence interval boundaries for the lower and
    upper bound given a number of events n.

    The default values for ther lower and upper bounds are equivalent to the
    1 sigma confidence interval of a normal distribution.

    :param n:
    :param n_sigma:
    :return:
    """


    lower = stats.chi2.ppf(bound_low, 2*n)/2
    upper = stats.chi2.ppf(bound_upp, 2*(n+1))/2

    return np.array([lower, upper])



class Survey(object):
    """Survey class is a container to hold information on the sources of a
    particular astronomical survey.

    It is used primarily to forward observational data in the calculation of
    luminosity function fits.

    Attributes
    ----------
    obj_df : pandas.DataFrame
        Data frame with information on astronomical sources in the survey.
    lum_colname : string
        The name of the data frame column holding the luminosity (absolute
        magnitude) information of the sources in the survey.
    redsh_colname : string
        The name of the data frame column holding the redshift information of
        the sources in the survey.
    sky_area : float
        Sky area the survey covers in square degrees
    selection_function: lumfun.selfun.QsoSelectionFunction
        Selection function of the survey (default = None).

    """

    def __init__(self, obj_df, lum_colname, redsh_colname, sky_area,
                 selection_function=None,
                 lum_range=None, redsh_range=None,
                 conf_interval='poisson'):
        """Initialize the Survey class.

        :param obj_df: Data frame with information on astronomical sources in the survey.
        :type obj_df: pandas.DataFrame
        :param lum_colname: The name of the data frame column holding the
         luminosity (absolute magnitude) information of the sources in the
         survey.
        :type lum_colname: string
        :param redsh_colname: The name of the data frame column holding the
        redshift information of the sources in the survey.
        :type redsh_colname: string
        :param sky_area: Sky area the survey covers in square degrees
        :type sky_area: float
        :param selection_function: Selection function of the survey
         (default = None).
        :type selection_function: lumfun.selfun.QsoSelectionFunction
        :param lum_range: Luminosity range for the luminosity function fit
         (2-element list).
        :type lum_range: list(float, float)
        :param redsh_range: Redshift range for the luminosity function fit
         (2-element list).
        :type redsh_range: list(float, float)
        :param poisson_conf_interval: Interval for calculation of poission
         uncertainties on the binned luminosity function values.
        :type poisson_conf_interval: string
        """

        self.sky_area = sky_area
        self.sky_area_srd = sky_area / 41253. * 4 * np.pi

        # For calculating the luminosity function fit
        self.lum_range = lum_range
        self.redsh_range = redsh_range

        if self.lum_range is not None:
            obj_df.query('{} < {} < {}'.format(self.lum_range[0],
                                               lum_colname,
                                               self.lum_range[1]),
                         inplace=True)
        if self.lum_range is not None:
            obj_df.query('{} < {} < {}'.format(self.redsh_range[0],
                                               redsh_colname,
                                               self.redsh_range[1]),
                         inplace=True)

        self.obj_lum = obj_df.loc[:, lum_colname].to_numpy()
        self.obj_redsh = obj_df.loc[:, redsh_colname].to_numpy()

        self.obj_df = obj_df
        self.lum_colname = lum_colname
        self.redsh_colname = redsh_colname


        self.conf_interval = conf_interval

        if selection_function is not None:
            self.selection_function = selection_function
            self.obj_selprob = self.selection_function.evaluate(self.obj_lum,
                                                       self.obj_redsh)
        else:
            self.selection_function = None
            self.obj_selprob = np.ones_like(self.obj_lum)

        self.obj_weights = np.clip(self.obj_selprob, 1e-20, 1.0) ** -1

        self.obj_df['weights'] = self.obj_weights
        self.obj_df['selprob'] = self.obj_selprob

    def calc_binned_lumfun_PC2000(self, lum_edges, redsh_edges, cosmology,
                                  kcorrection=None, app_mag_lim=None,
                                  **kwargs,):
        """ Calculation of the binned luminosity function based on the method
        outlined in Page & Carrera 2000

        ADS Link: https://ui.adsabs.harvard.edu/abs/2000MNRAS.311..433P/abstract

        This function is very similar to the other method for calculating the
        binned luminosity function below.

        :param lum_edges:
        :param redsh_edges:
        :param cosmology:
        :param kcorrection:
        :param app_mag_lim:
        :return:
        """

        # Get keyword arguments for the integration
        int_kwargs = {}
        # int_kwargs.setdefault('divmax', kwargs.pop('divmax', 20))
        # int_kwargs.setdefault('tol', kwargs.pop('epsabs', 1e-3))
        # int_kwargs.setdefault('rtol', kwargs.pop('epsrel', 1e-3))
        # int_kwargs.setdefault('divmax', kwargs.pop('divmax', 20))
        int_kwargs.setdefault('epsabs', kwargs.pop('epsabs', 1e-3))
        int_kwargs.setdefault('epsrel', kwargs.pop('epsrel', 1e-3))

        # Sort the bin edges
        lum_edges = np.sort(lum_edges)
        redsh_edges = np.sort(redsh_edges)

        # Set up the differential cosmological volume interpolation
        dVdzdO = lumfun.interp_dVdzdO([redsh_edges[0], redsh_edges[-1]],
                                       cosmology)

        # Take the sample DataFrame initialize groupby object
        # using the bin edges defined above
        gb = self.obj_df.groupby(
            [pd.cut(self.obj_df.loc[:, self.lum_colname], lum_edges),
             pd.cut(self.obj_df.loc[:, self.redsh_colname], redsh_edges)])

        result_dict = {
            'lum_bin_low': np.array([]),
            'lum_bin_upp': np.array([]),
            'redsh_bin_low': np.array([]),
            'redsh_bin_upp': np.array([]),
            'lum_bin_mid': np.array([]),
            'redsh_bin_mid': np.array([]),
            'lum_median': np.array([]),
            'redsh_median': np.array([]),
            'counts': np.array([]),
            'filled_bin': np.array([], dtype='bool'),
            'raw_phi': np.array([]),
            'phi': np.array([]),
            'phi_unc_low': np.array([]),
            'phi_unc_upp': np.array([]),
            'bin_volume': np.array([]),
            'bin_volume_corr': np.array([]),
        }

        # Iterate over all groups and calculate main properties
        for bins, group in gb:

            print(bins)

            # Calculate number counts and Poisson uncertainties
            # raw counts
            count = group.shape[0]
            # count uncertainty
            if self.conf_interval == 'rootn':

                count_unc = ap_stats.poisson_conf_interval(
                count, interval='root-n')

            elif self.conf_interval == 'poisson':

                count_unc = return_poisson_confidence(count)

            else:
                raise ValueError('[ERROR] Confidence interval value not '
                                 'understood. The options are '
                                 '"rootn" or "poisson".')

            # Calculate if bin is fully within survey magnitude limit
            if app_mag_lim is not None:
                Mbounds, zbounds = np.meshgrid([bins[0].left, bins[0].right],
                                               [bins[1].left, bins[1].right],
                                               indexing='ij')
                mbounds = kcorrection.M2m(Mbounds, zbounds)

                mbounds[np.isnan(mbounds)] = np.inf
                inbounds = mbounds < app_mag_lim

                inbounds = scipy.ndimage.filters.convolve(
                    inbounds[0].astype(int),
                    np.ones((2, 2)))[:-1, :-1]

                filled = inbounds == 4
            else:
                filled = None
                inbounds = None

            # Calculate bin volume
            if inbounds is None or (inbounds is not None and inbounds > 0):
                # The calculation of the bin volume takes into account the
                # survey selection function. This is different from the
                # method below.

                # Only use the dblquad integration if the bin is not fully
                # covered in the survey app_mag_lim.
                if inbounds is not None and inbounds < 4:
                    lum_limit = lambda redsh: np.clip(kcorrection.m2M(
                        app_mag_lim, redsh), bins[0].left, bins[0].right)

                    # Double integral
                    bin_volume, _ = scipy.integrate.dblquad(
                        lambda lum, redsh: dVdzdO(
                            redsh), bins[1].left, bins[1].right,
                        lambda redsh: bins[0].left, lum_limit)

                    # Double integral
                    bin_volume_corr, _ = scipy.integrate.dblquad(
                        lambda lum, redsh: dVdzdO(
                            redsh) * self.selection_function.evaluate(lum,
                                                                      redsh),
                        bins[1].left,
                        bins[1].right,
                        lambda redsh: bins[0].left, lum_limit)
                else:
                    integrand = lambda lum, redsh: dVdzdO(redsh)

                    integrand_corr = lambda lum, redsh: dVdzdO(redsh) * \
                                    self.selection_function.evaluate(lum, redsh)


                    inner_integral = lambda redsh: integrate.quad(
                        integrand, bins[0].left, bins[0].right, args=(redsh,),
                        **int_kwargs)[0]

                    inner_integral_corr = lambda redsh: integrate.quad(
                        integrand_corr, bins[0].left, bins[0].right,
                        args=(redsh,), **int_kwargs)[0]

                    bin_volume_corr = integrate.quad(inner_integral_corr,
                                                        bins[1].left,
                                                        bins[1].right,
                                                        **int_kwargs)[0]

                    bin_volume = integrate.quad(inner_integral,
                                                   bins[1].left,
                                                   bins[1].right,
                                                   **int_kwargs)[0]

                bin_volume_corr *= self.sky_area_srd
                bin_volume *= self.sky_area_srd


            # Calculate binned luminosity function
            if count == 0 or bin_volume == 0:
                raw_phi = None
                phi = None
                phi_unc_low = None
                phi_unc_upp = None
            else:

                raw_phi = count / bin_volume

                phi = count / bin_volume_corr
                count_unc = count_unc - count
                phi_unc = count_unc / bin_volume_corr
                phi_unc_low = phi_unc[0]
                phi_unc_upp = phi_unc[1]

            prop_names = ['lum_bin_low', 'lum_bin_upp',
                          'redsh_bin_low', 'redsh_bin_upp',
                          'lum_bin_mid', 'redsh_bin_mid',
                          'lum_median', 'redsh_median',
                          'counts', 'raw_phi',
                          'filled_bin', 'phi', 'phi_unc_low',
                          'phi_unc_upp', 'bin_volume', 'bin_volume_corr']

            props = [bins[0].left, bins[0].right,
                     bins[1].left, bins[1].right,
                     bins[0].mid, bins[1].mid,
                     group[self.lum_colname].median(),
                     group[self.redsh_colname].median(),
                     count, raw_phi,
                     filled, phi, phi_unc_low, phi_unc_upp,
                     bin_volume, bin_volume_corr]

            for idx, name in enumerate(prop_names):
                result_dict[name] = np.append(result_dict[name], props[idx])

        return pd.DataFrame(result_dict)

