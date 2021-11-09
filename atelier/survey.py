#!/usr/bin/env python

import scipy
import numpy as np
import pandas as pd


from atelier import lumfun


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
                 selection_function=None):
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
        :param selection_function: lumfun.selfun.QsoSelectionFunction
            Selection function of the survey (default = None).
        """

        self.sky_area = sky_area
        self.sky_area_srd = sky_area / 41253. * 4 * np.pi

        self.obj_lum = obj_df.loc[:, lum_colname].to_numpy()
        self.obj_redsh = obj_df.loc[:, redsh_colname].to_numpy()

        self.obj_df = obj_df
        self.lum_colname = lum_colname
        self.redsh_colname = redsh_colname

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


    def calc_binned_lumfun_new(self, lum_edges, redsh_edges, cosmology,
                           kcorrection, app_mag_lim=None):
        """ My rewrite of Ian's routine using pandas

        :param lum_edges:
        :param redsh_edges:
        :param cosmology:
        :param kcorrection:
        :param app_mag_lim:
        :return:
        """

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
            'raw_counts': np.array([]),
            'corr_counts': np.array([]),
            'corr_counts_unc': np.array([]),
            'filled_bin': np.array([], dtype='bool'),
            'raw_phi': np.array([]),
            'phi': np.array([]),

        }

        # Iterate over all groups and calculate main properties
        for bins, group in gb:

            # Calculate counts and corrected counts
            # raw counts
            raw_count = group.shape[0]
            # corrected counts (inverse selection probability)
            corr_count = np.sum(group['weights'])
            # uncertainty (squared inverse selection probability)
            corr_count_unc = np.sum(group['weights']**2)

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

            # Calculate bin volume
            if inbounds > 0:
                # Luminosity limit as a function of redshift
                lum_limit = lambda z: np.clip(kcorrection.m2M(app_mag_lim, z),
                                            bins[0].left, bins[0].right)
                bin_volume, _ = scipy.integrate.dblquad(
                    lambda M, z: dVdzdO(z), bins[1].left, bins[1].right,
                    lambda z: bins[0].left, lum_limit)
                bin_volume *= self.sky_area_srd

            else:

                bin_volume = 0



            # Calculate binned luminosity function
            if raw_count == 0 or bin_volume == 0:
                raw_phi = None
                phi = None
            else:
                raw_phi = raw_count/bin_volume
                phi = corr_count/bin_volume

            prop_names = ['lum_bin_low', 'lum_bin_upp',
                          'redsh_bin_low', 'redsh_bin_upp',
                          'lum_bin_mid', 'redsh_bin_mid',
                          'lum_median', 'redsh_median',
                          'raw_counts', 'corr_counts', 'corr_counts_unc',
                          'filled_bin', 'raw_phi', 'phi']

            props = [bins[0].left, bins[0].right,
                     bins[1].left, bins[1].right,
                     bins[0].mid, bins[1].mid,
                     group[self.lum_colname].median(),
                     group[self.redsh_colname].median(),
                     raw_count, corr_count, corr_count_unc,
                     filled, raw_phi, phi]

            for idx, name in enumerate(prop_names):
                result_dict[name] = np.append(result_dict[name], props[idx])


        return pd.DataFrame(result_dict)

    def calc_binned_lumfun(self, lum_edges, redsh_edges, cosmology,
                           kcorrection, app_mag_lim):
        """
        Ian's old code rewritten!
        :return:
        """

        lum_edges = np.sort(lum_edges)
        redsh_edges = np.sort(redsh_edges)

        dVdzdO = lumfun.interp_dVdzdO([redsh_edges[0], redsh_edges[-1]],
                                       cosmology)

        lum_bin_mid = lum_edges[:-1] + np.diff(lum_edges)/2.
        redsh_bin_mid = redsh_edges[:-1] + np.diff(redsh_edges)/2.

        print('lum_edges: ', lum_edges)
        print('redsh_edges: ', redsh_edges)

        print('lum_bins: ', lum_bin_mid)
        print('redsh_bins: ', redsh_bin_mid)

        lumfun_shape = lum_bin_mid.shape + redsh_bin_mid.shape

        # Bin survey sources
        lum_i = np.digitize(self.obj_lum, lum_edges) - 1
        redsh_i = np.digitize(self.obj_redsh, redsh_edges) - 1
        # Calc which sources are included in the bins
        incl_idx = np.where((lum_i >= 0) * (lum_i < len(lum_bin_mid)) &
                            (redsh_i >= 0) & (redsh_i < len(redsh_bin_mid)))

        # Initialize result table
        lum_bin_med = np.zeros(lumfun_shape)
        redsh_bin_med = np.zeros(lumfun_shape)
        raw_counts = np.zeros(lumfun_shape)
        corr_counts = np.zeros(lumfun_shape)
        count_uncertainty = np.zeros(lumfun_shape)
        filled = np.zeros(lumfun_shape)

        # Calculate median redshift per bin
        # TODO

        # Calculate the median luminosity per bin

        # Count the sources within the bins
        # (with and without selection probabilities)
        np.add.at(raw_counts, (lum_i[incl_idx], redsh_i[incl_idx]),
                  1)
        np.add.at(corr_counts, (lum_i[incl_idx], redsh_i[incl_idx]),
                  self.obj_weights[incl_idx])
        np.add.at(count_uncertainty, (lum_i[incl_idx], redsh_i[incl_idx]),
                  self.obj_weights[incl_idx] ** 2)

        # Calculate whether the bins are fully covered by the survey
        # TODO: K-correction necessary for this step!
        # Ian's code below
        # ----------------------------------------------------------------------
        # identify which bins are within the flux limit by converting the
        # the luminosity bins to fluxes
        Mbounds, zbounds = np.meshgrid(lum_edges, redsh_edges, indexing='ij')
        mbounds = kcorrection.M2m(Mbounds, zbounds)
        # if M is outside the definition of the k-correction, m2M returns
        # nan. This prevents a warning from the comparison to nan.
        mbounds[np.isnan(mbounds)] = np.inf
        inbounds = mbounds < app_mag_lim
        print('mbounds:', mbounds)
        print('inbounds:', inbounds.shape, inbounds[0].shape)

        # this sums the bin edges 2x2:
        #   4=full covg, 0=no covg, otherwise partial
        inbounds = scipy.ndimage.filters.convolve(inbounds[0].astype(int),
                                                  np.ones((2, 2)))[:-1, :-1]
        # ----------------------------------------------------------------------
        filled[:] = (inbounds == 4)

        bin_volume = np.zeros(lumfun_shape)



        for i, j in zip(*np.where(inbounds > 0)):
            print(i, j)
            lum_lim = lambda z: np.clip(kcorrection.m2M(app_mag_lim, z),
                                     lum_edges[i], lum_edges[i + 1])
            bin_volume[i, j], _ = scipy.integrate.dblquad(
                lambda M, z: dVdzdO(z), redsh_edges[j], redsh_edges[j + 1],
                                      lambda z: lum_edges[i], lum_lim)

        # Calculate luminosity function
        mask = (raw_counts == 0) | (bin_volume == 0)
        bin_volume = np.ma.array(bin_volume * self.sky_area_srd, mask=mask)

        phi = np.ma.divide(corr_counts, bin_volume)
        raw_phi = np.ma.divide(raw_counts, bin_volume)


        # Output is done per redshift bin

        return raw_counts, corr_counts, count_uncertainty, filled, raw_phi, \
               phi, bin_volume
