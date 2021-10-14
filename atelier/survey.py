#!/usr/bin/env python

import numpy as np

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

        if selection_function is not None:
            self.selection_function = selection_function
            self.obj_selprob = self.selection_function(self.obj_lum,
                                                       self.obj_redsh)
        else:
            self.selection_function = None
            self.obj_selprob = np.ones_like(self.obj_lum)

