#!/usr/bin/env python

import os
import time
import emcee
import numpy as np
from scipy import optimize
from multiprocessing import Pool

import atelier.lumfun as lumfun

os.environ["OMP_NUM_THREADS"] = "1"


# The prior and probability need to be defined globally to allow for
# multiprocessing.

def log_prior(theta, parameters):
    """ Logarithmic prior function for luminosity function maximum likelihood
     estimation (MLE) using emcee in multiprocessing mode.

    :param theta: parameter vector
    :type theta: list of parameter values
    :param parameters: Dictionary of parameters used in MLE. The
     Parameter.bounds attribute is used to keep the fit within the
     pre-defined parameter boundaries (bounds).
    :type parameters: dict(atelier.lumfun.Parameter)
    :return: Logarithmic prior
    :rtype: float
    """
    # Define an uninformed prior for now, possibly change later, generalize

    within_bounds = []

    for idx, value in enumerate(theta):
        key = list(parameters.keys())[idx]
        bounds = parameters[key].bounds
        if bounds is None:
            within_bounds.append(True)
        elif bounds[0] < value < bounds[1]:
            within_bounds.append(True)
        else:
            within_bounds.append(False)
    if all(within_bounds):
        return 0
    else:
        return -np.inf


def log_probability(theta, lumfun=None, lum_range=None,
                    redsh_range=None, surveys=None, dVdzdO=None, log_prior=None,
                    use_prior=True, int_mode='romberg'):
    """ Logarithmic probability for luminosity function maximum likelihood
    estimation (MLE) using emcee in multiprocessing mode.

    The logarithmic probability implemented here goes back to the definition
    of Marshall+1983 Equation 3.

    ADS reference: https://ui.adsabs.harvard.edu/abs/1983ApJ...269...35M/abstract

    :param theta: parameter vector
    :type theta: list of parameter values
    :param lumfun: Luminosity function model
    :type lumfun: atelier.lumfun.LuminosityFunction (or children classes)
    :param lum_range: Luminosity range
    :type lum_range: tuple
    :param redsh_range: Redshift range
    :type redsh_range: tuple
    :param surveys: List of surveys
    :type surveys: list(atelier.survey.Survey)
    :param dVdzdO: Differential comoving solid volume element
    :type dVdzdO: function
    :param log_prior: Functiond describing the logarithmic prior
    :type log_prior:
    :param use_prior: Boolean to indicate whether the logarithmic prior
     function will be used or strictly flat priors for all parameters will be
     adopted.
    :type use_prior: bool
    :return: Logarithmic probability
    :rtype: float
    """

    if lumfun is None:
        raise ValueError('[ERROR] The luminosity function keyword argument '
                         '"lumfun" is None.')

    if use_prior:
        # Get logarithmic prior for the free parameters of the luminosity
        # function class.
        lp = log_prior(theta, lumfun.get_free_parameters())
    else:
        lp = 0

    # Consider moving this to a selection function class
    minimum_probability = 1e-4

    # Return negative infinity if prior is infinite
    if not np.isfinite(lp):
        return -np.inf

    # Calculate the logarithmic probability
    else:
        source_term = 0
        lumfun_integral = 0

        # Update free parameters of the luminosity function
        lumfun.update_free_parameter_values(theta)

        for survey in surveys:

            lum_range = survey.lum_range
            redsh_range = survey.redsh_range

            # Adding the source term contribution
            idx = np.where(survey.obj_selprob > minimum_probability)

            product = survey.obj_selprob[idx] * lumfun.evaluate \
                (survey.obj_lum[idx], survey.obj_redsh[idx])

            source_term += np.sum(np.log(product))

            # Adding the luminosity function integral
            if int_mode == 'romberg':
                lf_sum = lumfun.integrate_over_lum_redsh(lum_range,
                                                         redsh_range,
                                                         dVdzdO=dVdzdO,
                                                         selfun=
                                                         survey.selection_function)
            elif int_mode == 'simpson':

                lf_sum = lumfun.integrate_over_lum_redsh_simpson(
                    lum_range,
                    redsh_range,
                    dVdzdO=dVdzdO,
                    selfun=survey.selection_function)
            else:
                raise NotImplementedError('[ERROR] Integration mode {} '
                                          'is not implemented.'.format(
                    int_mode))

            lumfun_integral -= survey.sky_area_srd * lf_sum

        return source_term + lumfun_integral



class LuminosityFunctionFit(object):
    """ Class that carries out maximum likelihood fits to a luminosity
    function model using MCMC (emcee).

    Attributes
    ----------
    lum_range : list(float, float)
        Luminosity range for the luminosity function fit (2-element list).
    redsh_range : list(float, float)
        Redshift range for the luminosity function fit (2-element list).
    cosmology : astropy.cosmology.Cosmology
        Cosmology object
    surveys : list(survey.Survey)
        List of survey objects used in to fit the luminosity function model to.
    emcee_sampler : emcee.EnsembleSampler
        Emcee Ensemble sampler object. This attribute cannot be initialized
        but is internally populated during the .run_mcmc or
        .run_mcmc_multiprocess methods.
    emcee_nwalker : int (default = 50)
        Number of walkers for the emcee sampler.
    emcee_steps : int (default = 1000)
        Number of steps for the emcee sampler.
    """

    def __init__(self, lum_range, redsh_range, cosmology, surveys,
                 emcee_nwalkers=50, emcee_steps=1000):
        """Initialize the LuminosityFunctionFit class.

        :param lum_range: Luminosity range for the luminosity function fit
         (2-element list).
        :type lum_range: list(float, float)
        :param redsh_range: Redshift range for the luminosity function fit
         (2-element list).
        :type redsh_range: list(float, float)
        :param cosmology: Cosmology object
        :type cosmology: astropy.cosmology.Cosmology
        :param surveys: List of survey objects used in to fit the luminosity
         function model to.
        :type surveys: list(survey.Survey)
        :param emcee_nwalkers: Number of walkers for the emcee sampler.
        :type emcee_nwalkers: int (default = 50)
        :param emcee_steps: Number of steps for the emcee sampler.
        :type emcee_steps: int (default = 1000)
        """

        # Main class parameters set by input arguments
        self.cosmology = cosmology
        self.lum_range = lum_range
        self.redsh_range = redsh_range
        self.surveys = surveys
        self.dVdzdO = lumfun.interp_dVdzdO(redsh_range, cosmology)

        # Emcee default parameters
        self.sampler = None
        self.nwalkers = emcee_nwalkers
        self.steps = emcee_steps

        # Nelder-Mead simplex algorithm default parameters
        self.nelder_mead_kwargs = {'full_output': True,
                                   'xtol': 1e-3,
                                   'ftol': 1e-3,
                                   'disp': 1}

# NOW GLOBALLY DEFINED FOR MULTIPROCESSING
    def log_prior(self, theta, parameters):
        """ Logarithmic prior function for luminosity function maximum likelihood
        estimation (MLE).

        :param theta: parameter vector
        :type theta: list of parameter values
        :param parameters: Dictionary of parameters used in MLE. The
         Parameter.bounds attribute is used to keep the fit within the
         pre-defined parameter boundaries (bounds).
        :type parameters: dict(atelier.lumfun.Parameter)
        :return: Logarithmic prior
        :rtype: float
        """

        # Define an uninformed prior for now, possibly change later, generalize
        within_bounds = []

        for idx, value in enumerate(theta):
            key = list(parameters.keys())[idx]
            bounds = parameters[key].bounds
            if bounds is None:
                within_bounds.append(True)
            elif bounds[0] < value < bounds[1]:
                within_bounds.append(True)
            else:
                within_bounds.append(False)
        if all(within_bounds):
            return 0
        else:
            return -np.inf

    def log_probability(self, theta, lumfun=None, use_prior=True,
                        int_mode='romberg'):
        """ Logarithmic probability for luminosity function maximum likelihood
        estimation (MLE).

        The logarithmic probability implemented here goes back to the definition
        of Marshall+1983 Equation 3.

        ADS reference: https://ui.adsabs.harvard.edu/abs/1983ApJ...269...35M/abstract

        :param theta: parameter vector
        :type theta: list of parameter values
        :param lumfun: Luminosity function model
        :type lumfun: atelier.lumfun.LuminosityFunction (or children classes)
        :type use_prior: bool
        :return: Logarithmic probability
        :rtype: float
        """

        if lumfun is None:
            raise ValueError('[ERROR] The luminosity function keyword argument '
                             '"lumfun" is None.')

        if use_prior:
            # Get logarithmic prior
            lp = self.log_prior(theta, lumfun.free_parameters)
        else:
            lp = 0

        # Consider moving this to a selection function class
        minimum_probability = 1e-3

        # Return negative infinity if prior is infinite
        if not np.isfinite(lp):
            return -np.inf
        # Calculate the logarithmic probability
        else:
            source_term = 0
            lumfun_integral = 0

            # Update free parameters of the luminosity function
            lumfun.update_free_parameter_values(theta)

            for survey in self.surveys:

                lum_range = survey.lum_range
                redsh_range = survey.redsh_range

                # Adding the source term contribution
                idx = np.where(survey.obj_selprob > minimum_probability)

                product = survey.obj_selprob[idx] * lumfun.evaluate \
                    (survey.obj_lum[idx], survey.obj_redsh[idx])

                source_term += np.sum(np.log(product))

                # Adding the luminosity function integral
                if int_mode == 'romberg':
                    lf_sum = lumfun.integrate_over_lum_redsh(lum_range,
                                                             redsh_range,
                                                             dVdzdO=self.dVdzdO,
                                                             selfun=
                                                             survey.selection_function)
                elif int_mode == 'simpson':

                    lf_sum = lumfun.integrate_over_lum_redsh_simpson(
                        lum_range,
                        redsh_range,
                        dVdzdO=self.dVdzdO,
                        selfun=survey.selection_function)
                else:
                    raise NotImplementedError('[ERROR] Integration mode {} '
                                              'is not implemented.'.format(
                                               int_mode))

                lumfun_integral += survey.sky_area_srd * lf_sum

            return source_term + lumfun_integral




    def run_mcmc(self, lumfun, initial_guess=None, nwalkers=None, steps=None,
                 int_mode='romberg'):
        """Run the emcee MCMC EnsembleSampler to fit the luminosity function
        model to the survey data using maximum likelihood estimation.

        :param lumfun: Luminosity function model
        :type lumfun: atelier.lumfun.LuminosityFunction (or children classes)
        :param initial_guess: Initial guess of the free parameters (ordered
         as in the luminosity function model.
        :type initial_guess: list(float) (default = None)
        :param nwalkers: Number of walkers for the emcee sampler overriding (
         not overwriting) the class attribute emcee_nwalkers.
        :type nwalkers: int (default = None)
        :param steps: Number of steps for the emcee sampler overriding (
         not overwriting) the class attribute emcee_nwalkers.
        :type steps: int (default = None)
        :return: None
        """

        if nwalkers is None:
            nwalkers = self.nwalkers
        if steps is None:
            steps = self.steps

        if initial_guess is None:
            parameters = lumfun.get_free_parameters()
            initial_guess = [parameters[par_name].value for par_name in parameters]
            print('[INFO] Initial guess is set to {}'.format(initial_guess))

        for survey in self.surveys:
            if survey.lum_range is None or survey.redsh_range is None:
                raise ValueError('[ERROR] Survey limits (lum, redsh) are not '
                                 'defined.')

        pos = initial_guess + 1e-1 * np.random.randn(nwalkers, len(initial_guess))

        ndim = len(initial_guess)

        kwargs = {'lumfun': lumfun,
                  'lum_range': self.lum_range,
                  'redsh_range': self.redsh_range,
                  'surveys': self.surveys,
                  'dVdzdO': self.dVdzdO,
                  'log_prior': log_prior,
                  'int_mode': int_mode}

        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                             kwargs=kwargs)

        self.sampler.run_mcmc(pos, steps, progress=True)


    def run_mcmc_multiprocess(self, lumfun, initial_guess=None,
                              nwalkers=None, steps=None, processes=None,
                              int_mode='romberg'):
        """Run the emcee MCMC EnsembleSampler to fit the luminosity function
        model to the survey data using maximum likelihood estimation in
        multiprocessing mode.

        :param lumfun: Luminosity function model
        :type lumfun: atelier.lumfun.LuminosityFunction (or children classes)
        :param initial_guess: Initial guess of the free parameters (ordered
         as in the luminosity function model.
        :type initial_guess: list(float) (default = None)
        :param nwalkers: Number of walkers for the emcee sampler overriding (
         not overwriting) the class attribute emcee_nwalkers.
        :type nwalkers: int (default = None)
        :param steps: Number of steps for the emcee sampler overriding (
         not overwriting) the class attribute emcee_nwalkers.
        :type steps: int (default = None)
        :param processes: Number of processes for multiprocessing. The
         default value of 'None' will use the maximum number determinded by
         the multiprocessing package.
        :type processes: int (default = None)
        :return:
        """

        if nwalkers is None:
            nwalkers = self.nwalkers
        if steps is None:
            steps = self.steps

        if initial_guess is None:
            parameters = lumfun.get_free_parameters()
            initial_guess = [parameters[par_name].value for par_name in parameters]
            print('[INFO] Initial guess is set to {}'.format(initial_guess))

        for survey in self.surveys:
            if survey.lum_range is None or survey.redsh_range is None:
                raise ValueError('[ERROR] Survey limits (lum, redsh) are not '
                                 'defined.')

        pos = initial_guess + 1e-1 * np.random.randn(nwalkers, len(initial_guess))

        ndim = len(initial_guess)

        kwargs = {'lumfun': lumfun,
                  'lum_range': self.lum_range,
                  'redsh_range': self.redsh_range,
                  'surveys': self.surveys,
                  'dVdzdO': self.dVdzdO,
                  'log_prior': log_prior,
                  'int_mode': int_mode}


        with Pool(processes=processes) as pool:
            start = time.time()
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                                 log_probability,
                                                 kwargs=kwargs,
                                                 pool=pool)

            self.sampler.run_mcmc(pos, steps, progress=True)
            end = time.time()
            multi_time = end - start
            print("[INFO] Multiprocessing took {0:.1f} seconds".format(
                multi_time))

    def negative_log_likelihood(self, theta, lumfun):
        """Return the negative logarithmic likelihood for a set of parameters
        'theta' and a luminosity function model 'lumfun' using the default
        logarithmic likelihood definition.

        :param theta: parameter vector
        :type theta: list of parameter values
        :param lumfun: Luminosity function model
        :type lumfun: atelier.lumfun.LuminosityFunction (or children classes)
        :return: Negative logarithmic likelihood
        :rtype: float
        """

        return - self.log_probability(self, theta, lumfun=lumfun)


    def run_NelderMead_simplex(self, lumfun, initial_guess=None, **kwargs):

        if initial_guess is None:
            parameters = lumfun.get_free_parameters()
            initial_guess = [parameters[par_name].value for par_name in
                             parameters]
            print('[INFO] Initial guess is set to {}'.format(initial_guess))

        fit_method = optimize.fmin

        use_prior = False

        result = fit_method(self.log_probability, np.array(initial_guess),
                   args=(lumfun, use_prior), **self.nelder_mead_kwargs)

        return result