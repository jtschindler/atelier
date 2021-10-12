#!/usr/bin/env python

import numpy as np

import emcee

import utils as ut

from scipy import optimize

import time

from scipy.interpolate import interp1d

from scipy import integrate

import matplotlib.pyplot as plt

from simqso import lumfun

from astropy.cosmology import FlatLambdaCDM

import os

os.environ["OMP_NUM_THREADS"] = "1"

from multiprocessing import Pool


# The prior and probability need to be defined globally to allow for
# multiprocessing.

def log_prior(theta, parameters):
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
                    use_prior=True):
    if lumfun is None:
        raise ValueError('[ERROR] The luminosity function keyword argument '
                         '"lumfun" is None.')

    if use_prior:
        # Get logarithmic prior
        lp = log_prior(theta, lumfun.free_parameters)
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

        for survey in surveys:
            # Adding the source term contribution
            idx = np.where(survey.obj_selprob > minimum_probability)

            product = survey.obj_selprob[idx] * lumfun.evaluate \
                (survey.obj_lum[idx], survey.obj_redsh[idx])

            source_term += np.sum(np.log(product))

            # Adding the luminosity function integral
            lf_sum = lumfun.integrate_over_lum_redsh(lum_range,
                                                     redsh_range,
                                                     dVdzdO,
                                                     selfun=survey.selection_function)

            lumfun_integral += survey.sky_area_srd * lf_sum

        return source_term + lumfun_integral



class LuminosityFunctionFit(object):

    def __init__(self, lum_range, redsh_range, cosmology, surveys):

        # Main class parameters set by input arguments
        self.cosmology = cosmology
        self.lum_range = lum_range
        self.redsh_range = redsh_range
        self.surveys = surveys
        self.dVdzdO = ut.interp_dVdzdO(redsh_range, cosmology)

        # Emcee default parameters
        self.samplers = None
        self.nwalkers = 50
        self.steps = 1000

        # Nelder-Mead simplex algorithm default parameters
        self.nelder_mead_kwargs = {'full_output': True,
                                   'xtol': 1e-3,
                                   'ftol': 1e-3,
                                   'disp': 1}

# NOW GLOBALLY DEFINED FOR MULTIPROCESSING
    def log_prior(self, theta, parameters):

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

    def log_probability(self, theta, lumfun=None, use_prior=True):

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

                # Adding the source term contribution
                idx = np.where(survey.obj_selprob > minimum_probability)

                product = survey.obj_selprob[idx] * lumfun.evaluate \
                    (survey.obj_lum[idx], survey.obj_redsh[idx])

                source_term += np.sum(np.log(product))

                # Adding the luminosity function integral
                lf_sum = lumfun.integrate_over_lum_redsh(self.lum_range,
                                                         self.redsh_range,
                                                         self.dVdzdO,
                                                         selfun=survey.selection_function)

                lumfun_integral += survey.sky_area_srd * lf_sum


            return source_term + lumfun_integral




    def run_mcmc(self, lumfun, initial_guess=None, nwalkers=None, steps=None):

        if nwalkers is None:
            nwalkers = self.nwalkers
        if steps is None:
            steps = self.steps

        if initial_guess is None:
            parameters = lumfun.get_free_parameters()
            initial_guess = [parameters[par_name].value for par_name in parameters]
            print('[INFO] Initial guess is set to {}'.format(initial_guess))

        pos = initial_guess + 1e-1 * np.random.randn(nwalkers, len(initial_guess))

        ndim = len(initial_guess)

        kwargs = {'lumfun': lumfun,
                  'lum_range': self.lum_range,
                  'redsh_range': self.redsh_range,
                  'surveys': self.surveys,
                  'dVdzdO': self.dVdzdO,
                  'log_prior': log_prior}


        self.sampler =  self.sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                                 log_probability,
                                                 kwargs=kwargs)

        self.sampler.run_mcmc(pos, steps, progress=True)


    def run_mcmc_multiprocess(self, lumfun, initial_guess=None,
                              nwalkers=None, steps=None, processes=None):

        if nwalkers is None:
            nwalkers = self.nwalkers
        if steps is None:
            steps = self.steps

        if initial_guess is None:
            parameters = lumfun.get_free_parameters()
            initial_guess = [parameters[par_name].value for par_name in parameters]
            print('[INFO] Initial guess is set to {}'.format(initial_guess))

        pos = initial_guess + 1e-1 * np.random.randn(nwalkers, len(initial_guess))

        ndim = len(initial_guess)

        kwargs = {'lumfun': lumfun,
                  'lum_range': self.lum_range,
                  'redsh_range': self.redsh_range,
                  'surveys': self.surveys,
                  'dVdzdO': self.dVdzdO,
                  'log_prior': log_prior}


        with Pool(processes=processes) as pool:
            start = time.time()
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                                 log_probability,
                                                 kwargs=kwargs,
                                                 pool=pool)

            self.sampler.run_mcmc(pos, steps, progress=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    def negative_log_likelihood(self, theta, lumfun=None):

        return - self.log_probability(self, theta, lumfun=lumfun)


    def run_NelderMead_simplex(self, lumfun, initial_guess=None, **kwargs):
        """This has not been tested successfully, yet!"""

        if initial_guess is None:
            parameters = lumfun.get_free_parameters()
            initial_guess = [parameters[par_name].value for par_name in
                             parameters]
            print('[INFO] Initial guess is set to {}'.format(initial_guess))

        fit_method = optimize.fmin

        use_prior = False

        result = fit_method(self.log_probability, np.array(initial_guess),
                   args=(lumfun,use_prior),**self.nelder_mead_kwargs)

        return result