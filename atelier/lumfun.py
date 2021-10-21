#!/usr/bin/env python

import numpy as np
from astropy import units
from scipy import integrate
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM


# Basic functionality needed for this class
def interp_dVdzdO(redsh_range, cosmo):
    """Interpolate the differential comoving solid volume element
    :math:`(dV/dz){d\Omega}` over the specified redshift range
    zrange = :math:`(z_1,z_2)`.

    This interpolations speeds up volume (redshift, solid angle) integrations
    for the luminosity function without significant loss in accuracy.

    The resolution of the redshift array, which will be interpolated is
    :math:`\Delta z=0.025`.

    :param redsh_range: Redshift range for interpolation
    :type redsh_range: tuple
    :param cosmo: Cosmology
    :type cosmo: astropy.cosmology.Cosmology
    :return: 1D interpolation function
    """

    redsharray = np.arange(redsh_range[0] - 0.01, redsh_range[1] + 0.0251, 0.025)
    diff_co_vol = [cosmo.differential_comoving_volume(redsh).value for redsh in
                   redsharray]
    return interp1d(redsharray, diff_co_vol)


def mag_double_power_law(mag, phi_star, mag_star, alpha, beta):
    """Evaluate a broken double power law luminosity function as a function
    of magnitude.

    :param mag: Magnitude
    :type mag: float or np.ndarray
    :param phi_star: Normalization of the broken power law at a value of
     mag_star
    :type phi_star: float
    :param mag_star: Break magnitude of the power law
    :type mag_star: float
    :param alpha: First slope of the broken power law
    :type alpha: float
    :param beta: Second slope of the broken power law
    :type beta: float
    :return: Value of the broken double power law at a magnitude of M
    :rtype: float or np.ndarray
    """
    A = pow(10, 0.4 * (alpha + 1) * (mag - mag_star))
    B = pow(10, 0.4 * (beta + 1) * (mag - mag_star))

    return phi_star / (A + B)


def lum_double_power_law(lum, phi_star, lum_star, alpha, beta):
    """Evaluate a broken double power law luminosity function as a function
    of luminosity.

    :param lum: Luminosity
    :type lum: float or np.ndarray
    :param phi_star: Normalization of the broken power law at a value of
     lum_star
    :type phi_star: float
    :param lum_star: Break luminosity of the power law
    :type lum_star: float
    :param alpha: First slope of the broken power law
    :type alpha: float
    :param beta: Second slope of the broken power law
    :type beta: float
    :return: Value of the broken double power law at a magnitude of M
    :rtype: float or np.ndarray
    """

    A = pow((lum / lum_star), alpha)
    B = pow((lum / lum_star), beta)

    return phi_star / (A + B)


def mag_single_power_law(mag, phi_star, mag_ref, alpha):
    """Evaluate a power law luminosity function as a function as a function
    of magnitude

    :param mag: Magnitude
    :type mag: float or np.ndarray
    :param phi_star: Normalization of the power law at a value of mag_ref
    :type phi_star: float
    :param mag_ref: Reference magnitude of the power law
    :type mag_ref: float
    :param alpha: Slope of the power law
    :type alpha: float
    :return: Value of the broken double power law at a magnitude of M
    :rtype: float or np.ndarray
    """

    A = pow(10, 0.4 * (alpha + 1) * (mag - mag_ref))

    return phi_star / A


# Class functions
class Parameter(object):
    """ A class providing a data container for a parameter used in the
    luminosity function class.

    Attributes
    ----------
    value : float
        Value of the parameter
    name : string
        Name of the parameter
    bounds : tupler
        Bounds of the parameter, used in fitting
    vary : bool
        Boolean to indicate whether this parameter should be varied, used in
        fitting
    one_sigma_unc: list (2 elements)
        1 sigma uncertainty of the parameter.

    """

    def __init__(self, value, name, bounds=None, vary=True, one_sigma_unc=None):
        """Initialize the Parameter class and its attributes.

        """

        self.value = value
        self.name = name
        self.bounds = bounds
        self.vary = vary
        self.one_sigma_unc = one_sigma_unc


class LuminosityFunction(object):
    """ The base luminosity function class.

    In this implementation a luminosity function is defined in terms of

    - luminosity ("lum") and
    - redshift ("redsh")
    - a list of main parameters ("main_parameters")

    The number of main parameters depend on the functional form and can
    themselves be functions ("param_functions") of luminosity, redshift or
    additional "parameters".

    This general framework should facilitate the implementation of a wide
    range of luminosity functions without confining limits.

    An automatic initialization will check whether the parameter functions
    and parameters define all necessary main parameters.

    While the code does not distinguish between continuum luminosities,
    broad band luminosities or magnitudes, some inherited functionality is
    based on specific luminosity definitions.
    In order for these functions to work a luminosity type "lum_type" has to
    be specified. The following luminosity types have special functionality:

    - "M1450" : Absolute continuum magnitude measured at 1450A in the
      rest-frame.


    Attributes
    ----------
    parameters : dict(atelier.lumfun.Parameter)
        Dictionary of Parameter objects, which are used either as a main
        parameters for the calculation of the luminosity function or as an
        argument for calculating a main parameter using a specified parameter
        function "param_function".
    param_functions : dict(functions}
        Dictionary of functions with argument names for which the parameter
        attribute provides a Parameter or the luminosity "lum" or the
        redshift "redsh".
    main_parameters : list(string)
        List of string providing names for the main parameters of the
        luminosity function. During the initialization the main parameters
        need to be either specified as a Parameter within the "parameters"
        attribute or as a function by the "param_functions" attribute.
    lum_type : string (default=None)
        Luminosity type checked for specific functionality
    verbose : int
        Verbosity (0: no output, 1: minimal output)

    """

    def __init__(self, parameters, param_functions, main_parameters,
                 lum_type=None, verbose=1):
        """ Initialize the base luminosity function class.
        """

        self.verbose = verbose
        self.parameters = parameters

        # The main parameters are the parameters which get passed into the
        # functional form of the luminosity function, they can themselves be
        # functions parameters (incl. redshift and luminosity dependence).
        self.main_parameters = main_parameters
        self.param_functions = param_functions

        self.lum_type = lum_type

        self._initialize_parameters_and_functions()


        self.free_parameters = self.get_free_parameters()
        self.free_parameter_names = list(self.free_parameters.keys())



    def __call__(self, lum, redsh, parameters=None):
        """ Call function that evaluates luminosity function at the given
        luminosity and redshift.

        :param lum: float or numpy.ndarray
        :type lum: Luminosity for evaluation
        :param redsh: float or numpy.ndarray
        :type redsh: Redshift for evaluation
        :param parameters: Dictionary of parameters used for this specific
            calculation. This does not replace the parameters with which the
            luminosity function was initialized.
        :type parameters: dict(atelier.lumfunParameters)
        :return: Luminosity function value at the provided luminosity and
            redshift
        :rtype: float or numpy.ndarray
        """
        if parameters is None:
            parameters = self.parameters

        return self.evaluate(lum, redsh, parameters)


    def _initialize_parameters_and_functions(self):
        """Internal function that checks the supplied parameters and
        parameter functions.

        """

        if self.verbose > 0:
            print('[INFO]---------------------------------------------------')
            print('[INFO] Performing initialization checks ')
            print('[INFO]---------------------------------------------------')

        # Check if main parameters are represented within the supplied
        # parameters and param_functions

        for main_param in self.main_parameters:

            if self.verbose > 0:
                print(
                    '[INFO]---------------------------------------------------')

            if main_param in self.param_functions and callable(
                    self.param_functions[main_param]):

                if self.verbose > 0:
                    print('[INFO] Main parameter {} is described by a '
                          'function.'.format(main_param))

                # Retrieve all parameter function parameters.
                function = self.param_functions[main_param]
                n_arg = function.__code__.co_argcount
                func_params = list(function.__code__.co_varnames[:n_arg])

                if self.verbose >0:
                    print('[INFO] The function parameters are: {}'.format(
                        func_params))

                # Remove parameters that are arguments of the luminosity.
                # function itself.
                if 'lum' in func_params:
                    func_params.remove('lum')
                if 'redsh' in func_params:
                    func_params.remove('redsh')

                # Check if all parameter function parameters are available.
                if all(param_name in self.parameters.keys() for param_name in
                       func_params):
                    if self.verbose > 0:
                        print('[INFO] All parameters are supplied.')
                        print('[INFO] Parameters "lum" and "redsh" were '
                              'ignored as they are luminosity function '
                              'arguments.')
                else:
                    raise ValueError('[ERROR] Main parameter function {} has '
                                     'not supplied parameters.'.format(
                        main_param))
            elif main_param in self.param_functions and not callable(
                    self.param_functions[main_param]):
                raise ValueError(
                    '[ERROR] Main parameter function {} is not callable'.format(
                        main_param))
            elif main_param in self.parameters:

                if self.verbose > 0:
                    print('[INFO] Main parameter {} is supplied as a normal '
                          'parameter.'.format(main_param))
            else:
                raise ValueError('[ERROR] Main parameter {} is not supplied '
                                 'as a parameter function or simple '
                                 'parameter.'.format(main_param))
        if self.verbose > 0:
            print('[INFO]---------------------------------------------------')
            print('[INFO] Initialization check passed.')
            print('[INFO]---------------------------------------------------')

    @staticmethod
    def _evaluate_param_function(param_function, parameters):
        """Internal function to evaluate the parameters function at the given
        parameter values.

        :param param_function: Parameter function
        :type param_function:
        :param parameters: Parameters of the luminosity function. Luminosity
        "lum" and redshift "redsh" need to be included along with the other
        parameters.
        :type parameters: dict(atelier.lumfun.Parameter)

        :return: Value of parameter function given the parameters
        :rtype: float
        """

        # Retrieve function parameters
        n_arg = param_function.__code__.co_argcount
        func_params = list(param_function.__code__.co_varnames[:n_arg])
        # Retrive the parameter values from the parameter dictionary
        parameter_values = [parameters[par_name].value for par_name in
                            func_params]

        # Evaluate the function and return the value
        return param_function(*parameter_values)

    def get_free_parameters(self):
        """Return a dictionary with all parameters for which vary == True.

        :return: All parameters with vary == True
        :rtype: dict(atelier.lumfun.Parameter)
        """

        free_parameters = {p: self.parameters[p] for p in self.parameters if
                           self.parameters[p].vary}

        if 'redsh' in free_parameters.keys():
            free_parameters.pop('redsh')
        if 'lum' in free_parameters.keys():
            free_parameters.pop('lum')

        return free_parameters

    def get_free_parameter_names(self):
        """Return a list of names with all parameters for which vary == True.

        :return: Names of parameters with vary == True
        :rtype: list(str)
        """

        if self.free_parameters is None:
            self.free_parameters = self.get_free_parameters()

        return list(self.free_parameters.keys())

    def print_free_parameters(self):
        """Print a list of all free (vary==True) parameters.
        """

        for param in self.free_parameters:
            name = self.free_parameters[param].name
            value = self.free_parameters[param].value
            bounds = self.free_parameters[param].bounds
            vary = self.free_parameters[param].vary

            print('Parameter {} = {}, bounds={}, vary={}'.format(name, value,
                                                                 bounds, vary))

    def print_parameters(self):
        """Print a list of all parameters.
        """

        for param in self.parameters:
            name = self.parameters[param].name
            value = self.parameters[param].value
            bounds = self.parameters[param].bounds
            vary = self.parameters[param].vary
            unc = self.parameters[param].one_sigma_unc

            print('Parameter {} = {}, bounds={}, vary={}, unc={}'.format(name,
                                                                     value,
                                                                 bounds,
                                                                         vary,
                                                                         unc))

    def update_free_parameter_values(self, values):
        """Update all free parameters with new values.

        :param values: Values in the same order as the order of free parameters.
        :type values: list(float)
        """

        for idx, value in enumerate(values):
            param_name = self.free_parameter_names[idx]
            if self.verbose > 1:
                print(
                    '[INFO] Updating {} from {} to {}'.format(param_name, value,
                                                              self.parameters[
                                                                  param_name].value))
            self.parameters[param_name].value = value

    def evaluate_main_parameters(self, lum, redsh, parameters=None):
        """Evaluate the main parameters of the luminosity function

        :param lum: float or numpy.ndarray
        :type lum: Luminosity for evaluation
        :param redsh: float or numpy.ndarray
        :type redsh: Redshift for evaluation
        :param parameters: Dictionary of parameters used for this specific
            calculation. This does not replace the parameters with which the
            luminosity function was initialized.
        :type parameters: dict(atelier.lumfun.Parameters)
        :return:
        """

        if parameters is None:
            parameters = self.parameters.copy()

        parameters['redsh'] = Parameter(redsh, 'redsh')
        parameters['lum'] = Parameter(lum, 'lum')

        main_param_values = {}

        for main_param in self.main_parameters:
            # If main parameter is a parameter function, evaluate the function.
            if main_param in self.param_functions:
                function = self.param_functions[main_param]

                main_param_values[main_param] = \
                    self._evaluate_param_function(function, parameters)
            # If main parameter is a parameter, get its value.
            elif main_param in self.parameters:
                main_param_values[main_param] = self.parameters[
                    main_param].value
            # Else, raise value error
            else:
                raise ValueError('[ERROR] Main parameter {} cannot be '
                                 'evaluated. Neither a function or a '
                                 'parameter could be associated with '
                                 'the main parameter. This should have '
                                 'been caught at the initialization '
                                 'stage.'.format(
                    main_param))

        return main_param_values

    def evaluate(self, lum, redsh, parameters=None):
        """Evaluate the luminosity function at the given luminosity and
        redshift.

        :raise NotImplementedError

        :param lum: Luminosity for evaluation
        :type lum: float or numpy.ndarray
        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param parameters: Dictionary of parameters used for this specific
            calculation. This does not replace the parameters with which the
            luminosity function was initialized. (default=None)
        :type parameters: dict(atelier.lumfun.Parameters)
        """
        raise NotImplementedError

    def _redshift_density_integrand(self, lum, redsh, dVdzdO):
        """Internal function providing the integrand for the sample function.

        :param lum: float or numpy.ndarray
        :type lum: Luminosity for evaluation
        :param redsh: float or numpy.ndarray
        :type redsh: Redshift for evaluation
        :param dVdzdO: Differential comoving solid volume element
        :type dVdzdO: function
        :return: Returns :math:`\Phi(L,z)\times (dV/dz){d\Omega}`
        :rtype: float or numpy.ndarray
        """

        return self.evaluate(lum, redsh) * dVdzdO(redsh)



    def _redshift_density_integrand_selfun(self, lum, redsh, dVdzdO, selfun):
        """Internal function providing the integrand for the sample function
        including a selection function contribution.

        :param lum: float or numpy.ndarray
        :type lum: Luminosity for evaluation
        :param redsh: float or numpy.ndarray
        :type redsh: Redshift for evaluation
        :param dVdzdO: Differential comoving solid volume element
        :type dVdzdO: function
        :param selfun: Selection function
        :type selfun: atelier.selfun.QsoSelectionFunction
        :return: Returns :math:`\Phi(L,z)\times (dV/dz){d\Omega}`
        :rtype: float or numpy.ndarray
        """

        return self.evaluate(lum, redsh) * dVdzdO(redsh) * selfun(lum,
                                                                  redsh)


    def integrate_over_lum_redsh(self, lum_range, redsh_range, dVdzdO=None,
                                 selfun=None,
                                 cosmology=None, mode='fast', **kwargs):
        """Calculate the number of sources described by the luminosity function
        over a luminosity and redshift interval in units of per steradian.

        Either a cosmology or dVdzdO have to be supplied.

        :param lum_range: Luminosity range
        :type lum_range: tuple
        :param redsh_range: Redshift range
        :type redsh_range: tuple
        :param dVdzdO: Differential comoving solid volume element (default =
            None)
        :type dVdzdO: function
        :param selfun: Selection function (default = None)
        :type selfun: atelier.selfun.QsoSelectionFunction
        :param cosmology: Cosmology (default = None)
        :type cosmology: astropy.cosmology.Cosmology
        :param mode: String setting the integration mode (default='fast').
            Only one mode ('fast') is currently implemented.
        :type mode: string
        :param kwargs:
        :return: :math:`N = \int\int\Phi(L,z) (dV/(dz d\Omega)) dL dz`
        :rtype: float
        """

        if mode != 'fast':
            raise ValueError('[ERROR] Only a "fast" integration using a '
                             'Romberg integration is currently available.')

        # Get keyword arguments for the integration
        int_kwargs = {}
        int_kwargs.setdefault('divmax', kwargs.pop('divmax', 20))
        int_kwargs.setdefault('tol', kwargs.pop('epsabs', 1e-3))
        int_kwargs.setdefault('rtol', kwargs.pop('epsrel', 1e-3))

        # Set up the interpolated differential comoving solid volume element
        if dVdzdO is None and cosmology is not None:
            dVdzdO = interp_dVdzdO(redsh_range, cosmology)
        elif dVdzdO is None and cosmology is None:
            raise ValueError(
                '[ERROR] Either a cosmology or dVdzdO have to be supplied.')

        if selfun is None:
            # Integrate over the luminosity and redshift range
            integrand = self._redshift_density_integrand

            inner_integral = lambda redsh: integrate.romberg(integrand,
                                                             *lum_range,
                                                             args=(redsh,
                                                                   dVdzdO),
                                                             **int_kwargs)
            outer_integral = integrate.romberg(inner_integral, *redsh_range,
                                               **int_kwargs)

        else:
            # Integrate over the luminosity and redshift range, including the
            # selection function.

            integrand = self._redshift_density_integrand_selfun

            inner_integral = lambda redsh: integrate.romberg(integrand,
                                                             *lum_range,
                                                             args=(redsh,
                                                                   dVdzdO,
                                                                   selfun),
                                                             **int_kwargs)
            outer_integral = integrate.romberg(inner_integral, *redsh_range,
                                               **int_kwargs)

        return outer_integral


    def redshift_density(self, redsh, lum_range, dVdzdO, **kwargs):
        """Calculate the volumetric source density described by the luminosity
        function at a given redshift and over a luminosity interval in units of
        per steradian per redshift.

        :param redsh: Redshift
        :type redsh: float
        :param lum_range: Luminosity range
        :type lum_range: tuple
        :param dVdzdO: Differential comoving solid volume element (default =
            None)
        :type dVdzdO: function
        :param kwargs:
        :return: :math:`\int \Phi(L,z) (dV/dz){d\Omega} dL`
        :rtype: float
        """

        # Get keyword arguments for the integration
        int_kwargs = {}
        int_kwargs.setdefault('divmax', kwargs.pop('divmax', 20))
        int_kwargs.setdefault('tol', kwargs.pop('epsabs', 1e-3))
        int_kwargs.setdefault('rtol', kwargs.pop('epsrel', 1e-3))

        integral = integrate.romberg(self._redshift_density_integrand,
                                     lum_range[0],
                                     lum_range[1],
                                     args=(redsh, dVdzdO), **int_kwargs)

        return integral


    def integrate_lum(self, redsh, lum_range, **kwargs):
        """Calculate the volumetric source density described by the luminosity
        function at a given redshift and over a luminosity interval in units of
        per Mpc^3.

        :param redsh: Redshift
        :type redsh: float
        :param lum_range: Luminosity range
        :type lum_range: tuple
        :param kwargs:
        :return: :math:`\int \Phi(L,z) dL`
        :rtype: float
        """

        # Get keyword arguments for the integration
        int_kwargs = {}
        int_kwargs.setdefault('divmax', kwargs.pop('divmax', 20))
        int_kwargs.setdefault('tol', kwargs.pop('epsabs', 1e-3))
        int_kwargs.setdefault('rtol', kwargs.pop('epsrel', 1e-3))

        integral = integrate.romberg(self.evaluate,
                                     lum_range[0],
                                     lum_range[1],
                                     args=(redsh,), **int_kwargs)

        return integral


    def sample(self, lum_range, redsh_range, cosmology, sky_area,
               seed=1234, lum_res=1e-2, redsh_res=1e-2, verbose=1, **kwargs):
        """Sample the luminosity function over a given luminosity and
            redshift range.

        This sampling routine is in large part inspired by
        https://github.com/imcgreer/simqso/blob/master/simqso/lumfun.py
        , lines 219 and following.

        If the integral over the luminosity function does not have an
        analytical implementation, integrals are calculated using
        integrate.romberg, which can take a substantial amount of time.

        :param lum_range: Luminosity range
        :type lum_range: tuple
        :param redsh_range: Redshift range
        :type redsh_range: tuple
        :param cosmology: Cosmology (default = None)
        :type cosmology: astropy.cosmology.Cosmology
        :param sky_area: Area of the sky to be sampled in square degrees
        :type sky_area: float
        :param seed: Random seed for the sampling
        :type seed: int
        :param lum_res: Luminosity resolution (default = 1e-2, equivalent to
            100 bins)
        :type lum_res: float
        :param redsh_res: Redshift resolution (default = 1e-2, equivalent to
            100 bins)
        :type redsh_res: float
        :param verbose: Verbosity
        :type verbose: int
        :return: Source sample luminosities and redshifts
        :rtype: (numpy.ndarray,numpy.ndarray)
        """

        # Get keyword arguments for the integration accuracy
        epsabs = kwargs.pop('epsabs', 1e-3)
        epsrel = kwargs.pop('epsrel', 1e-3)

        # Sky area in steradian
        sky_area_srd = sky_area / 41253. * 4 * np.pi

        # Instantiate differential comoving solid volume element
        dVdzdO = interp_dVdzdO(redsh_range, cosmology)

        # Number of luminosity and redshift ranges for the discrete
        # integrations below.
        n_lum = int(np.diff(lum_range) / lum_res)
        n_redsh = int(np.diff(redsh_range) / redsh_res)

        # Calculate the dN/dz distribution
        redsh_bins = np.linspace(redsh_range[0], redsh_range[1], n_redsh)

        # An array to store the number of sources within a given redshift bin
        redsh_n = np.zeros_like(redsh_bins)

        for idx in range(len(redsh_bins)-1):

            redsh_n[idx+1] = integrate.quad(self.redshift_density,
                                            redsh_bins[idx],
                                            redsh_bins[idx+1],
                                            args=(lum_range, dVdzdO),
                                            epsabs=epsabs,
                                            epsrel=epsrel)[0]

        # The total number of sources of the integration per steradian
        total = np.sum(redsh_n)
        # Interpolation of luminosity function in redshift space
        redsh_func = interp1d(np.cumsum(redsh_n)/total, redsh_bins)
        # The total number of sources as rounded to an integer value across
        # the sky_area specified in the input argument.
        total = np.int(np.round(total * sky_area_srd))
        if verbose > 0:
            print('[INFO] Integration returned {} sources'.format(total))

        # Draw random values from from the total number of sources
        np.random.seed(seed)
        redsh_rand = np.random.random(total)
        lum_rand = np.random.random(total)

        # Sample the redshift values using the interpolation above
        redsh_sample = redsh_func(redsh_rand)
        # Set up the luminosity range
        lum_sample = np.zeros_like(redsh_sample)

        for source in range(total):
            lum_bins = np.linspace(lum_range[0], lum_range[1], n_lum)

            # An array to store the number of sources within a given
            # luminosity bin.
            lum_n = np.zeros_like(lum_bins)

            for idx in range(len(lum_bins)-1):

                lum_n[idx] = self.redshift_density(redsh_sample[source],
                                                     [lum_bins[idx],
                                                     lum_bins[idx+1]],
                                                     dVdzdO)

            # Interpolation of the luminosity function an redshift of the
            # random source over luminosity
            lum_func = interp1d(np.cumsum(lum_n) / np.sum(lum_n), lum_bins)

            # Draw a random luminosity value for the source
            lum_sample[source] = lum_func(lum_rand[source])

        return lum_sample, redsh_sample


class DoublePowerLawLF(LuminosityFunction):
    """ Luminosity function, which takes the functional form of a double
    power law with the luminosity in absolute magnitudes.

    The luminosity function has four main parameters:

    - "phi_star": the overall normalization
    - "lum_star": the break luminosity/magnitude where the power law slopes
      change.
    - "alpha": the first power law slope
    - "beta": the second power law slope

    """

    def __init__(self, parameters, param_functions, lum_type=None, verbose=1):
        """Initialization of the double power law luminosity function class.
        """

        # The main parameters are the parameters which get passed into the
        # functional form of the luminosity function, they can themselves be
        # functions parameters (incl. redshift and luminosity dependence).
        self.main_parameters = ['phi_star', 'lum_star', 'alpha', 'beta']

        # Initialize the parent class
        super(DoublePowerLawLF, self).__init__(parameters, param_functions,
                                               self.main_parameters,
                                               lum_type=lum_type,
                                               verbose=verbose)

    def evaluate(self, lum, redsh, parameters=None):
        """Evaluate the double power law as a function of magnitude ("lum")
        and redshift ("redsh").

        Function to be evaluated: atelier.lumfun.mag_double_power_law()

        :param lum: Luminosity for evaluation
        :type lum: float or numpy.ndarray
        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param parameters: Dictionary of parameters used for this specific
            calculation. This does not replace the parameters with which the
            luminosity function was initialized. (default=None)
        :type parameters: dict(atelier.lumfun.Parameters)
        :return: Luminosity function value
        :rtype: (numpy.ndarray,numpy.ndarray)
        """

        if parameters is None:
            parameters = self.parameters.copy()

        main_parameter_values = self.evaluate_main_parameters(lum, redsh,
                                                        parameters=parameters)

        phi_star = main_parameter_values['phi_star']
        lum_star = main_parameter_values['lum_star']
        alpha = main_parameter_values['alpha']
        beta = main_parameter_values['beta']

        return mag_double_power_law(lum, phi_star, lum_star, alpha, beta)

    def calc_ionizing_emissivity_at_1450A(self, redsh, lum_range, **kwargs):
        """Calculate the ionizing emissivity at rest-frame 1450A,
        :math:`\epsilon_{1450}`, in units of
        erg s^-1 Hz^-1 Mpc^-3.

        This function integrates the luminosity function at redshift "redsh"
        over the luminosity interval "lum_range" to calculate the ionizing
        emissivity at rest-frame 1450A.

        Calling this function is only valid if the luminosity function
        "lum_type" argument is "lum_type"="M1450".

        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param lum_range: Luminosity range
        :type lum_range: tuple
        :return: Ionizing emissivity (erg s^-1 Hz^-1 Mpc^-3)
        :rtype: float
        """

        if self.lum_type != 'M1450':
            raise ValueError('[ERROR] Luminosity function is not defined as a'
                             ' function of M1450. Therefore, this calculating'
                             ' the ionizing emissivity with this function is'
                             ' not valid')

        # Get keyword arguments for the integration
        int_kwargs = {}
        int_kwargs.setdefault('divmax', kwargs.pop('divmax', 20))
        int_kwargs.setdefault('tol', kwargs.pop('epsabs', 1e-3))
        int_kwargs.setdefault('rtol', kwargs.pop('epsrel', 1e-3))

        # Integrate luminosity function times L1450 over luminosity
        integral = integrate.romberg(self._ionizing_emissivity_integrand,
                                     lum_range[0],
                                     lum_range[1],
                                     args=(redsh,),
                                     **int_kwargs)

        return integral

    def _ionizing_emissivity_integrand(self, lum, redsh):
        """Internal function that provides the integrand for the ionizing
        emissivity.

        :param lum: Luminosity for evaluation
        :type lum: float or numpy.ndarray
        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :return: Ionizing emissivity per magnitude (erg s^-1 Hz^-1 Mpc^-3
        M_1450^-1)
        :rtype: float

        """
        # Evaluate parameters
        parameters = self.parameters.copy()
        main_parameter_values = self.evaluate_main_parameters(lum, redsh,
                                                              parameters=parameters)
        # Modify slopes to integrate over Phi L dM
        phi_star = main_parameter_values['phi_star']
        lum_star = main_parameter_values['lum_star']
        alpha = main_parameter_values['alpha']+1
        beta = main_parameter_values['beta']+1

        # Reproducing Ians function (for now)
        c = 4. * np.pi * (10 * units.pc.to(units.cm)) ** 2
        LStar_nu = c * 10 ** (-0.4 * (lum_star + 48.6))

        return mag_double_power_law(lum, phi_star, lum_star, alpha, beta) * LStar_nu



class SinglePowerLawLF(LuminosityFunction):
    """ Luminosity function, which takes the functional form of a single
    power law with the luminosity in absolute magnitudes.

    The luminosity function has three main parameters:

    - "phi_star": the overall normalization
    - "alpha": the first power law slope
    - "lum_ref": the break luminosity/magnitude where the power law slopes
      change.


    """

    def __init__(self, parameters, param_functions, verbose=1):
        """Initialize the single power law luminosity function class.
        """

        self.main_parameters = ['phi_star', 'alpha', 'lum_ref']

        # Initialize the parent class
        super(SinglePowerLawLF, self).__init__(parameters, param_functions,
                                               self.main_parameters,
                                               verbose)

    def evaluate(self, lum, redsh, parameters=None):
        """Evaluate the single power law as a function of magnitude ("lum")
        and redshift ("redsh").

        Function to be evaluated: atelier.lumfun.mag_single_power_law()

        :param lum: Luminosity for evaluation
        :type lum: float or numpy.ndarray
        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param parameters: Dictionary of parameters used for this specific
            calculation. This does not replace the parameters with which the
            luminosity function was initialized. (default=None)
        :type parameters: dict(atelier.lumfun.Parameters)
        :return: Luminosity function value
        :rtype: (numpy.ndarray,numpy.ndarray)
        """

        if self.lum_type != 'M1450':
            raise ValueError('[ERROR] Luminosity function is not defined as a'
                             ' function of M1450. Therefore, this calculating'
                             ' the ionizing emissivity with this function is'
                             ' not valid')

        if parameters is None:
            parameters = self.parameters.copy()

        main_parameter_values = self.evaluate_main_parameters(lum, redsh,
                                                        parameters=parameters)

        phi_star = main_parameter_values['phi_star']
        lum_ref = main_parameter_values['lum_ref']
        alpha = main_parameter_values['alpha']

        return mag_single_power_law(lum, phi_star, lum_ref, alpha)


    def calc_ionizing_emissivity_at_1450A(self, redsh, lum_range, **kwargs):
        """Calculate the ionizing emissivity at rest-frame 1450A,
        :math:`\epsilon_{1450}`, in units of
        erg s^-1 Hz^-1 Mpc^-3.

        This function integrates the luminosity function at redshift "redsh"
        over the luminosity interval "lum_range" to calculate the ionizing
        emissivity at rest-frame 1450A.

        Calling this function is only valid if the luminosity function
        "lum_type" argument is "lum_type"="M1450".

        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param lum_range: Luminosity range
        :type lum_range: tuple
        :return: Ionizing emissivity (erg s^-1 Hz^-1 Mpc^-3)
        :rtype: float
        """

        # Get keyword arguments for the integration
        int_kwargs = {}
        int_kwargs.setdefault('divmax', kwargs.pop('divmax', 20))
        int_kwargs.setdefault('tol', kwargs.pop('epsabs', 1e-3))
        int_kwargs.setdefault('rtol', kwargs.pop('epsrel', 1e-3))

        # Integrate luminosity function times L1450 over luminosity
        integral = integrate.romberg(self._ionizing_emissivity_integrand,
                                     lum_range[0],
                                     lum_range[1],
                                     args=(redsh,),
                                     **int_kwargs)

        return integral

    def _ionizing_emissivity_integrand(self, lum, redsh):
        """Internal function that provides the integrand for the ionizing
        emissivity.

        :param lum: Luminosity for evaluation
        :type lum: float or numpy.ndarray
        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :return: Ionizing emissivity per magnitude (erg s^-1 Hz^-1 Mpc^-3
        M_1450^-1)
        :rtype: float

        """
        # Evaluate parameters
        parameters = self.parameters.copy()
        main_parameter_values = self.evaluate_main_parameters(lum, redsh,
                                                              parameters=parameters)
        # Modify slopes to integrate over Phi L dM
        phi_star = main_parameter_values['phi_star']
        lum_ref = main_parameter_values['lum_ref']
        alpha = main_parameter_values['alpha'] + 1

        # Reproducing Ians function (for now)
        c = 4. * np.pi * (10 * units.pc.to(units.cm)) ** 2
        LStar_nu = c * 10 ** (-0.4 * (lum_ref + 48.6))

        return mag_single_power_law(lum, phi_star, lum_ref, alpha) * LStar_nu


class Hopkins2007QLF(DoublePowerLawLF):
    """Implementation of the bolometric quasar luminosity function of
    Hopkins+2007.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2007ApJ...654..731H/abstract

    The luminosity function is described by Equations 6, 9, 10, 17, 19.
    The values for the best fit model adopted here are found in Table 3 in
    the row named "Full".

    """

    def __init__(self):
        """Initialize the Hopkins+2007 bolometric quasar luminosity function.
        """

        # OLD properties to maybe be incorporated in Luminosity Function model
        # self.name = "Hopkins2007"
        # self.band = "bolometric"
        # self.band_wavelength = None  # in nm
        # self.type = 1  # 0 = magnitudes, 1 = luminosity double power law
        #
        # self.z_min = 0  # lower redshift limit of data for the QLF fit
        # self.z_max = 4.5  # upper redshift limit of data for the QLF fit
        #
        # self.x_max = 18  # log(L_bol)
        # self.x_min = 8  # log(L_bol)
        #
        # self.x = 12  # default magnitude value
        # self.z = 1.0  # default redshift value

        # best fit values Table 7
        log_phi_star = Parameter(-4.825, 'log_phi_star', one_sigma_unc=[0.06,
                                                                        0.06])
        log_lum_star = Parameter(13.036, 'log_lum_star', one_sigma_unc=[
            0.043, 0.043])
        gamma_one = Parameter(0.417, 'gamma_one', one_sigma_unc=[0.055, 0.055])
        gamma_two = Parameter(2.174, 'gamma_two', one_sigma_unc=[0.055, 0.055])

        kl1 = Parameter(0.632, 'kl1', one_sigma_unc=[0.077, 0.077])
        kl2 = Parameter(-11.76, 'kl2', one_sigma_unc=[0.38, 0.38])
        kl3 = Parameter(-14.25, 'kl2', one_sigma_unc=[0.8, 0.8])

        kg1 = Parameter(-0.623, 'kg1', one_sigma_unc=[0.132, 0.132])
        kg2_1 = Parameter(1.46, 'kg2_1', one_sigma_unc=[0.096, 0.096])
        kg2_2 = Parameter(-0.793, 'kg2_2', one_sigma_unc=[0.057, 0.057])

        z_ref = Parameter(2.0, 'z_ref', vary=False)

        parameters = {'log_phi_star': log_phi_star,
                      'log_lum_star': log_lum_star,
                      'gamma_one': gamma_one,
                      'gamma_two': gamma_two,
                      'kl1': kl1,
                      'kl2': kl2,
                      'kl3': kl3,
                      'kg1': kg1,
                      'kg2_1': kg2_1,
                      'kg2_2': kg2_2,
                      'z_ref': z_ref}

        param_functions = {'lum_star':self.lum_star,
                     'phi_star':self.phi_star,
                     'alpha': self.alpha,
                     'beta': self.beta}

        lum_type = 'bolometric'

        super(Hopkins2007QLF, self).__init__(parameters, param_functions,
                                             lum_type=lum_type)

    @staticmethod
    def lum_star(redsh, z_ref, log_lum_star, kl1, kl2, kl3):
        """Calculate the redshift dependent break luminosity (Eq. 9, 10)

        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param z_ref: Reference redshift
        :type z_ref: float
        :param log_lum_star: Logarithmic break magnitude at z_ref
        :type log_lum_star: float
        :param kl1: Function parameter kl1
        :type kl1: float
        :param kl2: Function parameter kl2
        :type kl2: float
        :param kl3: Function parameter kl3
        :type kl3: float
        :return: Redshift dependent break luminosity
        :rtype: float
        """

        # Equation 10
        xi = np.log10((1 + redsh) / (1 + z_ref))

        # Equation 9
        log_lum_star = log_lum_star + kl1 * xi + kl2 * xi ** 2 + kl3 * xi ** 3

        lum_star = pow(10, log_lum_star)

        return lum_star

    @staticmethod
    def phi_star(log_phi_star):
        """ Calculate the break luminosity number density

        :param log_phi_star: Logarithmic break luminosity number density
        :return:
        """
        return pow(10, log_phi_star)

    @staticmethod
    def alpha(redsh, z_ref, gamma_one, kg1):
        """Calculate the redshift dependent luminosity function slope alpha.

        Equations 10, 17

        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param z_ref: Reference redshift
        :type z_ref: float
        :param gamma_one: Luminosity function power law slope at z_ref
        :type gamma_one: float
        :param kg1: Evolutionary parameter
        :type kg1: float
        :return:
        """
        # Equation 10
        xi = np.log10((1 + redsh) / (1 + z_ref))

        # Equation 17
        alpha = gamma_one * pow(10, kg1 * xi)

        return alpha

    @staticmethod
    def beta(redsh, z_ref, gamma_two, kg2_1, kg2_2):
        """Calculate the redshift dependent luminosity function slope beta.

        Equations 10, 19 and text on page 744 (bottom right column)

        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param z_ref: Reference redshift
        :type z_ref: float
        :param gamma_two: Luminosity function power law slope at z_ref
        :type gamma_two: float
        :param kg2_1: Evolutionary parameter
        :type kg2_1: float
        :param kg2_2: Evolutionary parameter
        :type kg2_2: float
        :return:
        """

        # Equation 10
        xi = np.log10((1 + redsh) / (1 + z_ref))

        # Equation 19
        beta = gamma_two * 2 /  (pow(10, kg2_1 * xi) +
                                      ( pow(10,kg2_2 *  xi)))

        # note pg. 744 right column bottom
        if (beta < 1.3) & (redsh > 5.0):
            beta = 1.3

        return beta

    def evaluate(self, lum, redsh, parameters=None):
        """Evaluate the Hopkins+2007 bolometric luminosity function.

        Function to be evaluated: atelier.lumfun.lum_double_power_law()

        :param lum: Luminosity for evaluation
        :type lum: float or numpy.ndarray
        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param parameters: Dictionary of parameters used for this specific
            calculation. This does not replace the parameters with which the
            luminosity function was initialized. (default=None)
        :type parameters: dict(atelier.lumfun.Parameters)
        :return: Luminosity function value
        :rtype: (numpy.ndarray,numpy.ndarray)
        """

        if parameters is None:
            parameters = self.parameters.copy()

        main_parameter_values = self.evaluate_main_parameters(lum, redsh,
                                                        parameters=parameters)

        phi_star = main_parameter_values['phi_star']
        lum_star = main_parameter_values['lum_star']
        alpha = main_parameter_values['alpha']
        beta = main_parameter_values['beta']

        return lum_double_power_law(np.power(10, lum), phi_star, lum_star,
                                              alpha, beta)


class McGreer2018QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    McGreer+2018 at z~5 (z=4.9).

    ADS reference: https://ui.adsabs.harvard.edu/abs/2018AJ....155..131M/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the best maximum likelihood estimate fit from
    the second column in Table 2.


    """

    def __init__(self):
        """ Initialize the McGreer+2018 type-I quasar UV luminosity function.
        """

        # best MLE fit values Table 2 second column
        log_phi_star_z6 = Parameter(-8.97, 'log_phi_star_z6', one_sigma_unc=[
            0.18,
                                                                       0.15])
        lum_star = Parameter(-27.47, 'lum_star', one_sigma_unc=[
            0.26, 0.22])
        alpha = Parameter(-1.97, 'alpha', one_sigma_unc=[0.09, 0.09])
        beta = Parameter(-4.0, 'beta', one_sigma_unc=None)

        k = Parameter(-0.47, 'k')

        z_ref = Parameter(6, 'z_ref')

        parameters = {'log_phi_star_z6': log_phi_star_z6,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta,
                      'k': k,
                      'z_ref': z_ref}

        param_functions = {'phi_star': self.phi_star}

        lum_type = 'M1450'

        super(McGreer2018QLF, self).__init__(parameters, param_functions,
                                             lum_type=lum_type)


    @staticmethod
    def phi_star(redsh, log_phi_star_z6, k, z_ref):
        """Calculate the redshift dependent luminosity function normalization.

        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param log_phi_star_z6: Logarithmic source density at z=6
        :type log_phi_star_z6: float
        :param k: Power law exponent of density evolution
        :type k: float
        :param z_ref: Reference redshift
        :type z_ref: float
        :return:
        """

        log_phi_star = log_phi_star_z6 + k * (redsh-z_ref)

        return pow(10, log_phi_star)


class WangFeige2019SPLQLF(SinglePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Wang+2019 at z~6.7.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2019ApJ...884...30W/abstract

    The luminosity function is parameterized as a single power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the single power law fit described in Section 5.5
    """

    def __init__(self):

        phi_star = Parameter(6.34e-10, 'phi_star', one_sigma_unc=[1.73e-10,
                                                                  1.73e-10])
        alpha = Parameter(-2.35, 'alpha', one_sigma_unc=[0.22, 0.22])

        lum_ref = Parameter(-26, 'lum_ref')


        parameters = {'phi_star': phi_star,
                      'alpha': alpha,
                      'lum_ref': lum_ref}

        param_functions = {}

        lum_type = 'M1450'

        super(WangFeige2019SPLQLF, self).__init__(parameters, param_functions,
                                                  lum_type=lum_type)


class WangFeige2019DPLQLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Wang+2019 at z~6.7.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2019ApJ...884...30W/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the double power law fit described in Section 5.5
    """

    def __init__(self):
        """Initialize the Wang+2019 type-I quasar UV luminosity function.

        """

        phi_star_z6p7 = Parameter(3.17e-9, 'phi_star_z6p7',
                                one_sigma_unc=[0.85e-9, 0.85e-9])
        lum_star = Parameter(-25.2, 'lum_star', one_sigma_unc=None)

        alpha = Parameter(-1.9, 'gamma_one', one_sigma_unc=None)
        beta = Parameter(-2.54, 'gamma_two', one_sigma_unc=[0.29, 0.29])

        k = Parameter(-0.78, 'k')

        z_ref = Parameter(6.7, 'z_ref')

        parameters = {'phi_star_z6p7': phi_star_z6p7,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta,
                      'k': k,
                      'z_ref': z_ref}

        param_functions = {'phi_star': self.phi_star}

        lum_type = 'M1450'

        super(WangFeige2019DPLQLF, self).__init__(parameters, param_functions,
                                                  lum_type=lum_type)


    @staticmethod
    def phi_star(redsh, phi_star_z6p7, k, z_ref):
        """Calculate the redshift dependent luminosity function normalization.

        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param phi_star_z6p7: Logarithmic source density at z=6
        :type phi_star_z6p7: float
        :param k: Power law exponent of density evolution
        :type k: float
        :param z_ref: Reference redshift
        :type z_ref: float
        :return:
        """

        return phi_star_z6p7 * 10**(k * (redsh - z_ref))


class JiangLinhua2016QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Jiang+2016 at z~6.

    ADS reference:

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the double power law fit described in Section 4.5
    """

    def __init__(self):
        """Initialize the Jiang+2016 type-I quasar UV luminosity function.
        """

        phi_star_z6 = Parameter(9.93e-9, 'phi_star_z6', one_sigma_unc=[])

        lum_star = Parameter(-25.2, 'lum_star', one_sigma_unc=[3.8, 1.2])

        alpha = Parameter(-1.9, 'alpha', one_sigma_unc=[0.58, 0.44])

        beta = Parameter(-2.8, 'beta', one_sigma_unc=None)

        k = Parameter(-0.72, 'k')

        z_ref = Parameter(6, 'z_ref')

        parameters = {'phi_star_z6': phi_star_z6,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta,
                      'k': k,
                      'z_ref': z_ref}

        param_functions = {'phi_star': self.phi_star}

        lum_type = 'M1450'

        super(JiangLinhua2016QLF, self).__init__(parameters, param_functions,
                                                 lum_type=lum_type)

    @staticmethod
    def phi_star(redsh, phi_star_z6, k, z_ref):
        """Calculate the redshift dependent luminosity function normalization.

        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param phi_star_z6: Logarithmic source density at z=6
        :type phi_star_z6: float
        :param k: Power law exponent of density evolution
        :type k: float
        :param z_ref: Reference redshift
        :type z_ref: float
        :return:
        """

        return phi_star_z6 * 10**(k * (redsh - z_ref))


class Matsuoka2018QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Matsuoka+2018 at z~6.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2018ApJ...869..150M/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the double power law fit presented in Table 5
    ("standard").
    """

    def __init__(self):
        """Initialize the Matsuoka+2018 type-I quasar UV luminosity function.
        """


        # ML fit parameters from the "standard" model in Table 5
        phi_star_z6 = Parameter(10.9e-9, 'phi_star_z6', one_sigma_unc=[6.8e-9,
                                                                       10e-9])

        lum_star = Parameter(-24.9, 'lum_star', one_sigma_unc=[0.9, 0.75])

        alpha = Parameter(-1.23, 'alpha', one_sigma_unc=[0.34, 0.44])

        beta = Parameter(-2.73, 'beta', one_sigma_unc=[0.31, 0.23])

        k = Parameter(-0.7, 'k')

        z_ref = Parameter(6, 'z_ref')

        parameters = {'phi_star_z6': phi_star_z6,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta,
                      'k': k,
                      'z_ref': z_ref}

        param_functions = {'phi_star': self.phi_star}

        super(Matsuoka2018QLF, self).__init__(parameters, param_functions)

    @staticmethod
    def lum_star(mag_star):

        return mag_star

    @staticmethod
    def phi_star(redsh, phi_star_z6, k, z_ref):
        """Calculate the redshift dependent luminosity function normalization.

        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param phi_star_z6: Logarithmic source density at z=6
        :type phi_star_z6: float
        :param k: Power law exponent of density evolution
        :type k: float
        :param z_ref: Reference redshift
        :type z_ref: float
        :return:
        """

        return phi_star_z6 * 10**(k * (redsh - z_ref))




class BinnedLuminosityFunction(object):

    def __init__(self, lum=None, lum_type=None, lum_unit=None,
                 phi=None, log_phi=None, phi_unit=None,
                 sigma_phi=None, sigma_log_phi=None, cosmology=None,
                 redshift=None, redshift_range=None):

        if lum is not None:
            self.lum = lum

            if lum_type is None:
                raise ValueError('[ERROR] Luminosity type not specified!')
            else:
                self.lum_type = lum_type
            if lum_unit is None:
                raise ValueError('[ERROR] Luminosity unit not specified!')
            else:
                self.lum_unit = lum_unit

        if phi is not None or log_phi is not None:

            if phi_unit is None:
                raise ValueError('[ERROR] Luminosity function unit not '
                                 'specified!')

            if phi is not None and log_phi is None:
                self.phi = phi
                self.log_phi = self._get_logphi_from_phi()
            elif log_phi is not None and phi is None:
                self.log_phi = log_phi
                self.phi = self._get_phi()

            elif log_phi is not None and phi is not None:
                self.phi = phi
                self.log_phi = log_phi

        if sigma_phi is not None or sigma_log_phi is not None:

            if sigma_phi is not None and sigma_log_phi is None:

                self.sigma_phi = sigma_phi
                self.sigma_log_phi = self._get_sigma_logphi_from_sigma_phi()

            elif sigma_log_phi is not None and sigma_phi is None:

                self.sigma_log_phi = sigma_log_phi
                self.sigma_phi = self._get_sigma_phi_from_sigma_logphi()

            else:
                self.sigma_phi = sigma_phi
                self.sigma_log_phi = sigma_log_phi

        if cosmology is None:
            raise ValueError('[ERROR] No cosmology specified!')
        else:
            self.cosmology = cosmology

        self.redshift = redshift
        self.redshift_range =redshift_range


    def _get_logphi_from_phi(self):

        return np.log10(self.phi)

    def _get_phi_from_logphi(self):

        return np.pow(10, self.log_phi)

    def _get_sigma_logphi_from_sigma_phi(self):

        pass

    def _get_sigma_phi_from_sigma_logphi(self):
        pass



mcgreer2013_str82 = \
      {'lum': np.array([-27.0, -26.45, -25.9, -25.35, -24.8, -24.25]),
       'log_phi': np.array([-8.4, -7.84, -7.9, -7.53, -7.36, -7.14]),
       'sigma_phi': np.array([2.81, 6.97, 5.92, 10.23, 11.51, 19.9])*1e-9,
       'phi_unit': units.Mpc ** -3 * units.mag ** -1,
       'lum_type': 'M1450',
       'lum_unit': units.mag,
       'cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
       'redshift': 4.9,
       'redshift_range': [4.7, 5.1]
       }

mcgreer2013_dr7 = \
       {'lum': np.array([-28.05, -27.55, -27.05, -26.55, -26.05]),
        'log_phi': np.array([-9.45, -9.24, -8.51, -8.20, -7.9]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([0.21, 0.26, 0.58, 0.91, 1.89])*1e-9,
        'cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
        'redshift': 4.9,
        'redshift_range': [4.7, 5.1]
        }

mcgreer2018_main = \
       {'lum': np.array([-28.55, -28.05, -27.55, -27.05, -26.55, -26.05]),
        'log_phi': np.array([-9.90, -9.70, -8.89, -8.41, -8.10, -8.03]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([0.12, 0.14, 0.37, 0.72, 1.08, 1.74])*1e-9,
        'cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
        'redshift': 5,
        'redshift_range': [4.7, 5.3]
        }

mcgreer2018_s82 = \
       {'lum': np.array([-27.00, -26.45, -25.90, -25.35, -24.80, -24.25]),
        'log_phi': np.array([-8.06, -7.75, -8.23, -7.47, -7.24, -7.22]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([5.57, 6.97, 3.38, 10.39, 13.12, 21.91])*1e-9,
        'cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
        'redshift': 5,
        'redshift_range': [4.7, 5.3]
        }

mcgreer2018_cfhtls_wide = \
       {'lum': np.array([-26.35, -25.25, -24.35, -23.65, -22.90]),
        'log_phi': np.array([-8.12, -7.56, -7.25, -7.32, -7.32]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([4.34, 12.70, 18.05, 23.77, 28.24])*1e-9,
        'cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
        'redshift': 5,
        'redshift_range': [4.7, 5.3]
        }

matsuoka2018 =  \
       {'lum': np.array([-22, -22.75, -23.25, -23.75, -24.25, -24.75, -25.25,
                         -25.75, -26.25, -26.75, -27.5, -29]),
        'phi': np.array([16.2, 23.0, 10.9, 8.3, 6.6, 7.0, 4.6, 1.33, 0.9, 0.58,
                         0.242, 0.0079])*1e-9,
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([16.2, 8.1, 3.6, 2.6, 2.0, 1.7, 1.2,
                                   0.6, 0.32, 0.17, 0.061, 0.0079])*1e-9,
        'cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 6.1,
        'redshift_range': [5.7, 6.5]
        }


# M0 M1 Z0 Z1 M1450_Mean M1450_err z_Mean N N_corr Phi Phi_err
# -27.6 -26.9 6.45 7.05 -27.1849570034 0.198454192575 6.68253 4 6.57997180974 1.49806504375e-10 4.76706351322e-11
# -26.9 -26.2 6.45 7.05 -26.4394997883 0.141104303161 6.70167 9 15.7066648272 3.5759432125e-10 7.57916215887e-11
# -26.2 -25.5 6.45 7.05 -25.8306028526 0.265699553273 6.65747 4 36.3303873458 8.27135508817e-10 3.19637408061e-10

wangfeige2019 =  \
       {'lum': np.array([-27.1849570034, -26.4394997883, -25.8306028526]),
        'sigma_lum': np.array([0.198454192575, 0.141104303161, 0.265699553273]),

        'lum_bins': np.array([[-27.6, -26.9], [-26.9, -26.2], [-26.2, -25.5]]),
        'phi': np.array([1.49806504375e-10, 3.5759432125e-10,
                         8.27135508817e-10]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([4.76706351322e-11, 7.57916215887e-11,
                               3.19637408061e-10]),
        'cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift_mean': np.array([6.68253, 6.70167, 6.65747]),
        'redshift': 6.7,
        'redshift_range': [6.45, 7.05]
        }


# Add Jiang2016
# Add Willott 2010