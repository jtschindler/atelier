#!/usr/bin/env python

import numpy as np
from astropy import units
from scipy import integrate
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

import matplotlib.pyplot as plt

from atelier import plot_defaults
from atelier import tol_colors

# Basic functionality needed for this class
def interp_dVdzdO(redsh_range, cosmo):
    """Interpolate the differential comoving solid volume element
    :math:`(dV/dz){d\Omega}` over the specified redshift range
    zrange = :math:`(z_1,z_2)`.

    This interpolation speeds up volume (redshift, solid angle) integrations
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


def richards_single_power_law(mag, phi_star, mag_ref, alpha):
    """

    :param mag:
    :param phi_star:
    :param mag_ref:
    :param alpha:
    :return:
    """
    return phi_star * 10**(alpha*(mag - mag_ref))


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
                 lum_type=None, cosmology=None, ref_cosmology=None,
                 ref_redsh=None, verbose=1):
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

        self.cosmology = cosmology
        self.ref_cosmology = ref_cosmology
        self.ref_redsh = ref_redsh

        if self.cosmology is not None and self.ref_cosmology is not \
                None:
            print('[INFO] Cosmology and reference cosmology are not the '
                  'sample. Cosmological conversions will be applied.')


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

    def update(self):
        """ Update the lumfun class parameters after a manual input.

        :return:
        """

        self.free_parameters = self.get_free_parameters()
        self.free_parameter_names = list(self.free_parameters.keys())



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

        return self.evaluate(lum, redsh) * dVdzdO(redsh) * selfun.evaluate(lum,
                                                                  redsh)

    def integrate_over_lum_redsh(self, lum_range, redsh_range, dVdzdO=None,
                                 selfun=None, cosmology=None, **kwargs):
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
        :param kwargs:
        :return: :math:`N = \int\int\Phi(L,z) (dV/(dz d\Omega)) dL dz`
        :rtype: float
        """

        # Sort input lum/redsh ranges
        lum_range = np.sort(np.array(lum_range))
        redsh_range = np.sort(np.array(redsh_range))

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

    def integrate_over_lum_redsh_simpson(self, lum_range, redsh_range,
                                       dVdzdO=None, selfun=None,
                                 cosmology=None, initial_lum_bin_width=0.1,
                                         initial_redsh_bin_width=0.05,
                                         minimum_probability=1e-3,
                                         **kwargs):
        """Calculate the number of sources described by the luminosity function
        over a luminosity and redshift interval in units of per steradian.

        The integration is done on a grid using the Simpson rule.

        This allows the selection function to be precalculated on the grid
        values for speed up of the integration process.

        This code is in large part adopted from
        https://github.com/imcgreer/simqso/blob/master/simqso/lumfun.py
        lines 591 and following.

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
        :param kwargs:
        :return: :math:`N = \int\int\Phi(L,z) (dV/(dz d\Omega)) dL dz`
        :rtype: float
        """

        # Set up the interpolated differential comoving solid volume element
        if dVdzdO is None and cosmology is not None:
            dVdzdO = interp_dVdzdO(redsh_range, cosmology)
        elif dVdzdO is None and cosmology is None:
            raise ValueError(
                '[ERROR] Either a cosmology or dVdzdO have to be supplied.')

        # Sort input lum/redsh ranges
        lum_range = np.sort(np.array(lum_range))
        redsh_range = np.sort(np.array(redsh_range))

        # Setting up the integration grid
        num_lum_bins = int(np.diff(lum_range) / initial_lum_bin_width) + 1
        num_redsh_bins = int(np.diff(redsh_range) / initial_redsh_bin_width) + 1

        lum_edges = np.linspace(lum_range[0], lum_range[1], num_lum_bins)
        redsh_edges = np.linspace(redsh_range[0], redsh_range[1],
                                  num_redsh_bins)

        lum_bin_width = np.diff(lum_edges)[0]
        redsh_bin_width = np.diff(redsh_edges)[0]
        diffvol_grid = dVdzdO(redsh_edges)

        # Generate grid points
        lum_points, redsh_points = np.meshgrid(lum_edges, redsh_edges,
                                           indexing='ij')

        # Calculate selection function grid
        if selfun is not None:
            if selfun.simps_grid is None:
                selfun_grid = selfun.evaluate(lum_points, redsh_points)
                selfun.simps_grid = selfun_grid
            else:
                selfun_grid = selfun.simps_grid

            # selection_mask = selfun_grid > minimum_probability

        # Calculate the luminosity function grid
        lumfun_grid = self.evaluate(lum_points, redsh_points)

        # Calculate the double integral via the Simpson rule
        if selfun is not None:
            inner_integral = integrate.simps(lumfun_grid * selfun_grid *
                                             diffvol_grid,
                                             dx=redsh_bin_width)
        else:
            inner_integral = integrate.simps(lumfun_grid *
                                             diffvol_grid,
                                             dx=redsh_bin_width)

        outer_integral = integrate.simps(inner_integral, dx=lum_bin_width)

        return outer_integral

    # TODO: New, needs to be tested

    def _luminosity_density_integrand(self, lum, redsh):
        """Internal function providing the integrand for the luminosity density
        integration.

        :param lum: float or numpy.ndarray
        :type lum: Luminosity for evaluation
        :param redsh: float or numpy.ndarray
        :type redsh: Redshift for evaluation
        :param dVdzdO: Differential comoving solid volume element
        :type dVdzdO: function
        :return: Returns :math:`\Phi(L,z)\times (dV/dz){d\Omega}`
        :rtype: float or numpy.ndarray
        """

        return self.evaluate(lum, redsh) * 10**lum

    def integrate_to_luminosity_density(self, lum_range, redsh,
                                        **kwargs):
        """

        :param lum_range:
        :param redsh:
        :param dVdzdO:
        :return: :math:`\int \Phi(L,z) L (dV/dz){d\Omega} dL`
        """

        # Get keyword arguments for the integration
        int_kwargs = {}
        int_kwargs.setdefault('epqsabs', kwargs.pop('epsabs', 1e-3))
        int_kwargs.setdefault('epsrel', kwargs.pop('epsrel', 1e-3))

        integrand = self._luminosity_density_integrand

        # lum_den = integrate.romberg(integrand, *lum_range, args=(redsh,),
        #                             **int_kwargs)

        lum_den = integrate.quad(integrand, *lum_range, args=(redsh,),
                                    **int_kwargs)[0]

        return lum_den

    def integrate_over_lum_redsh_appmag_limit(self, lum_range, redsh_range,
                                              appmag_limit, kcorrection,
                                       dVdzdO=None, selfun=None,
                                 cosmology=None, initial_lum_bin_width=0.1,
                                         initial_redsh_bin_width=0.05,
                                         minimum_probability=1e-3,
                                         **kwargs):
        """

        :param lum_range:
        :param redsh_range:
        :param appmag_limit:
        :param kcorrection:
        :param dVdzdO:
        :param selfun:
        :param cosmology:
        :param initial_lum_bin_width:
        :param initial_redsh_bin_width:
        :param minimum_probability:
        :param kwargs:
        :return:
        """

        # Set up the interpolated differential comoving solid volume element
        if dVdzdO is None and cosmology is not None:
            dVdzdO = interp_dVdzdO(redsh_range, cosmology)
        elif dVdzdO is None and cosmology is None:
            raise ValueError(
                '[ERROR] Either a cosmology or dVdzdO have to be supplied.')

        # Sort input lum/redsh ranges
        lum_range = np.sort(np.array(lum_range))
        redsh_range = np.sort(np.array(redsh_range))

        # Setting up the integration grid
        num_lum_bins = int(np.diff(lum_range) / initial_lum_bin_width) + 1
        num_redsh_bins = int(np.diff(redsh_range) / initial_redsh_bin_width) + 1

        lum_edges = np.linspace(lum_range[0], lum_range[1], num_lum_bins)
        redsh_edges = np.linspace(redsh_range[0], redsh_range[1],
                                  num_redsh_bins)

        lum_bin_width = np.diff(lum_edges)[0]
        redsh_bin_width = np.diff(redsh_edges)[0]
        diffvol_grid = dVdzdO(redsh_edges)

        # Generate grid points
        lum_points, redsh_points = np.meshgrid(lum_edges, redsh_edges,
                                               indexing='ij')

        # Calculate selection function grid
        if selfun is not None:
            if selfun.simps_grid is None:
                selfun_grid = selfun.evaluate(lum_points, redsh_points)
                selfun.simps_grid = selfun_grid
            else:
                selfun_grid = selfun.simps_grid



        # Mask by apparent magnitude limit
        lum_lim = lambda redsh: np.clip(kcorrection.m2M(appmag_limit, redsh),
                                        lum_range[0], lum_range[1])[0]

        # Create mask for bins outside apparent magnitude limit
        m = lum_points > lum_lim(redsh_points)

        # Calculate the luminosity function grid
        lumfun_grid = self.evaluate(lum_points, redsh_points)

        # Set the lum function to 0, where the mask is false
        lumfun_grid[m] = 0

        # Calculate the double integral via the Simpson rule
        if selfun is not None:

            inner_integral = integrate.simps(lumfun_grid * selfun_grid *
                                             diffvol_grid,
                                             dx=redsh_bin_width)
        else:
            inner_integral = integrate.simps(lumfun_grid *
                                             diffvol_grid,
                                             dx=redsh_bin_width)

        outer_integral = integrate.simps(inner_integral, dx=lum_bin_width)

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
        # int_kwargs.setdefault('divmax', kwargs.pop('divmax', 20))
        # int_kwargs.setdefault('tol', kwargs.pop('epsabs', 1e-3))
        # int_kwargs.setdefault('rtol', kwargs.pop('epsrel', 1e-3))
        int_kwargs.setdefault('epsabs', kwargs.pop('epsabs', 1.49e-08))
        int_kwargs.setdefault('epsrel', kwargs.pop('epsrel', 1.49e-08))

        # integral = integrate.romberg(self._redshift_density_integrand,
        #                              lum_range[0],
        #                              lum_range[1],
        #                              args=(redsh, dVdzdO), **int_kwargs)

        integral = integrate.quad(self._redshift_density_integrand,
                                     lum_range[0],
                                     lum_range[1],
                                     args=(redsh, dVdzdO), **int_kwargs)[0]

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
        # int_kwargs.setdefault('divmax', kwargs.pop('divmax', 20))
        # int_kwargs.setdefault('tol', kwargs.pop('epsabs', 1e-3))
        # int_kwargs.setdefault('rtol', kwargs.pop('epsrel', 1e-3))
        int_kwargs.setdefault('epsabs', kwargs.pop('epsabs', 1.49e-08))
        int_kwargs.setdefault('epsrel', kwargs.pop('epsrel', 1.49e-08))

        integral = integrate.quad(self.evaluate,
                                     lum_range[0],
                                     lum_range[1],
                                     args=(redsh,), **int_kwargs)[0]

        return integral


    def sample(self, lum_range, redsh_range, cosmology, sky_area,
               seed=1234, lum_res=1e-2, redsh_res=1e-2, verbose=1, **kwargs):
        """Sample the luminosity function over a given luminosity and
            redshift range.

        This sampling routine is in large part adopted from
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

    def __init__(self, parameters, param_functions, lum_type=None,
                 cosmology=None, ref_cosmology=None, ref_redsh=None, verbose=1):
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
                                               cosmology=cosmology,
                                               ref_cosmology=ref_cosmology,
                                               ref_redsh = ref_redsh,
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

        # TODO: Try to precalculate factors in init for better performance
        if self.cosmology is not None and self.ref_cosmology is not \
                None:


            distmod_ref = self.ref_cosmology.distmod(self.ref_redsh)
            distmod_cos = self.cosmology.distmod(self.ref_redsh)

            # Convert luminosity according to new cosmology
            if self.lum_type in ['M1450']:
                self.cosm_lum_conv = distmod_ref.value - distmod_cos.value
            else:
                raise NotImplementedError(
                    '[ERROR] Conversions for luminosity '
                    'type {} are not implemented.'.format(
                        self.lum_type))

            self.cosm_density_conv = self.ref_cosmology.h ** 3 / \
                                     self.cosmology.h ** 3

            lum_star = lum_star + self.cosm_lum_conv
            phi_star = phi_star * self.cosm_density_conv

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
        # int_kwargs.setdefault('divmax', kwargs.pop('divmax', 20))
        # int_kwargs.setdefault('tol', kwargs.pop('epsabs', 1e-3))
        # int_kwargs.setdefault('rtol', kwargs.pop('epsrel', 1e-3))
        int_kwargs.setdefault('epsabs', kwargs.pop('epsabs', 1.49e-08))
        int_kwargs.setdefault('epsrel', kwargs.pop('epsrel', 1.49e-08))

        # Integrate luminosity function times L1450 over luminosity
        # integral = integrate.romberg(self._ionizing_emissivity_integrand,
        #                              lum_range[0],
        #                              lum_range[1],
        #                              args=(redsh,),
        #                              **int_kwargs)

        integral = integrate.quad(self._ionizing_emissivity_integrand,
                                     lum_range[0],
                                     lum_range[1],
                                     args=(redsh,),
                                     **int_kwargs)[0]

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

        # Convert to different cosmology
        # TODO: Move to integral function for better performance!
        if self.cosmology is not None and self.ref_cosmology is not \
                None:

            distmod_ref = self.ref_cosmology.distmod(self.ref_redsh)
            distmod_cos = self.cosmology.distmod(self.ref_redsh)

            # Convert luminosity according to new cosmology
            if self.lum_type in ['M1450']:
                self.cosm_lum_conv = distmod_ref.value - distmod_cos.value
            else:
                raise NotImplementedError(
                    '[ERROR] Conversions for luminosity '
                    'type {} are not implemented.'.format(
                        self.lum_type))

            self.cosm_density_conv = self.ref_cosmology.h ** 3 / \
                                     self.cosmology.h ** 3

            lum_star = lum_star + self.cosm_lum_conv
            phi_star = phi_star * self.cosm_density_conv

        # Reproducing Ian's function (for now)
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

    def __init__(self, parameters, param_functions, lum_type=None,
                 ref_cosmology=None, ref_redsh=None, cosmology=None,
        verbose=1):
        """Initialize the single power law luminosity function class.
        """

        self.main_parameters = ['phi_star', 'alpha', 'lum_ref']

        # Initialize the parent class
        super(SinglePowerLawLF, self).__init__(parameters, param_functions,
                                               self.main_parameters,
                                               lum_type=lum_type,
                                               ref_cosmology=ref_cosmology,
                                               ref_redsh=ref_redsh,
                                               cosmology=cosmology,
                                               verbose=verbose)

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

        # TODO: Try to precalculate factors in init for better performance
        if self.cosmology is not None and self.ref_cosmology is not \
                None:


            distmod_ref = self.ref_cosmology.distmod(self.ref_redsh)
            distmod_cos = self.cosmology.distmod(self.ref_redsh)

            # Convert luminosity according to new cosmology
            if self.lum_type in ['M1450']:
                self.cosm_lum_conv = distmod_ref.value - distmod_cos.value
            else:
                raise NotImplementedError(
                    '[ERROR] Conversions for luminosity '
                    'type {} are not implemented.'.format(
                        self.lum_type))

            self.cosm_density_conv = self.ref_cosmology.h ** 3 / \
                                     self.cosmology.h ** 3

            lum_ref = lum_ref + self.cosm_lum_conv
            phi_star = phi_star * self.cosm_density_conv

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
        # int_kwargs.setdefault('divmax', kwargs.pop('divmax', 20))
        # int_kwargs.setdefault('tol', kwargs.pop('epsabs', 1e-3))
        # int_kwargs.setdefault('rtol', kwargs.pop('epsrel', 1e-3))
        int_kwargs.setdefault('epsabs', kwargs.pop('epsabs', 1.49e-08))
        int_kwargs.setdefault('epsrel', kwargs.pop('epsrel', 1.49e-08))

        # Integrate luminosity function times L1450 over luminosity
        # integral = integrate.romberg(self._ionizing_emissivity_integrand,
        #                              lum_range[0],
        #                              lum_range[1],
        #                              args=(redsh,),
        #                              **int_kwargs)

        integral = integrate.quad(self._ionizing_emissivity_integrand,
                                     lum_range[0],
                                     lum_range[1],
                                     args=(redsh,),
                                     **int_kwargs)[0]

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

        # Convert to different cosmology
        # TODO: Move to integral function for better performance!
        if self.cosmology is not None and self.ref_cosmology is not \
                None:

            distmod_ref = self.ref_cosmology.distmod(self.ref_redsh)
            distmod_cos = self.cosmology.distmod(self.ref_redsh)

            # Convert luminosity according to new cosmology
            if self.lum_type in ['M1450']:
                self.cosm_lum_conv = distmod_ref.value - distmod_cos.value
            else:
                raise NotImplementedError(
                    '[ERROR] Conversions for luminosity '
                    'type {} are not implemented.'.format(
                        self.lum_type))

            self.cosm_density_conv = self.ref_cosmology.h ** 3 / \
                                     self.cosmology.h ** 3

            lum_ref = lum_ref + self.cosm_lum_conv
            phi_star = phi_star * self.cosm_density_conv


        # Reproducing Ians function (for now)
        c = 4. * np.pi * (10 * units.pc.to(units.cm)) ** 2
        LStar_nu = c * 10 ** (-0.4 * (lum_ref + 48.6))

        return mag_single_power_law(lum, phi_star, lum_ref, alpha) * LStar_nu


class ShenXuejian2020QLF(DoublePowerLawLF):
    """
    Shen+2020 bolometric quasar luminosity function; global fit B
    """


    def __init__(self):
        """

        """


        # Parameters

        # alpha
        a0 = Parameter(0.3653, 'a0',
                       # one_sigma_unc=[0.06,0.06]
                       )
        a1 = Parameter(-0.6006, 'a1',
                       # one_sigma_unc=[0.06,0.06]
                       )
        # a2 = Parameter(0, 'a2',
        #                # one_sigma_unc=[0.06,0.06]
        #                )

        # beta
        b0 = Parameter(2.4709, 'b0',
                       # one_sigma_unc=[0.06,0.06]
                       )
        b1 = Parameter(-0.9963, 'b1',
                       # one_sigma_unc=[0.06,0.06]
                       )
        b2 = Parameter(1.0716, 'b2',
                       # one_sigma_unc=[0.06,0.06]
                       )

        # log L_star
        c0 = Parameter(12.9656, 'c0',
                       # one_sigma_unc=[0.06,0.06]
                       )
        c1 = Parameter(-0.5758, 'c1',
                       # one_sigma_unc=[0.06,0.06]
                       )
        c2 = Parameter(0.4698, 'c2',
                       # one_sigma_unc=[0.06,0.06]
                       )

        # log phi_star
        d0 = Parameter(-3.6276, 'd0',
                       # one_sigma_unc=[0.06,0.06]
                       )
        d1 = Parameter(-0.3444, 'd1',
                       # one_sigma_unc=[0.06,0.06]
                       )

        z_ref = Parameter(2.0, 'z_ref')

        parameters = {'a0':a0,
                      'a1':a1,
                      # 'a2':a2,
                      'b0':b0,
                      'b1':b1,
                      'b2':b2,
                      'c0':c0,
                      'c1':c1,
                      'c2':c2,
                      'd0':d0,
                      'd1':d1,
                      'z_ref':z_ref}

        param_functions = {'alpha': self.alpha,
                           'beta': self.beta,
                           'lum_star': self.lum_star,
                           'phi_star': self.phi_star
                           }

        lum_type = 'bolometric'

        super(ShenXuejian2020QLF, self).__init__(parameters, param_functions,
                                             lum_type=lum_type)

    @staticmethod
    def alpha(redsh, a0, a1, z_ref):
        """

        :param redsh:
        :param a0:
        :param a1:
        :param z_ref:
        :return:
        """

        zterm = (1. + redsh) / (1. + z_ref)

        # T0 = 1
        # T1 = (1+redsh)
        # T2 = 2 * (1+redsh)**2 -1
        #
        # gamma_1 = a0*T0 + a1*T1 + a2*T2


        return a0 * zterm**a1

    @staticmethod
    def beta(redsh, b0, b1, b2, z_ref):
        """

        :param redsh:
        :param b0:
        :param b1:
        :param b2:
        :param z_ref:
        :return:
        """

        zterm = (1. + redsh) / (1. + z_ref)

        return 2 * b0 / (zterm ** b1 + zterm ** b2)

    @staticmethod
    def lum_star(redsh, c0, c1, c2, z_ref):
        """

        :param redsh:
        :param c0:
        :param c1:
        :param c2:
        :param z_ref:
        :return:
        """

        zterm = (1. + redsh) / (1. + z_ref)

        log_lum_star = 2 * c0 / (zterm ** c1 + zterm ** c2)

        return 10**log_lum_star # in L_sun

    @staticmethod
    def phi_star(redsh, d0, d1):

        T0 = 1
        T1 = (1+redsh)

        log_phi_star = d0*T0 + d1*T1

        return 10**log_phi_star


    def evaluate(self, lum, redsh, parameters=None):
        """Evaluate the Shen+2020 bolometric luminosity function.

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

        param_functions = {'lum_star': self.lum_star,
                           'phi_star': self.phi_star,
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

    def __init__(self, cosmology=None):
        """ Initialize the McGreer+2018 type-I quasar UV luminosity function.
        """

        # best MLE fit values Table 2 second column
        log_phi_star_z6 = Parameter(-8.97, 'log_phi_star_z6',
                                    one_sigma_unc=[0.18, 0.15])
        lum_star = Parameter(-27.47, 'lum_star',
                             one_sigma_unc=[0.26, 0.22])
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

        # Hinshaw+2013
        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.272)
        ref_redsh = 5.0

        super(McGreer2018QLF, self).__init__(parameters, param_functions,
                                             lum_type=lum_type,
                                             cosmology=cosmology,
                                             ref_cosmology=ref_cosmology,
                                             ref_redsh=ref_redsh)


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


class Willott2010QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    McGreer+2018 at z~5 (z=4.9).

    ADS reference: https://ui.adsabs.harvard.edu/abs/2018AJ....155..131M/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the best maximum likelihood estimate fit from
    the second column in Table 2.


    """

    def __init__(self, cosmology=None):
        """ Initialize the McGreer+2018 type-I quasar UV luminosity function.
        """

        # best MLE fit values Table 2 second column
        phi_star_z6 = Parameter(1.14e-8, 'phi_star_z6')
        lum_star = Parameter(-25.13, 'lum_star',)
        alpha = Parameter(-1.5, 'alpha')
        beta = Parameter(-2.81, 'beta', one_sigma_unc=None)

        k = Parameter(-0.47, 'k')

        z_ref = Parameter(6, 'z_ref')

        parameters = {'phi_star_z6': phi_star_z6,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta,
                      'k': k,
                      'z_ref': z_ref}

        param_functions = {'phi_star': self.phi_star}

        lum_type = 'M1450'

        # Komatsu+2009
        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.28)
        ref_redsh = 6.0

        super(Willott2010QLF, self).__init__(parameters, param_functions,
                                             lum_type=lum_type,
                                             cosmology=cosmology,
                                             ref_cosmology=ref_cosmology,
                                             ref_redsh=ref_redsh)


    @staticmethod
    def phi_star(redsh, phi_star_z6, k, z_ref):
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

        log_phi_star = phi_star_z6 * 10**(k * (redsh-z_ref))

        return pow(10, log_phi_star)


class WangFeige2019SPLQLF(SinglePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Wang+2019 at z~6.7.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2019ApJ...884...30W/abstract

    The luminosity function is parameterized as a single power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the single power law fit described in Section 5.5
    """

    def __init__(self, cosmology=None):

        phi_star = Parameter(6.34e-10, 'phi_star', one_sigma_unc=[1.73e-10,
                                                                  1.73e-10])
        alpha = Parameter(-2.35, 'alpha', one_sigma_unc=[0.22, 0.22])

        lum_ref = Parameter(-26, 'lum_ref')


        parameters = {'phi_star': phi_star,
                      'alpha': alpha,
                      'lum_ref': lum_ref}

        param_functions = {}

        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 6.7

        super(WangFeige2019SPLQLF, self).__init__(parameters, param_functions,
                                                  lum_type=lum_type,
                                                  ref_cosmology=ref_cosmology,
                                                  cosmology = cosmology,
                                                  ref_redsh=ref_redsh)


class WangFeige2019DPLQLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Wang+2019 at z~6.7.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2019ApJ...884...30W/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the double power law fit described in Section 5.5
    """

    def __init__(self, cosmology=None):
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

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 6.7

        super(WangFeige2019DPLQLF, self).__init__(parameters, param_functions,
                                                  lum_type=lum_type,
                                                  ref_cosmology=ref_cosmology,
                                                  ref_redsh=ref_redsh,
                                                  cosmology=cosmology)


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

    def __init__(self, cosmology=None):
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

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 6.05

        super(JiangLinhua2016QLF, self).__init__(parameters, param_functions,
                                                 lum_type=lum_type,
                                                 ref_cosmology=ref_cosmology,
                                                 ref_redsh=ref_redsh,
                                                 cosmology=cosmology)

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


class Akiyama2018QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Akiyama+2018 at z~4.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2018ApJ...869..150M/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the double power law fit presented in Table 5
    ("Maximum Likelihood").
    """

    def __init__(self, cosmology=None):
        """Initialize the Akiyama+2018 type-I quasar UV luminosity function.
        """

        # ML fit parameters from the "standard" model in Table 5
        phi_star = Parameter(2.66e-7, 'phi_star',
                             one_sigma_unc=[0.05e-7, 0.05e-7])

        lum_star = Parameter(-25.37, 'lum_star', one_sigma_unc=[0.13, 0.13])

        alpha = Parameter(-1.30, 'alpha', one_sigma_unc=[0.05, 0.05])

        beta = Parameter(-3.11, 'beta', one_sigma_unc=[0.07, 0.07])


        parameters = {'phi_star': phi_star,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta}

        param_functions = {}

        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 4.0

        super(Akiyama2018QLF, self).__init__(parameters, param_functions,
                                              lum_type=lum_type,
                                              ref_cosmology=ref_cosmology,
                                              ref_redsh=ref_redsh,
                                              cosmology=cosmology)

class Matsuoka2018QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Matsuoka+2018 at z~6.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2018ApJ...869..150M/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the double power law fit presented in Table 5
    ("standard").
    """

    def __init__(self, cosmology=None):
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

        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 6.1

        super(Matsuoka2018QLF, self).__init__(parameters, param_functions,
                                                 lum_type=lum_type,
                                                 ref_cosmology=ref_cosmology,
                                                 ref_redsh=ref_redsh,
                                                 cosmology=cosmology)

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



class Schindler2022QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Schindler+2022 at z~6.

    ADS reference: TBD

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the double power law fit presented in Table XXX
    (first row).
    """

    def __init__(self, cosmology=None):
        """Initialize the Matsuoka+2018 type-I quasar UV luminosity function.
        """


        # ML fit parameters from the "standard" model in Table 5
        log_phi_star_z6 = Parameter(-8.62, 'log_phi_star_z6')

        lum_star = Parameter(-26.37, 'lum_star', one_sigma_unc=[0.9, 0.75])

        alpha = Parameter(-1.60, 'alpha', one_sigma_unc=[0.34, 0.44])

        beta = Parameter(-4.13, 'beta', one_sigma_unc=[0.31, 0.23])

        k = Parameter(-0.7, 'k')

        z_ref = Parameter(6, 'z_ref')

        parameters = {'log_phi_star_z6': log_phi_star_z6,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta,
                      'k': k,
                      'z_ref': z_ref}

        param_functions = {'phi_star': self.phi_star}

        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 6.0

        super(Matsuoka2018QLF, self).__init__(parameters, param_functions,
                                                 lum_type=lum_type,
                                                 ref_cosmology=ref_cosmology,
                                                 ref_redsh=ref_redsh,
                                                 cosmology=cosmology)

    @staticmethod
    def phi_star(redsh, log_phi_star_z6, k, z_ref):
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

        phi_star_z6 = 10**(log_phi_star_z6)

        return phi_star_z6 * 10**(k * (redsh - z_ref))



class Willott2010QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Willott+2010 at z~6.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2010AJ....139..906W/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the double power law fit presented in Section
    5.2.
    """


    def __init__(self, cosmology=None):
        """Initialize the Willott+2010 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Section 5.1
        phi_star_z6 = Parameter(1.14e-8, 'phi_star_z6')

        lum_star = Parameter(-25.13, 'lum_star')

        alpha = Parameter(-1.5, 'alpha')

        beta = Parameter(-2.81, 'beta')

        k = Parameter(-0.47, 'k')

        z_ref = Parameter(6, 'z_ref')

        parameters = {'phi_star_z6': phi_star_z6,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta,
                      'k': k,
                      'z_ref': z_ref}

        param_functions = {'phi_star': self.phi_star}

        lum_type = 'M1450'

        # Komatsu+2009
        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.28)
        ref_redsh = 6.0

        super(Willott2010QLF, self).__init__(parameters, param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)

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




class YangJinyi2016QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Yang+2016 at z~5.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2016ApJ...829...33Y/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The maximum likelihood fit in Yang+2016 includes the quasar samples of
    McGreer+2013 at fainter magnitudes (i.e., the SDSS DR7 and Stripe 82
    samples).

    This implementation adopts the double power law fit presented in Section
    5.2.
    """


    def __init__(self, cosmology=None):
        """Initialize the Willott+2010 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Section 5.1
        log_phi_star_z6 = Parameter(-8.82, 'phi_star_z6', one_sigma_unc=[
            0.15, 0.15])

        lum_star = Parameter(-26.98, 'lum_star', one_sigma_unc=[0.23, 0.23])

        alpha = Parameter(-2.03, 'alpha') # motivated by McGreer+2013

        beta = Parameter(-3.58, 'beta', one_sigma_unc=[0.24, 0.24])

        k = Parameter(-0.47, 'k') # motivated by Fan+2001

        z_ref = Parameter(6, 'z_ref')

        parameters = {'log_phi_star_z6': log_phi_star_z6,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta,
                      'k': k,
                      'z_ref': z_ref}

        param_functions = {'phi_star': self.phi_star}

        lum_type = 'M1450'

        # Komatsu+2009
        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.272, Ob0=0.0456)
        ref_redsh = 5.05

        super(YangJinyi2016QLF, self).__init__(parameters, param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)

    @staticmethod
    def phi_star(redsh, log_phi_star_z6, k, z_ref):
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

        return 10**log_phi_star_z6 * 10**(k * (redsh - z_ref))


class Schindler2019_LEDE_QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Schindler+2019 with LEDE evolution.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2019ApJ...871..258S/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The LEDE model was fit to a range of binned QLF measurements using
    chi-squared minimzation.

    This implementation adopts the LEDE double power law fit presented in
    Table 8.
    """


    def __init__(self, cosmology=None):
        """Initialize the Schindler+2019 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Table 7
        log_phi_star_z2p2 = Parameter(-6.11, 'log_phi_star_z2p2',
                                  one_sigma_unc=[0.05,
                                                 0.05]
                                  )

        lum_star_z2p2 = Parameter(-26.09, 'lum_star_z2p2',
                                  one_sigma_unc=[0.05, 0.05])

        alpha = Parameter(-1.55, 'alpha', one_sigma_unc=[0.02, 0.02])

        beta = Parameter(-3.65, 'beta', one_sigma_unc=[0.06, 0.06])

        c1 = Parameter(-0.61, 'c1', one_sigma_unc=[0.02, 0.02])

        c2 = Parameter(-0.1, 'c2', one_sigma_unc=[0.03, 0.03])


        parameters = {'log_phi_star_z2p2': log_phi_star_z2p2,
                      'lum_star_z2p2': lum_star_z2p2,
                      'alpha': alpha,
                      'beta': beta,
                      'c1': c1,
                      'c2': c2}

        param_functions = {'phi_star': self.phi_star,
                           'lum_star': self.lum_star}


        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 2.9

        super(Schindler2019_LEDE_QLF, self).__init__(parameters, param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)

    @staticmethod
    def phi_star(redsh,log_phi_star_z2p2, c1):
        """Calculate the redshift dependent luminosity function normalization.

        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param log_phi_star_z2p2: Logarithmic source density at z=2.2
        :type log_phi_star_z2p2: float
        :param c1: Redshift evolution parameter
        :type c1: float
        :return:
        """

        return 10**(log_phi_star_z2p2 + c1 * (redsh-2.2))

    @staticmethod
    def lum_star(redsh, lum_star_z2p2, c2):
        """Calculate the redshift dependent break magnitude.

        :param redsh: Redshift for evaluation
        :type redsh: float or numpy.ndarray
        :param lum_star_z2p2: Break magnitude at z=2.2
        :type lum_star_z2p2: float
        :param c2: Redshift evolution parameter
        :type c2: float
        :return:
        """

        return lum_star_z2p2 + c2 * (redsh-2.2)


class Schindler2019_4p25_QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Schindler+2019 at z~4.25.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2019ApJ...871..258S/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The maximum likelihood fit in Schindler+2019 includes the quasar sample of
    Richards+2096 at fainter magnitudes.

    This implementation adopts the double power law fit presented in Table 7.
    """


    def __init__(self, cosmology=None):
        """Initialize the Schindler+2019 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Table 7
        phi_star = Parameter(10**(-8.16), 'phi_star',
                             one_sigma_unc=[3.96*1E-9, 3.96*1E-9])

        lum_star = Parameter(-27.57, 'lum_star', one_sigma_unc=[0.24, 0.24])

        alpha = Parameter(-1.65, 'alpha', one_sigma_unc=[0.46, 0.46])

        beta = Parameter(-4.5, 'beta', one_sigma_unc=[0.18, 0.18])


        parameters = {'phi_star': phi_star,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta}

        param_functions = {}


        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 4.25

        super(Schindler2019_4p25_QLF, self).__init__(parameters,
                                                     param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)

class Schindler2019_3p75_QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Schindler+2019 at z~3.75.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2019ApJ...871..258S/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The maximum likelihood fit in Schindler+2019 includes the quasar sample of
    Richards+2096 at fainter magnitudes.

    This implementation adopts the double power law fit presented in Table 7.
    """


    def __init__(self, cosmology=None):
        """Initialize the Schindler+2019 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Table 7
        phi_star = Parameter(10**(-7.65), 'phi_star',
                             one_sigma_unc=[15.51*1E-9, 15.51*1E-9])

        lum_star = Parameter(-27.17, 'lum_star', one_sigma_unc=[0.28, 0.28])

        alpha = Parameter(-1.7, 'alpha', one_sigma_unc=[0.66, 0.66])

        beta = Parameter(-4.52, 'beta', one_sigma_unc=[0.15, 0.15])


        parameters = {'phi_star': phi_star,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta}

        param_functions = {}


        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 3.75

        super(Schindler2019_3p75_QLF, self).__init__(parameters,
                                                     param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)

class Schindler2019_3p25_QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Schindler+2019 at z~3.25.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2019ApJ...871..258S/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The maximum likelihood fit in Schindler+2019 includes the quasar sample of
    Ross+2013 at fainter magnitudes.

    This implementation adopts the double power law fit presented in Table 7.
    """


    def __init__(self, cosmology=None):
        """Initialize the Schindler+2019 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Table 7
        phi_star = Parameter(10**(-7.33), 'phi_star',
                             one_sigma_unc=[22.07*1E-9, 22.07*1E-9])

        lum_star = Parameter(-27.13, 'lum_star', one_sigma_unc=[0.21, 0.21])

        alpha = Parameter(-1.92, 'alpha', one_sigma_unc=[0.16, 0.16])

        beta = Parameter(-4.58, 'beta', one_sigma_unc=[0.18, 0.18])


        parameters = {'phi_star': phi_star,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta}

        param_functions = {}


        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 3.25

        super(Schindler2019_3p25_QLF, self).__init__(parameters, param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)

class Schindler2019_2p9_QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Schindler+2019 at z~2.9.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2019ApJ...871..258S/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The maximum likelihood fit in Schindler+2019 includes the quasar sample of
    Ross+2013 at fainter magnitudes.

    This implementation adopts the double power law fit presented in Table 7.
    """


    def __init__(self, cosmology=None):
        """Initialize the Schindler+2019 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Table 7
        phi_star = Parameter(10**(-6.23), 'phi_star',
                             one_sigma_unc=[185.93*1E-9, 185.93*1E-9])

        lum_star = Parameter(-25.58, 'lum_star', one_sigma_unc=[0.22, 0.22])

        alpha = Parameter(-1.27, 'alpha', one_sigma_unc=[0.2, 0.2])

        beta = Parameter(-3.44, 'beta', one_sigma_unc=[0.07, 0.07])


        parameters = {'phi_star': phi_star,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta}

        param_functions = {}


        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 2.9

        super(Schindler2019_2p9_QLF, self).__init__(parameters, param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)



class Onken2022_Niida_4p52_QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Onken+2022 at z~4.52.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2021arXiv210512215O/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The maximum likelihood fit in Onken+2022 includes constraints of
    Niida+2020 at fainter magnitudes.

    This implementation adopts the double power law fit presented in Table 4.
    """


    def __init__(self, cosmology=None):
        """Initialize the Onken+2022 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Table 7
        phi_star = Parameter(10**(-8.24), 'phi_star')

        lum_star = Parameter(-27.32, 'lum_star', one_sigma_unc=[0.13, 0.13])

        alpha = Parameter(-2.00, 'alpha')

        beta = Parameter(-3.92, 'beta', one_sigma_unc=[0.32, 0.32])


        parameters = {'phi_star': phi_star,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta}

        param_functions = {}


        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 4.52

        super(Onken2022_Niida_4p52_QLF, self).__init__(parameters, param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)


class Onken2022_Niida_4p83_QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Onken+2022 at z~4.83.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2021arXiv210512215O/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The maximum likelihood fit in Onken+2022 includes constraints of
    Niida+2020 at fainter magnitudes.

    This implementation adopts the double power law fit presented in Table 4.
    """


    def __init__(self, cosmology=None):
        """Initialize the Onken+2022 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Table 7
        phi_star = Parameter(10**(-8.32), 'phi_star')

        lum_star = Parameter(-27.09, 'lum_star', one_sigma_unc=[0.3, 0.3])

        alpha = Parameter(-2.00, 'alpha')

        beta = Parameter(-3.6, 'beta', one_sigma_unc=[0.37, 0.37])


        parameters = {'phi_star': phi_star,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta}

        param_functions = {}


        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 4.83

        super(Onken2022_Niida_4p83_QLF, self).__init__(parameters,
                                                      param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)


class Boutsia2021_QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Boutsia+2021 at z~3.9.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2021ApJ...912..111B/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The fit in Boutsia+2021 includes QLF determinations of Fontanot+2007,
    Glikman+2011, Boutsia+2018, and Giallongo+2019 at fainter magnitudes.

    This implementation adopts the double power law fit presented in Table 4.
    """


    def __init__(self, cosmology=None):
        """Initialize the Boutsia+2021 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Table 7
        phi_star = Parameter(10**(-6.85), 'phi_star',
                             one_sigma_unc=[0.45, 0.6])

        lum_star = Parameter(-26.5, 'lum_star', one_sigma_unc=[0.6, 0.85])

        alpha = Parameter(-1.85, 'alpha', one_sigma_unc=[0.25, 0.15])

        beta = Parameter(-4.025, 'beta', one_sigma_unc=[0.425, 0.575])


        parameters = {'phi_star': phi_star,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta}

        param_functions = {}


        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 3.9

        super(Boutsia2021_QLF, self).__init__(parameters,
                                                      param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)


class Kim2020_QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Kim+2020 at z~5.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2020ApJ...904..111K/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The fit in Kim+2020 includes data from Yang+2016 at brighter luminosities.

    This implementation adopts the double power law fit presented in Table 6,
    Case 1.
    """

    def __init__(self, cosmology=None):
        """Initialize the Boutsia+2021 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Table 7
        log_phi_star = Parameter(-7.36, 'log_phi_star',
                             one_sigma_unc=[0.81, 0.56])

        lum_star = Parameter(-25.78, 'lum_star', one_sigma_unc=[1.1, 1.35])

        alpha = Parameter(-1.21, 'alpha', one_sigma_unc=[0.64, 1.36])

        beta = Parameter(-3.44, 'beta', one_sigma_unc=[0.84, 0.66])


        parameters = {'log_phi_star': log_phi_star,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta}

        param_functions = {'phi_star': self.phi_star}


        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 5.0

        super(Kim2020_QLF, self).__init__(parameters,
                                          param_functions,
                                          lum_type=lum_type,
                                          ref_cosmology=ref_cosmology,
                                          ref_redsh=ref_redsh,
                                          cosmology=cosmology)


    @staticmethod
    def phi_star(redsh, log_phi_star):
        """

        :param redsh:
        :param log_phi_star:
        :return:
        """

        return 10**log_phi_star



class Niida2020_QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Niida+2020 at z~5.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2020ApJ...904...89N/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The fit in Kim+2020 includes data from SDSS at brighter luminosities.

    This implementation adopts the double power law fit presented in Table 6,
    Case 1.
    """


    def __init__(self, cosmology=None):
        """Initialize the Boutsia+2021 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Table 7
        phi_star = Parameter(1.01e-7, 'log_phi_star',
                             one_sigma_unc=[0.29e-7, 0.21e-7])

        lum_star = Parameter(-25.05, 'lum_star', one_sigma_unc=[0.24, 0.1])

        alpha = Parameter(-1.22, 'alpha', one_sigma_unc=[0.1, 0.03])

        beta = Parameter(-2.9, 'beta')


        parameters = {'phi_star': phi_star,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta}

        param_functions = {}


        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 5.0

        super(Niida2020_QLF, self).__init__(parameters,
                                                      param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)





class Giallongo2019_4p5_QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Giallongo+2019 at z~4.5.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2019ApJ...884...19G/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The fit in Giallongo+2019 (model 1) includes QLF determinations of
    Fontanot+2007, Boutsia+2018, and Akiyama+2018 at brighter magnitudes.

    This implementation adopts the double power law fit presented in Table 3
    (model 1).
    """


    def __init__(self, cosmology=None):
        """Initialize the Giallongo+2019 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Table 7
        phi_star = Parameter(10**(-6.68), 'phi_star')

        lum_star = Parameter(-25.81, 'lum_star')

        alpha = Parameter(-1.7, 'alpha')

        beta = Parameter(-3.71, 'beta')


        parameters = {'phi_star': phi_star,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta}

        param_functions = {}


        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 4.5

        super(Giallongo2019_4p5_QLF, self).__init__(parameters,
                                                      param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)

class Giallongo2019_5p6_QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Giallongo+2019 at z~5.6.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2019ApJ...884...19G/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    The fit in Giallongo+2019 (model 4) includes data from CANDELS and SDSS.

    This implementation adopts the double power law fit presented in Table 3
    (model 4).
    """


    def __init__(self, cosmology=None):
        """Initialize the Giallongo+2019 type-I quasar UV luminosity function.
        """

        # ML fit parameters from Table 7
        phi_star = Parameter(10**(-7.05), 'phi_star')

        lum_star = Parameter(-25.37, 'lum_star')

        alpha = Parameter(-1.74, 'alpha')

        beta = Parameter(-3.72, 'beta')


        parameters = {'phi_star': phi_star,
                      'lum_star': lum_star,
                      'alpha': alpha,
                      'beta': beta}

        param_functions = {}


        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 5.6

        super(Giallongo2019_5p6_QLF, self).__init__(parameters,
                                                      param_functions,
                                                     lum_type=lum_type,
                                                     ref_cosmology=ref_cosmology,
                                                     ref_redsh=ref_redsh,
                                                     cosmology=cosmology)


class Richards2006QLF(SinglePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
       Richards+2006 (z=1-5).

       ADS reference:

       The luminosity function is parameterized as a double power law with the
       luminosity variable in absolute magnitudes Mi(z=2. We convert the Mi(
       z=2) magnitudes to M1450 using a simple conversion factor (see
       Ross+2013).

       M1450(z=0) = Mi(z=2) + 1.486

       This implementation adopts the double power law fit presented in Table 7
       (variable power law).
       """


    def __init__(self, cosmology=None):
        """Initialize the Richards2006 type-I quasar UV luminosity function.
        """
        z_pivot = Parameter(2.4, 'z_pivot')
        a1_upp = Parameter(0.83, 'a1_upp', one_sigma_unc=0.01)
        a2_upp = Parameter(-0.11, 'a2_upp', one_sigma_unc=0.01)
        b1_upp = Parameter(1.43, 'b1_upp', one_sigma_unc=0.04)
        b2_upp = Parameter(36.63, 'b2_upp', one_sigma_unc=0.1)
        b3_upp = Parameter(34.39, 'b3_upp', one_sigma_unc=0.26)

        a1_low = Parameter(0.84, 'a1_low')
        a2_low = Parameter(0, 'a2_low')
        b1_low = Parameter(1.43, 'b1_low', one_sigma_unc=0.04)
        b2_low = Parameter(36.63, 'b2_low', one_sigma_unc=0.1)
        b3_low = Parameter(34.39, 'b3_low', one_sigma_unc=0.26)

        z_ref = Parameter(2.45, 'z_ref')
        log_phi_star = Parameter(-5.7, 'log_phi_star')
        lum_ref_star = Parameter(-26 + 1.486, 'lum_ref_star')

        parameters = {'z_pivot': z_pivot,
                      'a1_low': a1_low,
                      'a2_low': a2_low,
                      'a1_upp': a1_upp,
                      'a2_upp': a2_upp,
                      'b1_low': b1_low,
                      'b2_low': b2_low,
                      'b3_low': b3_low,
                      'b1_upp': b1_upp,
                      'b2_upp': b2_upp,
                      'b3_upp': b3_upp,
                      'z_ref': z_ref,
                      'log_phi_star': log_phi_star,
                      'lum_ref_star': lum_ref_star}

        param_functions = {'phi_star': self.phi_star,
                           'lum_ref': self.lum_ref,
                           'alpha': self.alpha}

        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = 2.4

        super(Richards2006QLF, self).__init__(parameters, param_functions,
                                              lum_type=lum_type,
                                              ref_cosmology=ref_cosmology,
                                              ref_redsh=ref_redsh,
                                              cosmology=cosmology)

    @staticmethod
    def lum_ref(redsh, lum_ref_star, b1_low, b2_low, b3_low, b1_upp, b2_upp,
                 b3_upp, z_ref, z_pivot):


        psi = np.log10( (1+redsh) / (1+z_ref))

        if redsh <= z_pivot:
            return lum_ref_star + (b1_low * psi + b2_low * psi**2 + b3_low *
                               psi**3 )
        else:
            return lum_ref_star + (b1_upp * psi + b2_upp * psi**2 + b3_upp *
                               psi**3 )


    @staticmethod
    def alpha(redsh, a1_low, a2_low, a1_upp, a2_upp, z_ref, z_pivot):

        if redsh <= z_pivot:
            return a1_low + a2_low * (redsh - z_ref)
        else:
            return a1_upp + a2_upp * (redsh - z_ref)


    @staticmethod
    def phi_star(redsh, log_phi_star):
        """

        :param redsh:
        :param log_phi_star:
        :return:
        """

        return 10**log_phi_star


    def evaluate(self, lum, redsh, parameters=None):
        """Evaluate the single power law as a function of magnitude ("lum")
        and redshift ("redsh") for the Richards 2006 QLF.

        Function to be evaluated: atelier.lumfun.richards_single_power_law()

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

        # TODO: Try to precalculate factors in init for better performance
        if self.cosmology is not None and self.ref_cosmology is not \
                None:


            distmod_ref = self.ref_cosmology.distmod(self.ref_redsh)
            distmod_cos = self.cosmology.distmod(self.ref_redsh)

            # Convert luminosity according to new cosmology
            if self.lum_type in ['M1450']:
                self.cosm_lum_conv = distmod_ref.value - distmod_cos.value
            else:
                raise NotImplementedError(
                    '[ERROR] Conversions for luminosity '
                    'type {} are not implemented.'.format(
                        self.lum_type))

            self.cosm_density_conv = self.ref_cosmology.h ** 3 / \
                                     self.cosmology.h ** 3

            lum_ref = lum_ref + self.cosm_lum_conv
            phi_star = phi_star * self.cosm_density_conv

        return richards_single_power_law(lum, phi_star, lum_ref, alpha)


class Kulkarni2019QLF(DoublePowerLawLF):
    """Implementation of the type-I quasar UV(M1450) luminosity function of
    Kulkarni+2019 at z~1-6.

    ADS reference: https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.1035K/abstract

    The luminosity function is parameterized as a double power law with the
    luminosity variable in absolute magnitudes at 1450A, M1450.

    This implementation adopts the double power law fit presented in Table 3
    ("Model 3").
    """

    def __init__(self, cosmology=None):
        """Initialize the Kulkarni+2019 type-I quasar UV luminosity function.
        """

        # ML fit parameters from of Model 3 from Table 3.
        c0_0 = Parameter(-6.942, 'c0_0', one_sigma_unc=[0.086, 0.086])
        c0_1 = Parameter(0.629, 'c0_1', one_sigma_unc=[0.045, 0.046])
        c0_2 = Parameter(-0.086, 'c0_2', one_sigma_unc=[0.003, 0.003])

        c1_0 = Parameter(-15.038, 'c1_0', one_sigma_unc=[0.150, 0.156])
        c1_1 = Parameter(-7.046, 'c1_1', one_sigma_unc=[0.101, 0.100])
        c1_2 = Parameter(0.772, 'c1_2', one_sigma_unc=[0.013, 0.013])
        c1_3 = Parameter(-0.030, 'c1_3', one_sigma_unc=[0.001, 0.001])

        c2_0 = Parameter(-2.888, 'c2_0', one_sigma_unc=[0.093, 0.097])
        c2_1 = Parameter(-0.383, 'c2_1', one_sigma_unc=[0.041, 0.039])

        c3_0 = Parameter(-1.602, 'c3_0', one_sigma_unc=[0.028, 0.029])
        c3_1 = Parameter(-0.082, 'c3_1', one_sigma_unc=[0.009, 0.009])

        parameters = {'c0_0': c0_0,
                      'c0_1': c0_1,
                      'c0_2': c0_2,
                      'c1_0': c1_0,
                      'c1_1': c1_1,
                      'c1_2': c1_2,
                      'c1_3': c1_3,
                      'c2_0': c2_0,
                      'c2_1': c2_1,
                      'c3_0': c3_0,
                      'c3_1': c3_1}

        param_functions = {'phi_star': self.phi_star,
                           'lum_star': self.lum_star,
                           'alpha': self.alpha,
                           'beta': self.beta}

        lum_type = 'M1450'

        ref_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        ref_redsh = None

        super(Kulkarni2019QLF, self).__init__(parameters, param_functions,
                                              lum_type=lum_type,
                                              ref_cosmology=ref_cosmology,
                                              ref_redsh=ref_redsh,
                                              cosmology=cosmology)


    @staticmethod
    def lum_star(redsh, c1_0, c1_1, c1_2, c1_3):
        """

        :param redsh:
        :param c1_0:
        :param c1_1:
        :param c1_2:
        :param c1_3:
        :return:
        """

        return np.polynomial.chebyshev.chebval(1+redsh,
                                               c=[c1_0, c1_1, c1_2, c1_3])

    @staticmethod
    def phi_star(redsh, c0_0, c0_1, c0_2):
        """

        :param redsh:
        :param c0_0:
        :param c0_1:
        :param c0_2:
        :return:
        """

        log_phi_star = np.polynomial.chebyshev.chebval(1+redsh,
                                                       c=[c0_0, c0_1, c0_2])

        return 10**log_phi_star

    @staticmethod
    def alpha(redsh, c2_0, c2_1):
        """

        :param redsh:
        :param c2_0:
        :param c2_1:
        :return:
        """

        return np.polynomial.chebyshev.chebval(1+redsh,
                                               c=[c2_0, c2_1])

    @staticmethod
    def beta(redsh, c3_0, c3_1):
        """

        :param redsh:
        :param c3_0:
        :param c3_1:
        :return:
        """

        return np.polynomial.polynomial.polyval(1+redsh, c=[c3_0, c3_1])


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#  BINNED QUASAR LUMINOSITY FUNCTION CLASS AND DATA BELOW
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class BinnedLuminosityFunction(object):

    def __init__(self, lum=None, lum_type=None, lum_unit=None,
                 phi=None, log_phi=None, phi_unit=None,
                 sigma_phi=None, sigma_log_phi=None, ref_cosmology=None,
                 redshift=None, redshift_range=None, cosmology=None, **kwargs):

        self.redshift = redshift
        self.redshift_range =redshift_range

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
                self._get_logphi_from_phi()
            elif log_phi is not None and phi is None:
                self.log_phi = log_phi
                self._get_phi_from_logphi()

            elif log_phi is not None and phi is not None:
                self.phi = phi
                self.log_phi = log_phi

        if sigma_phi is not None or sigma_log_phi is not None:

            if sigma_phi is not None and sigma_log_phi is None:

                self.sigma_phi = sigma_phi
                self._get_sigma_logphi_from_sigma_phi()

            elif sigma_log_phi is not None and sigma_phi is None:

                self.sigma_log_phi = sigma_log_phi
                self._get_sigma_phi_from_sigma_logphi()

            else:
                self.sigma_phi = sigma_phi
                self.sigma_log_phi = sigma_log_phi

        if ref_cosmology is None:
            raise ValueError('[ERROR] No reference cosmology specified!')
        else:
            self.ref_cosmology = ref_cosmology

        if cosmology is None:
            print('[INFO] No cosmology specified, reference cosmology is '
                  'used.')
        else:
            print('[INFO] Converting measurements from reference to specified'
                  ' cosmology.')

            self.cosmology = cosmology

            self._convert_to_cosmology()

    def _get_logphi_from_phi(self):

        self.log_phi = np.log10(self.phi)

    def _get_phi_from_logphi(self):

        self.phi = np.power(10, self.log_phi)

    def _get_sigma_logphi_from_sigma_phi(self):

        if self.sigma_phi.ndim == 1:
            s_logphi_low = np.log10(self.phi-self.sigma_phi) - self.log_phi
            s_logphi_upp = np.log10(self.phi+self.sigma_phi) - self.log_phi

            self.sigma_log_phi = np.abs(np.array([s_logphi_low, s_logphi_upp]))

        if self.sigma_phi.ndim == 2:

            s_logphi_low = np.log10(self.phi - self.sigma_phi[0, :]) \
                           - self.log_phi
            s_logphi_upp = np.log10(self.phi + self.sigma_phi[1, :]) \
                           - self.log_phi

            self.sigma_log_phi = np.abs(np.array([s_logphi_low, s_logphi_upp]))

    def _get_sigma_phi_from_sigma_logphi(self):

        if self.sigma_log_phi.ndim == 1:
            s_phi_low = 10**(-self.sigma_log_phi + self.log_phi) - self.phi
            s_phi_upp = 10**(self.sigma_log_phi + self.log_phi) - self.phi

            self.sigma_phi = np.abs(np.array([s_phi_low, s_phi_upp]))

        if self.sigma_log_phi.ndim == 2:
            s_phi_low = 10 ** (-self.sigma_log_phi[0, :] + self.log_phi) - \
                        self.phi
            s_phi_upp = 10 ** (self.sigma_log_phi[1, :] + self.log_phi) - \
                        self.phi

            self.sigma_phi = np.abs(np.array([s_phi_low, s_phi_upp]))

    def _convert_to_cosmology(self):

        # Luminosity conversion
        distmod_ref = self.ref_cosmology.distmod(self.redshift)
        distmod_cos = self.cosmology.distmod(self.redshift)

        # Convert luminosity according to new cosmology
        if self.lum_type in ['M1450']:
            self.lum = self.lum + distmod_ref.value - distmod_cos.value
        else:
            raise NotImplementedError('[ERROR] Conversions for luminosity '
                                      'type {} are not implemented.'.format(
                                      self.lum_type))

        # Convert density according to new cosmology
        # Note: number density scales as h^-3
        # phi
        phi_h_inv = self.phi * self.ref_cosmology.h**3
        self.phi = phi_h_inv / self.cosmology.h**3

        self._get_logphi_from_phi()
        # sigma_phi

        sigma_phi_inv = self.sigma_phi * self.ref_cosmology.h**3
        self.sigma_phi = sigma_phi_inv / self.cosmology.h**3

        self._get_sigma_logphi_from_sigma_phi()




mcgreer2013_str82 = \
      {'lum': np.array([-27.0, -26.45, -25.9, -25.35, -24.8, -24.25]),
       'log_phi': np.array([-8.4, -7.84, -7.9, -7.53, -7.36, -7.14]),
       'sigma_phi': np.array([2.81, 6.97, 5.92, 10.23, 11.51, 19.9])*1e-9,
       'phi_unit': units.Mpc ** -3 * units.mag ** -1,
       'lum_type': 'M1450',
       'lum_unit': units.mag,
       'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
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
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
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
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
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
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
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
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
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
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
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
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift_mean': np.array([6.68253, 6.70167, 6.65747]),
        'redshift': 6.7,
        'redshift_range': [6.45, 7.05]
        }

jianglinhua2016 =  \
       {'lum': np.array([-26.599, -27.199, -27.799, -28.699, -24.829,
                         -25.929, -26.449]),
        'phi': np.array([5.27E-10, 3.43E-10, 1.36E-10, 1.51E-11, 7.09E-09,
                         3.16E-09, 1.06E-09]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([1.7387E-01, 1.2838E-01, 5.0821E-02, 1.4950E-02,
                               2.6535E+00, 1.2782E+00, 3.3034E-01]) * 1E-9,
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'lum_median': [-26.78, -27.11, -27.61, -29.1, -24.73, -25.74, -26.27],
        'survey': ['main_survey', 'main_survey', 'main_survey', 'main_survey',
                   'stripe82', 'stripe82', 'overlap_region'],
        'redshift': 6.05,
        'redshift_range': [5.7, 6.4]
        }

# Add Willott 2010
# z=6 optical quasar luminosity function data from Willott et al. 2010, AJ, 139, 906.
# Bins for CFHQS, SDSS main and SDSS deep samples. See Sec. 5.1 of the paper.
# Note that the SDSS data have been rebinned by me, so are not the same binned numbers as found in Fan et al. and Jiang et al. papers.
#
# Sample  M_1450(avg) M_1450(low) M_1450(high)  rho (Mpc^-3 mag^-1)  rho_err
#==================================================================================
# SDSS main   -27.74      -28.00      -27.50        2.2908E-10        1.1454E-10
# SDSS main   -27.23      -27.50      -27.00        3.7653E-10        1.6839E-10
# SDSS main   -26.78      -27.00      -26.68        1.0794E-09        4.8271E-10
# SDSS deep   -25.97      -27.00      -25.50        2.1946E-09        9.8146E-10
# SDSS deep   -25.18      -25.50      -24.00        6.6670E-09        2.9816E-09
# CFHQS       -26.15      -27.00      -25.50        1.0070E-09        5.0249E-10
# CFHQS       -24.66      -25.50      -24.00        6.9481E-09        2.0047E-09
# CFHQS       -22.21      -23.50      -22.00        5.1438E-08        5.1437E-08

willott2010 = {'lum': np.array([-27.74,  -27.23, -26.78, -25.97, -25.18,
                                -26.15, -24.66, -22.21]),
               'lum_bins': np.array(
                   [[-28, -27.5], [-27.5, -27], [-27.0, -26.68],
                    [-27.0, -25.5], [-25.5, -24], [-27.0, -25.5],
                    [-25.5, -24.0], [-23.5, -22.0]]),
        'phi': np.array([2.2908E-10, 3.7653E-10, 1.0794E-09,
                         2.1946E-09, 6.6670E-09, 1.0070E-09,
                         6.9481E-09, 5.1438E-08]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([1.1454E-10, 1.6839E-10, 4.8271E-10,
                               9.8146E-10, 2.9816E-09, 5.0249E-10,
                               2.0047E-09, 5.1437E-08]),
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.28),
        'redshift': 6,
        'redshift_range': [5.8, 6.4]
        }


willott2010_cfhqs = {'lum': np.array([-26.15, -24.66, -22.21]),
               'lum_bins': np.array(
                   [[-27.0, -25.5],
                    [-25.5, -24.0], [-23.5, -22.0]]),
        'phi': np.array([1.0070E-09,
                         6.9481E-09, 5.1438E-08]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([5.0249E-10,
                               2.0047E-09, 5.1437E-08]),
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.28),
        'redshift': 6,
        'redshift_range': [5.8, 6.4]
        }


# Add Jinyi Yang 2016
yangjinyi2016 =  \
       {'lum': np.array([-28.99, -28.55, -28.05, -27.55, -27.05]),
        'log_phi': np.array([-9.48, -9.86, -9.36, -9.09, -8.7]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([0.33, 0.08, 0.15, 0.19, 0.32]) * 1E-9,
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
        'redshift': 5,
        'redshift_range': [4.7, 5.4]
        }


# Schindler+2019 ELQS QLF

# lum bin edges −29.1, −28.7, −28.3, −28, −27.7, −27.5

#M1450 N Ncorr log10Φ σΦ Bin
#(mag) (Mpc−3 mag−1) (Gpc−3 mag−1) Filled
# 2.8 - 3.0
# −28.9 2 3.9 −9.46 0.25 True
# −28.5 4 10.1 −9.05 0.50 True
# −28.15 11 28.6 −8.47 1.08 True
# −27.85 7 21.2 −8.60 1.00 True
# −27.6 9 41.3 −8.13 2.59 True

schindler2019_2p9 = \
       {'lum': np.array([-28.9, -28.5, -28.15, -27.85, -27.6]),
        'lum_bins': np.array([[-29.1, -28.7], [-28.7, -28.3],
                              [-28.3, -28.0], [-28.0, -27.7],
                              [-27.7, -27.5]]),
        'log_phi': np.array([-9.46, -9.05, -8.47, -8.6, -8.13]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([0.25, 0.50, 1.08, 1.00, 2.59]) * 1E-9,
        'bin_filled': np.array([True, True, True, True, True]),
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 2.9,
        'redshift_range': [2.8, 3.0]
        }

# 3.0 - 3.5
# −28.9 3 3.9 −9.85 0.08 True
# −28.5 10 12.9 −9.33 0.15 True
# −28.15 28 42.0 −8.69 0.39 True
# −27.85 31 67.6 −8.48 0.60 True
# −27.6 17 69.9 −8.29 1.28 False

schindler2019_3p25 = \
       {'lum': np.array([-28.9, -28.5, -28.15, -27.85, -27.6]),
        'lum_bins': np.array([[-29.1, -28.7], [-28.7, -28.3],
                              [-28.3, -28.0], [-28.0, -27.7],
                              [-27.7, -27.5]]),
        'log_phi': np.array([-9.85, -9.33, -8.69, -8.48, -8.29]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([0.08, 0.15, 0.39, 0.6, 1.28]) * 1E-9,
        'bin_filled': np.array([True, True, True, True, False]),
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 3.25,
        'redshift_range': [3.0, 3.5]
        }

# 3.5 - 4.0
# −28.9 2 2.1 −10.09 0.06 True
# −28.5 6 7.2 −9.56 0.11 True
# −28.15 12 18.8 −9.02 0.28 True
# −27.85 6 15.4 −9.08 0.34 False
# −27.6 2 9.3 −8.68 1.49 False

schindler2019_3p75 = \
       {'lum': np.array([-28.9, -28.5, -28.15, -27.85, -27.6]),
        'lum_bins': np.array([[-29.1, -28.7], [-28.7, -28.3],
                              [-28.3, -28.0], [-28.0, -27.7],
                              [-27.7, -27.5]]),
        'log_phi': np.array([-10.09, -9.56, -9.02, -9.08, -8.68]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([0.06, 0.11, 0.28, 0.34, 1.49]) * 1E-9,
        'bin_filled': np.array([True, True, True, False, False]),
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 3.75,
        'redshift_range': [3.5, 4.0]
        }


# 4.0 - 4.5
# −28.9 2 2.3 −10.04 0.06 True
# −28.5 5 7.8 −9.50 0.14 True

schindler2019_4p25 = \
       {'lum': np.array([-28.9, -28.5]),
        'lum_bins': np.array([[-29.1, -28.7], [-28.7, -28.3]]),
        'log_phi': np.array([-10.04, -9.5]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([0.06, 0.14]) * 1E-9,
        'bin_filled': np.array([True, True]),
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 4.25,
        'redshift_range': [4.0, 4.25]
        }

# Glikman+2011
# M1450 Bin Center 〈M1450 〉a ΦNQSO
# (mag) (mag) (10−8 Mpc−3 mag−1)
# SDSS
# −28.5 −28.45 0.008+0.011−0.005 2
# −27.5 −27.33 0.20+0.04−0.03 41
# −26.5 −26.46 0.93 ± 0.07 169
# −25.5 −25.70 4.3 ± 0.5 102
# −24.5 −24.72 0.4+0.3−0.2 4
# NDWFS+DLS
# −25.5 −25.37 24+13−9 7
# −24.5 −24.71 8.8+8.5−4.8 3
# −23.5 −23.47 143+77−53 7
# −22.5 −22.61 307+208−133 5
# −21.5 −21.61 434+572−280 2

glikman2011_ndwfs_dls = \
       {'lum': np.array([-25.5, -24.5, -23.5, -22.5, -21.5]),
        'lum_mean': np.array([-25.37, -24.71, -23.47, -22.61, -21.61]),
        'phi': np.array([24, 8.8, 143, 307, 434])*1E-8,
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([[9, 4.8, 53, 133, 280],[13, 8.5, 77, 208,
                                                       572]]) * 1E-8,
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 4.0,
        'redshift_range': [3.74, 5.06]
        }

# Giallongo2019
# Δz M 1450 fobs fcorr Nobj fMC
# 4–5
# −19 6.81 -+14.54 5.81 8.72 6 18.05±4.27
# −20 4.74 -+11.47 4.59 6.88 6 8.03±3.34
# −21 3.29 -+5.08 2.21 3.45 5 4.52±1.15
# −22 1.24 -+1.31 0.87 1.74 2 1.33±0.11
# 5–6.1
# −19 3.62 -+7.27 4.02 7.12 3 6.27±3.42
# −20 3.12 -+4.77 2.31 3.79 4 2.91±1.84
# −21 0.65 -+0.69 0.60 1.61 1 1.13±0.70
# −22 0.61 -+0.62 0.54 1.44 1 0.80±0.33


giallongo2019_z4p5 = \
       {'lum': np.array([-19, -20, -21, -22]),
        'phi': np.array([14.54, 11.47, 5.08, 1.31])*1E-6,
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([[5.81, 4.59, 2.21, 0.87],
                               [8.72, 6.88, 3.45, 1.74]]) * 1E-6,
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 4.5,
        'redshift_range': [4, 5]
        }

giallongo2019_z5p05 = \
       {'lum': np.array([-19, -20, -21, -22]),
        'phi': np.array([7.27, 4.77, 0.69, 0.62])*1E-6,
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([[4.02, 2.31, 0.6, 0.54],
                               [7.12, 3.79, 1.61, 1.44]]) * 1E-6,
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 5.05,
        'redshift_range': [5, 6.1]
        }

# Boutsia2018 COSMOS
# M1450 ΦupsF lowsF NAGN corrF
# Mpc Mag3 1 - -
# −24.5 3.509e-07 2.789e-07 1.699e-07 4 7.018e-07
# −23.5 7.895e-07 3.616e-07 2.595e-07 9 1.579e-06

boutsia2018 = \
       {'lum': np.array([-24.5, -23.5]),
        'phi': np.array([7.018e-07, 1.579e-06]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([[1.699e-07, 2.595e-07],
                               [2.789e-07, 3.616e-07]]),
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 3.9,
        'redshift_range': [3.6, 4.2]
        }


#Boutsia2021 QUBRICS
# Interval <M1450 >NQSO ΦσΦ(up)σΦ(low)
# cMpc−3 cMpc−3 cMpc−3
# −28.5 M1450 −28.0 −28.25 36 1.089E-09 2.136E-10 1.809E-10
# −29.0 M1450 −28.5 −28.75 9 2.611E-10 1.196E-10 8.581E-11
# −29.5 M1450 −29.0 −29.25 2 5.802E-11 7.712E-11 3.838E-11

boutsia2021  = \
       {'lum': np.array([-28.25, -28.75, -29.25]),
        'lum_bins': np.array([[-28.5, -28.0], [-29.0, -28.5], [-29.5, -29.0]]),
        'phi': np.array([1.089E-09, 2.611E-10, 5.802E-11]),
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([[1.809E-10, 8.581E-11, 3.838E-11],
                               [2.136E-10, 1.196E-10, 7.712E-11]]),
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 3.9,
        'redshift_range': [3.6, 4.2]
        }


# Kim, Yongjung 2021 IMS
# −27.25 1 4.7±4.7 0 L45 1.52±0.23
# −26.75 1 4.7±4.7 1 7.1±7.1 0 L
# −26.25 3 15.1±8.7 3 19.6±11.3 0 L
# −25.75 3 16.5±9.5 3 19.2±11.1 0 L
# −25.25 5 30.8±13.8 4 28.8±14.4 0 L
# −24.75 7 48.1±18.2 6 51.3±20.9 0 L
# −24.25 10 70.7±22.4 6 55.0±22.5 0 L
# −23.75 7 59.0±22.3 6 69.5±28.4 0 L
# −23.25 6 104.1±42.5 3 68.1±39.3 0 L

kim2021  = \
       {'lum': np.array([-27.25, -26.75, -26.25, -25.75, -25.25, -24.75,
                         -24.25,-23.75, -23.25]),
        'phi': np.array([4.7, 4.7, 15.1, 16.5, 30.8, 48.1, 70.7, 59.0,
                         104.1])*1E-9,
        'phi_unit': units.Mpc**-3 * units.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': units.mag,
        'sigma_phi': np.array([[4.7, 4.7, 8.7, 9.5, 13.8, 18.2, 22.4, 22.3,
                                42.5],
                               [4.7, 4.7, 8.7, 9.5, 13.8, 18.2, 22.4, 22.3,
                                42.5]])*1E-9,
        'ref_cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 5,
        'redshift_range': [4.7, 5.4]
        }



def verification_plots_kulkarni2019QLF():

    plot_defaults.set_paper_defaults()

    qlf = Kulkarni2019QLF()

    redshifts = np.linspace(0, 7, 200)
    lum = -27

    main_parameters = np.zeros((4, len(redshifts)))

    for idx, redsh in enumerate(redshifts):

        params = qlf.evaluate_main_parameters(lum, redsh)
        main_parameters[0, idx] = params['phi_star']
        main_parameters[1, idx] = params['lum_star']
        main_parameters[2, idx] = params['alpha']
        main_parameters[3, idx] = params['beta']

    # Set up figure
    fig = plt.figure(num=None, figsize=(6, 4), dpi=120)
    fig.subplots_adjust(left=0.13, bottom=0.15, right=0.87, top=0.92,
                        hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(redshifts, np.log10(main_parameters[0, :]))
    ax1.set_xlabel(r'$\rm{Redshift}$', fontsize=12)
    ax1.set_ylabel(r'$\log (\Phi^*/\rm{mag}^{-1}\rm{cMpc}^{-3})$', fontsize=12)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(redshifts, main_parameters[1, :])
    ax2.set_xlabel(r'$\rm{Redshift}$', fontsize=12)
    ax2.set_ylabel(r'$M_{1450}^*$', fontsize=12)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(redshifts, main_parameters[2, :])
    ax3.set_xlabel(r'$\rm{Redshift}$', fontsize=12)
    ax3.set_ylabel(r'$\alpha$', fontsize=12)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(redshifts, main_parameters[3, :])
    ax4.set_xlabel(r'$\rm{Redshift}$', fontsize=12)
    ax4.set_ylabel(r'$\beta$', fontsize=12)

    lum = np.linspace(-32, -19, 200)
    redshifts = [4.4, 5.1, 6.0, 7.0]

    # Set up figure
    fig2 = plt.figure(num=None, figsize=(6, 4), dpi=120)
    fig2.subplots_adjust(left=0.13, bottom=0.15, right=0.87, top=0.92)

    ax = fig2.add_subplot(1, 1, 1)

    ax.plot(lum, np.log10(qlf(lum, redshifts[0])), label='z=4.4')
    ax.plot(lum, np.log10(qlf(lum, redshifts[1])), label='z=5.1')
    ax.plot(lum, np.log10(qlf(lum, redshifts[2])), label='z=6.0')
    ax.plot(lum, np.log10(qlf(lum, redshifts[3])), label='z=7.0')

    ax.set_xlabel(r'$M_{1450}$', fontsize=14)
    ax.set_ylabel(r'$\log (\Phi/\rm{mag}^{-1}\rm{cMpc}^{-3})$', fontsize=14)
    ax.set_xlim(-19, -32)
    ax.set_ylim(-12, -4)
    ax.legend(fontsize=12)

    plt.show()


def verification_plots_richards2006QLF():

    plot_defaults.set_paper_defaults()

    qlf = Richards2006QLF()

    # # Set up figure
    # fig = plt.figure(num=None, figsize=(6, 4), dpi=120)
    # fig.subplots_adjust(left=0.13, bottom=0.15, right=0.87, top=0.92,
    #                     hspace=0.3, wspace=0.3)
    #
    # lum = np.arange(-30, -24, 0.1)
    #
    # redsh = 2.01
    # ax1 = fig.add_subplot(2, 2, 1)
    # ax1.plot(lum, np.log10(qlf(lum, redsh)))
    # ax1.set_xlabel(r'$\rm{Redshift}$', fontsize=12)
    # ax1.set_ylabel(r'$\log (\Phi^*/\rm{mag}^{-1}\rm{cMpc}^{-3})$', fontsize=12)
    #
    #
    # redsh = 0.49
    # ax1.plot(lum, np.log10(qlf(lum, redsh)))
    #
    # redsh = 5.0
    # ax1.plot(lum, np.log10(qlf(lum, redsh)))
    #
    # redsh = 2.4
    # ax2 = fig.add_subplot(2, 2, 2)
    # ax2.plot(lum, np.log10(qlf(lum, redsh)))
    # ax2.set_xlabel(r'$\rm{Redshift}$', fontsize=12)
    # ax2.set_ylabel(r'$\log (\Phi^*/\rm{mag}^{-1}\rm{cMpc}^{-3})$', fontsize=12)
    #
    # redsh = 2.8
    # ax3 = fig.add_subplot(2, 2, 3)
    # ax3.plot(lum, np.log10(qlf(lum, redsh)))
    # ax3.set_xlabel(r'$\rm{Redshift}$', fontsize=12)
    # ax3.set_ylabel(r'$\log (\Phi^*/\rm{mag}^{-1}\rm{cMpc}^{-3})$', fontsize=12)
    #
    # redsh = 3.25
    # ax4 = fig.add_subplot(2, 2, 4)
    # ax4.plot(lum, np.log10(qlf(lum, redsh)))
    # ax4.set_xlabel(r'$\rm{Redshift}$', fontsize=12)
    # ax4.set_ylabel(r'$\log (\Phi^*/\rm{mag}^{-1}\rm{cMpc}^{-3})$', fontsize=12)
    #
    # plt.show()
    #



    # Set up figure
    fig = plt.figure(num=None, figsize=(6, 4), dpi=120)
    fig.subplots_adjust(left=0.13, bottom=0.15, right=0.87, top=0.92,
                        hspace=0.3, wspace=0.3)

    redshifts = np.arange(0.5, 5, 0.01)
    qso_density = np.zeros_like(redshifts)
    mlow = -31
    mupp = -26

    for idx, redsh in enumerate(redshifts):

        qso_density[idx] = qlf.integrate_lum(redsh, [mlow, mupp]) / 1e-9

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(redshifts, qso_density)
    ax.semilogy()
    ax.set_ylabel(r'$n\,(M_{1450}<-26,z)\ (\rm{cGpc}^{-3})$',
                  fontsize=15)
    ax.set_xlabel(r'$\rm{Redshift}$', fontsize=15)
    plt.show()

if __name__ == '__main__':

    # verification_plots_kulkarni2019QLF()

    verification_plots_richards2006QLF()