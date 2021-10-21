
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u


mcgreer2013_str82 = \
      {'lum': np.array([-27.0, -26.45, -25.9, -25.35, -24.8, -24.25]),
       'log_phi': np.array([-8.4, -7.84, -7.9, -7.53, -7.36, -7.14]),
       'sigma_log_phi': np.array([2.81, 6.97, 5.92, 10.23, 11.51, 19.9])*1e-9,
       'phi_unit': u.Mpc ** -3 * u.mag ** -1,
       'lum_type': 'M1450',
       'lum_unit': u.mag,
       'cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
       'redshift': 4.9,
       'redshift_range': [4.7, 5.1]
       }

mcgreer2013_dr7 = \
       {'lum': np.array([-28.05, -27.55, -27.05, -26.55, -26.05]),
        'log_phi': np.array([-9.45, -9.24, -8.51, -8.20, -7.9]),
        'phi_unit': u.Mpc**-3 * u.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': u.mag,
        'sigma_log_phi': np.array([0.21, 0.26, 0.58, 0.91, 1.89])*1e-9,
        'cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
        'redshift': 4.9,
        'redshift_range': [4.7, 5.1]
        }

mcgreer2018_main = \
       {'lum': np.array([-28.55, -28.05, -27.55, -27.05, -26.55, -26.05]),
        'log_phi': np.array([-9.90, -9.70, -8.89, -8.41, -8.10, -8.03]),
        'phi_unit': u.Mpc**-3 * u.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': u.mag,
        'sigma_log_phi': np.array([0.12, 0.14, 0.37, 0.72, 1.08, 1.74])*1e-9,
        'cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
        'redshift': 5,
        'redshift_range': [4.7, 5.3]
        }

mcgreer2018_s82 = \
       {'lum': np.array([-27.00, -26.45, -25.90, -25.35, -24.80, -24.25]),
        'log_phi': np.array([-8.06, -7.75, -8.23, -7.47, -7.24, -7.22]),
        'phi_unit': u.Mpc**-3 * u.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': u.mag,
        'sigma_log_phi': np.array([5.57, 6.97, 3.38, 10.39, 13.12, 21.91])*1e-9,
        'cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
        'redshift': 5,
        'redshift_range': [4.7, 5.3]
        }

mcgreer2018_cfhtls_wide = \
       {'lum': np.array([-26.35, -25.25, -24.35, -23.65, -22.90]),
        'log_phi': np.array([-8.12, -7.56, -7.25, -7.32, -7.32]),
        'phi_unit': u.Mpc**-3 * u.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': u.mag,
        'sigma_log_phi': np.array([4.34, 12.70, 18.05, 23.77, 28.24])*1e-9,
        'cosmology': FlatLambdaCDM(H0=70, Om0=0.272),
        'redshift': 5,
        'redshift_range': [4.7, 5.3]
        }

matsuoka2018 =        \
       {'lum': np.array([-22, -22.75, -23.25, -23.75, -24.25, -24.75, -25.25,
                         -25.75, -26.25, -26.75, -27.5, -29]),
        'phi': np.array([16.2, 23.0, 10.9, 8.3, 6.6, 7.0, 4.6, 1.33, 0.9, 0.58,
                         0.242, 0.0079])*1e-9,
        'phi_unit': u.Mpc**-3 * u.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': u.mag,
        'sigma_log_phi': np.array([16.2, 8.1, 3.6, 2.6, 2.0, 1.7, 1.2,
                                   0.6, 0.32, 0.17, 0.061, 0.0079])*1e-9,
        'cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 6.1,
        'redshift_range': [5.7, 6.5]
        }


wangfeige2019 =        \
       {'lum': np.array([-22, -22.75, -23.25, -23.75, -24.25, -24.75, -25.25,
                         -25.75, -26.25, -26.75, -27.5, -29]),
        'phi': np.array([16.2, 23.0, 10.9, 8.3, 6.6, 7.0, 4.6, 1.33, 0.9, 0.58,
                         0.242, 0.0079])*1e-9,
        'phi_unit': u.Mpc**-3 * u.mag**-1,
        'lum_type': 'M1450',
        'lum_unit': u.mag,
        'sigma_log_phi': np.array([16.2, 8.1, 3.6, 2.6, 2.0, 1.7, 1.2,
                                   0.6, 0.32, 0.17, 0.061, 0.0079])*1e-9,
        'cosmology': FlatLambdaCDM(H0=70, Om0=0.3),
        'redshift': 6.7,
        'redshift_range': [6.45, 7.05]
        }

