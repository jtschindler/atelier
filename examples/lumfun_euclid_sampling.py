import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import RegularGridInterpolator
from astropy.cosmology import Planck18
from astropy import units as u
from IPython import embed

from atelier import lumfun
from enigma.tpe.m912 import lamL_lam as lamL_lam
from qso_fitting.utils.get_paths import get_HI_DW_path

cosmology = Planck18
Matsuoka2023 = lumfun.Matsuoka2023DPLQLF()
Wang2019 = lumfun.WangFeige2019DPLQLF()
Jiang2016 = lumfun.JiangLinhua2016QLF()

J_range = np.array([22, 23])
z_range = np.array([7, 12])
sky_area = 15000
seed = 274245628

sample_file = os.path.join(get_HI_DW_path(), 'euclid_forecast/euclid_samples_z_{}-{}_J_{}-{}_seed_{}.hdf5'.format(*z_range, *J_range, seed))
plotting = True

#def k_corr(z, a_nu=-0.5):
#    return -2.5*(1+a_nu)*np.log10(1+z) - 2.5*a_nu*np.log10(145/1254)

#def dist_mod(z, cosmology=Planck18):
#    return 5*np.log10((cosmology.luminosity_distance(z)).to('pc').value/10)

#def Jmag_to_M1450(Jmag, z, cosmology=Planck18, a_nu=-.5):
#    return Jmag + dist_mod(z, cosmology=cosmology) + k_corr(z, a_nu=a_nu)

from astropy import units as u
from astropy.cosmology import Planck18
from enigma.tpe.m912 import lamL_lam as lamL_lam

def Jmag_to_M1450(Jmag, z, cosmology=Planck18, ALPHA_EUV=1.7):

    lam_l = lamL_lam(z, Jmag, 'J', 1450.0, cosmo=cosmology, IGNORE=False, ALPHA_EUV=ALPHA_EUV)*u.erg/u.s
    lognu_1450 = np.log10((1450.0*u.Angstrom).to('Hz',equivalencies=u.spectral()).value)

    Lnu_1450 = lam_l/(np.power(10.0,lognu_1450)*u.Hz)
    fnu_1450 = (Lnu_1450/4.0/np.pi/(10.0*u.pc)**2).to('Jansky')
    Mnu_1450 = -2.5*np.log10(fnu_1450/(3631.0*u.Jansky))

    return Mnu_1450

def Jmag_to_M1450_interp(Jmag_to_M1450, J_range, z_range, ngrid_J=51, ngrid_z=51):
    J_grid = np.linspace(*J_range, ngrid_J)
    z_grid = np.linspace(*z_range, ngrid_z)
    M1450_grid = np.zeros((ngrid_J, ngrid_z))
    for i in range(ngrid_J):
        for j in range(ngrid_z):
            M1450_grid[i,j] = Jmag_to_M1450(J_grid[i], z_grid[j])
    interp = lambda x: RegularGridInterpolator((J_grid, z_grid), M1450_grid)(x)[0]
    return interp

#print(Wang2019.integrate_lum(6.5, [-30, -26]))
#print(Wang2019.integrate_lum(6.75, [-30, -26]))
#print(Wang2019.integrate_lum(7.0, [-30, -26]))
#print(Jiang2016.integrate_over_lum_redsh([-30, -25], [7.5, 8.5], cosmology=cosmology)*15000/(180/np.pi)**2)
#print(Jiang2016.integrate_over_lum_redsh([-30, -24.5], [7., 7.5], cosmology=cosmology)*15000/(180/np.pi)**2)
#print(Jiang2016.integrate_over_lum_redsh([-30, -24.5], [7.5, 8.], cosmology=cosmology)*15000/(180/np.pi)**2)
#print(Jiang2016.integrate_over_lum_redsh([-30, -24.5], [8., 8.5], cosmology=cosmology)*15000/(180/np.pi)**2)
#print(Jiang2016.integrate_over_lum_redsh([-30, -24], [8.5, 9.], cosmology=cosmology)*15000/(180/np.pi)**2)
#Jiang2016.sample([-30, -25], [7.5, 8.5], cosmology, sky_area)
#Jiang2016.sample_mcmc([-30, -25], [7.5, 8.5], cosmology, sky_area)
#embed()
sample_J, sample_z = Wang2019.sample_mcmc(J_range, z_range, cosmology, sky_area, app2absmag=Jmag_to_M1450_interp(Jmag_to_M1450, J_range, z_range), seed=seed)

with h5py.File(sample_file, 'w') as f:
    group = f.create_group('QLF_samples')
    group.attrs['J_min'] = J_range[0]
    group.attrs['J_max'] = J_range[1]
    group.attrs['z_min'] = z_range[0]
    group.attrs['z_max'] = z_range[1]
    group.attrs['sky_area'] = sky_area
    group.attrs['seed'] = seed
    group.attrs['QLF'] = 'Wang2019'
    group.create_dataset('Jmag', data=sample_J)
    group.create_dataset('z', data=sample_z)


if plotting == True:
    plt.plot(sample_z, sample_J, '.')
    plt.xlabel('Redshift', fontsize=14)
    plt.ylabel('$J$', fontsize=14)
    plt.title('Sampled sources from the Wang+2019 QLF')
    plt.show()