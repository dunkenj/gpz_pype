import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column, vstack
import astropy.units as u
import os
import glob
import pickle

from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from collections import OrderedDict

"""
Global 5-sigma depths estimated from the peak of the depth distribution:

depth_summary['depthlo'][1:-1][np.argmax(depth_summary['counts_ptsrc_g'][1:-1])]+0.05
depth_summary['depthlo'][1:-1][np.argmax(depth_summary['counts_ptsrc_r'][1:-1])]+0.05
depth_summary['depthlo'][1:-1][np.argmax(depth_summary['counts_ptsrc_z'][1:-1])]+0.05

"""
#####
f0 = (3631*u.Jy).to(u.uJy).value
nanomaggy_to_ujy = 10**((23.9-22.5)/2.5)
nanovega_to_ujy_w1 = 10**((23.9-2.699-22.5)/2.5)
nanovega_to_ujy_w2 = 10**((23.9-3.339-22.5)/2.5)

def makedir(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)

class Params(object):
    def __init__(self, input_path=None):
        if input_path != None:
            self.read(input_path)

    def read(self, file):
        with open(file, 'r') as param:
            lines = param.readlines()
            self.params = OrderedDict()

            for line in lines:
                if line[0] not in ['#', '\n']:
                    try:
                        parline = line.strip().split('#')[0]
                    except:
                        parline = line
                    parts = parline.strip().split('=')
                    self.params[parts[0].strip()] = parts[1].strip()

    def write(self, file=None):
        if file == None:
            print('No output file specified...')
        else:
            output = open(file, 'w')
            for param in self.params.keys():
                output.write('{0:30s} = {1}\n'.format(param, self.params[param]))
                    #str = '%-25s %'+self.formats[param]+'\n'
            #
            output.close()

    def __getitem__(self, param_name):
        """
        Get item from ``params`` dict.
        """
        if param_name not in self.params.keys():
            print('Column {0} not found.  Check `column_names` attribute.'.format(param_name))
            return None
        else:
            #str = 'out = self.%s*1' %column_name
            #exec(str)
            return self.params[param_name]

    def __setitem__(self, param_name, value):
        self.params[param_name] = value


def flux_to_lupt(flux, fluxerr, b):
    """ Calculate asinh magnitudes 'luptitudes' for given fluxes.

    Parameters
    ----------
    flux : array
        flux density (in units of microjansky)
    fluxerr : array
        Uncertainty on flux (in units of microjansky)
    b : float
        Dimensionless softening parameter for asinh magnitudes. Set externally
        and

    Returns
    -------

    lupt : array
        asinh magnitudes (luptitudes) with AB zeropoint
    lupterr : array
        Uncertainty on lupt

    """
    lupt = (-2.5/np.log(10)) * (np.arcsinh((flux/f0)/(2*b)) + np.log(b))
    lupterr = (2.5/np.log(10)) * np.abs(fluxerr/flux) / np.sqrt(1 + (2*b / (flux/f0))**2)
    return lupt, lupterr

def load_legacy(catalog, format='fits', offsets=None):
    """ Load a Legacy Surveys optical catalog and process ready for use.

    Parameters
    ----------
    legacy_path : str
        Path to input Legacy Catalog brick or sweep
    format : str (default = 'fits')
        astropy compatible catalog format.


    Returns
    -------
    sweep : astropy.Table class
        Processed Legacy catalog suitable for likelihood ratio or photo-z use.
    """

    if isinstance(catalog, Table):
        sweep = catalog
    else:
        sweep = Table.read(catalog, format=format)

    for name in sweep.colnames:
        sweep[name].name = name.lower()

    withobs = (sweep['release'] > 0)
    sweep = sweep[withobs]

    unique_id = np.array(['{0}_{1}'.format(row['brickid'], row['objid']) for row in sweep])

    bands_all = ['g', 'r', 'z', 'w1', 'w2' , 'w3', 'w4']
    bands_lupt = ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']

    for band in bands_all:
        sweep[f'flux_{band}'] *= nanomaggy_to_ujy / sweep[f'mw_transmission_{band}']
        sweep[f'fluxerr_{band}'] = (sweep[f'flux_ivar_{band}']**-0.5 *
                                    nanomaggy_to_ujy /
                                    sweep[f'mw_transmission_{band}'])

        if band in bands_lupt:
            b = ((1/np.sqrt(sweep[f'psfdepth_{band}'])) * nanomaggy_to_ujy) / f0
            lupt, lupterr = flux_to_lupt(sweep[f'flux_{band}'], sweep[f'fluxerr_{band}'], b)

            sweep.add_column(Column(data=lupt,
                                    name=f'lupt_{band}', meta={'b':b}))
            sweep.add_column(Column(data=lupterr,
                                    name=f'lupterr_{band}', meta={'b':b}))

    sweep['ls_id'] = unique_id
    sweep['ANYMASK_OPT'] = ((sweep['anymask_g'] + sweep['anymask_r'] + sweep['anymask_z']) > 0)

    if offsets != None:
        mag_offsets = pickle.load(open(offsets, 'rb'))
        g_z = np.clip(sweep['lupt_g'] - sweep['lupt_z'], 0.3, 3.3)

        north = (sweep['dec'] > 32.375)

        dg = (mag_offsets['g'][0] + mag_offsets['g'][1]*g_z +
              mag_offsets['g'][2]*(g_z**2) + mag_offsets['g'][3]*(g_z**3))

        dr = (mag_offsets['r'][0] + mag_offsets['r'][1]*g_z +
              mag_offsets['r'][2]*(g_z**2) + mag_offsets['r'][3]*(g_z**3))

        dz = (mag_offsets['z'][0] + mag_offsets['z'][1]*g_z +
              mag_offsets['z'][2]*(g_z**2) + mag_offsets['z'][3]*(g_z**3))

        sweep['lupt_g'][north] -= dg[north]
        sweep['lupt_r'][north] -= dr[north]
        sweep['lupt_z'][north] -= dz[north]

    return sweep, withobs

dr14q_specz = Table.read('specz/DR14Q_v4_4.fits', format='fits')
dr14_specz = Table.read('specz/specObj-dr14.fits', format='fits')

dr14_north, withobs = load_legacy('specz/survey-dr8-north-specObj-dr14.fits', offsets='legacy_north_south_offset.pkl')
dr14_north['z_spec'] = dr14_specz['Z'][withobs]
cut = (dr14_specz['ZWARNING'][withobs] == 0) * (dr14_specz['CLASS'][withobs] != 'STAR  ') * (dr14_specz['Z'][withobs] > 0.01) * (dr14_specz['Z'][withobs] <= 5)
stars = (dr14_specz['ZWARNING'][withobs] == 0) * (dr14_specz['CLASS'][withobs] == 'STAR  ')
dr14_north_star = dr14_north[stars]
dr14_north = dr14_north[cut]

dr14_south, withobs = load_legacy('specz/survey-dr8-south-specObj-dr14.fits')
dr14_south['z_spec'] = dr14_specz['Z'][withobs]
cut = (dr14_specz['ZWARNING'][withobs] == 0) * (dr14_specz['CLASS'][withobs] != 'STAR  ') * (dr14_specz['Z'][withobs] > 0.01) * (dr14_specz['Z'][withobs] <= 5)
stars = (dr14_specz['ZWARNING'][withobs] == 0) * (dr14_specz['CLASS'][withobs] == 'STAR  ')
dr14_south_star = dr14_south[stars]
dr14_south = dr14_south[cut]

dr14q_north, withobs = load_legacy('specz/survey-dr8-north-dr14Q_v4_4.fits', format='fits', offsets='legacy_north_south_offset.pkl')
dr14q_north['z_spec'] = dr14q_specz['Z'][withobs]
dr14q_south, withobs = load_legacy('specz/survey-dr8-south-dr14Q_v4_4.fits', format='fits')
dr14q_south['z_spec'] = dr14q_specz['Z'][withobs]

#in_quasar_cat = np.array([id in dr14q['ls_id'] for id in dr14_north['ls_id']])

dr14 = vstack([dr14q_south, dr14q_north, dr14_south, dr14_north], join_type='inner')
unique, keep, reject = np.unique(dr14['ls_id'], return_index=True, return_inverse=True)
dr14 = dr14[keep]

sdss_stars = vstack([dr14_south_star, dr14_north_star])
unique, keep, reject = np.unique(sdss_stars['ls_id'], return_index=True, return_inverse=True)
sdss_stars = sdss_stars[keep]

index = np.arange(len(dr14))
i_train, i_reserve = train_test_split(index, test_size=0.5)

dr14[i_train].write(f'sdss_dr14sz_combined_train.fits', format='fits', overwrite=True)
dr14[i_reserve].write(f'sdss_dr14sz_combined_reserve.fits', format='fits', overwrite=True)

dr14 = dr14[i_train]


help_specz, _ = load_legacy('specz/help_specz_lsmatch_v1_wise_corr.fits', format='fits', offsets='legacy_north_south_offset.pkl')
help_specz['t1_z_spec'].name = 'z_spec'

vhzq_specz, _ = load_legacy('specz/vhzq_ls_all_wise_corr.csv', format='ascii.csv', offsets='legacy_north_south_offset.pkl')
#vhzq_specz['t1_redshift'].name = 'z_spec'

# hmatch = vhzq_specz.copy()
#
# w1f = np.copy(hmatch['flux_w1'])
# w1f[hmatch['dec'] < 32.375] = hmatch['flux_ivar_w1'][hmatch['dec'] < 32.375]
# w1fe = np.copy(hmatch['flux_ivar_w1'])
# w1fe[hmatch['dec'] < 32.375] = hmatch['flux_w1'][hmatch['dec'] < 32.375]
#
# w2f = np.copy(hmatch['flux_w2'])
# w2f[hmatch['dec'] < 32.375] = hmatch['flux_ivar_w2'][hmatch['dec'] < 32.375]
# w2fe = np.copy(hmatch['flux_ivar_w2'])
# w2fe[hmatch['dec'] < 32.375] = hmatch['flux_w2'][hmatch['dec'] < 32.375]
#
# w3f = np.copy(hmatch['flux_w3'])
# w3f[hmatch['dec'] < 32.375] = hmatch['flux_ivar_w3'][hmatch['dec'] < 32.375]
# w3fe = np.copy(hmatch['flux_ivar_w3'])
# w3fe[hmatch['dec'] < 32.375] = hmatch['flux_w3'][hmatch['dec'] < 32.375]
#
# w4f = np.copy(hmatch['flux_w4'])
# w4f[hmatch['dec'] < 32.375] = hmatch['flux_ivar_w4'][hmatch['dec'] < 32.375]
# w4fe = np.copy(hmatch['flux_ivar_w4'])
# w4fe[hmatch['dec'] < 32.375] = hmatch['flux_w4'][hmatch['dec'] < 32.375]
#
# hmatch['flux_w1'] = w1f
# hmatch['flux_ivar_w1'] = w1fe
#
# hmatch['flux_w2'] = w2f
# hmatch['flux_ivar_w2'] = w2fe
#
# hmatch['flux_w3'] = w3f
# hmatch['flux_ivar_w3'] = w3fe
#
# hmatch['flux_w4'] = w4f
# hmatch['flux_ivar_w4'] = w4fe

#help_specz['brick_primary'].dtype='bool'
#help_specz.keep_columns(dr14q_north.colnames)

# all_specz = vstack([dr14q_north, dr14q_south, help_specz], join_type='inner')
#
# all_specz = vstack([help_specz[:], help_specz[:10]], join_type='inner')

str_x = np.array([sdss_stars['lupt_g']-sdss_stars['lupt_r'],
                  sdss_stars['lupt_r']-sdss_stars['lupt_z'],
                  sdss_stars['lupt_z']-sdss_stars['lupt_w1'],
                  sdss_stars['lupt_w1']-sdss_stars['lupt_w2'],
                  ]).T

sobs = (sdss_stars['nobs_g'] > 0) * (sdss_stars['nobs_r'] > 0) * (sdss_stars['nobs_z'] > 0)
sobs *= (sdss_stars['lupterr_w1'] < 0.51) * (sdss_stars['lupterr_z'] < 0.51) * (sdss_stars['type'] == 'PSF ')
str_x = str_x[sobs,:]

good = ((np.isinf(str_x).sum(1) + np.isnan(str_x).sum(1)) == 0)
str_x = str_x[good,:]

qso_x = np.array([dr14['lupt_g']-dr14['lupt_r'],
                  dr14['lupt_r']-dr14['lupt_z'],
                  dr14['lupt_z']-dr14['lupt_w1'],
                  dr14['lupt_w1']-dr14['lupt_w2'],
                  ]).T

qobs = (dr14['nobs_g'] > 0) * (dr14['nobs_r'] > 0) * (dr14['nobs_z'] > 0)
qobs *= (dr14['type'] == 'PSF ')
qso_x = qso_x[qobs,:]

good = ((np.isinf(qso_x).sum(1) + np.isnan(qso_x).sum(1)) == 0)
qso_x = qso_x[good,:]

# Fig, Ax = plt.subplots(1,3,figsize=(9,3.5))
# Ax[0].hexbin(str_x[:,2], str_x[:,1], extent=[-1,3,-0.5,2], bins='log')
# Ax[1].hexbin(str_x[:,3], str_x[:,2], extent=[-3,2,-1,3], bins='log')
# Ax[2].hexbin(str_x[:,3], str_x[:,1], extent=[-3,2,-0.5,2], bins='log')
#
# Fig, Ax = plt.subplots(1,3,figsize=(9,3.5))
# Ax[0].hexbin(qso_x[:,2], qso_x[:,1], extent=[-1,3,-0.5,2], bins='log')
# Ax[1].hexbin(qso_x[:,3], qso_x[:,2], extent=[-3,2,-1,3], bins='log')
# Ax[2].hexbin(qso_x[:,3], qso_x[:,1], extent=[-3,2,-0.5,2], bins='log')
#
# plt.show()

gmm_stars = GaussianMixture(n_components=20,
                            max_iter=500).fit(str_x)
gmm_gal = GaussianMixture(n_components=20,
                            max_iter=500).fit(qso_x)

# gmm_stars = BayesianGaussianMixture(n_components=10, weight_concentration_prior=1e-2,
#                                     max_iter=500).fit(str_x)
# gmm_gal = BayesianGaussianMixture(n_components=10, weight_concentration_prior=1e-2,
#                                     max_iter=500).fit(qso_x)


# sky_sample_paths = glob.glob('tractor/gmmsample/*.fits')
# sky_sample = vstack([Table.read(cat) for cat in sky_sample_paths[::]])
#
# sky_sample = sky_sample[sky_sample['brick_primary'] == True]
#
# for i, morph in enumerate(['COMP', 'DEV ']): #, 'EXP ', 'PSF ', 'REX ']):
#     subset = (sky_sample['type'] == morph)
#     morph = morph.strip()
#     if i > 1:
#         sky_sample[subset][::2].write(f'tractor/gmmsample/tractor_morph_{morph}.fits',
#                                       format='fits', overwrite=True)
#     else:
#         sky_sample[subset].write(f'tractor/gmmsample/tractor_morph_{morph}.fits',
#                                  format='fits', overwrite=True)


# ls_pop, _ = load_legacy(sky_sample, offsets='legacy_north_south_offset.pkl')
# obs = (ls_pop['nobs_g'] > 0) * (ls_pop['nobs_r'] > 0) * (ls_pop['nobs_z'] > 0)
#
# ls_str = np.array([ls_pop['lupt_g']-ls_pop['lupt_r'],
#                    ls_pop['lupt_r']-ls_pop['lupt_z'],
#                    ls_pop['lupt_z']-ls_pop['lupt_w1'],
#                    ls_pop['lupt_w1']-ls_pop['lupt_w2'],
#                    ]).T
#
# good = ((np.isinf(ls_str).sum(1) + np.isnan(ls_str).sum(1)) == 0)
# ls_str = ls_str[good,:]
#
# ls_star_d = gmm_stars.score_samples(ls_str)
# ls_gal_d = gmm_gal.score_samples(ls_str)
# star_prob = np.exp(ls_star_d) / (np.exp(ls_star_d) + np.exp(ls_gal_d))
#
# parallax_snr = np.abs(ls_pop['parallax'] * np.sqrt(ls_pop['parallax_ivar']))
# likely_star = (ls_pop['type'][good] == 'PSF ') * np.logical_or(parallax_snr[good] > 3, star_prob > 0.9)
#
# ls_pop = ls_pop[obs * good][np.invert(likely_star)]

# Fig, Ax = plt.subplots(1,1)
# Ax.hist(star_prob, bins=50, density=False, label='All', histtype='step', lw=2)
# Ax.hist(star_prob[parallax_snr > 3], bins=50, density=False, lw=2,
#         histtype='step', label=r'Parallax $>3\sigma$')
# Leg = Ax.legend(loc='upper left')
# Ax.set_ylabel('N')
# #Ax.set_yscale('log')
# Ax.set_xlabel('P(Star)')
# plt.show()


"""
Global Settings
"""
NTOTBF = 500
NCOMP = 15
CSL = 'weight'
GMM_THRESHOLD = 0.5
INCSIZE = 'size'

COV = 'gpvd'
GPZPATH = '/disk2/kdun/photoz/gpzpp/bin/gpz++'
OUTDIR = 'gmm{0}_n{1}_{2}_{3}'.format(GMM_THRESHOLD, NCOMP, CSL, INCSIZE)

makedir(OUTDIR)

morph_params = Params('gpz.sz.psf.param')
morph_params['TRAIN_VALID_RATIO'] = 0.75

all_bash = open('{0}/merged_sz_subset_runscript.sh'.format(OUTDIR), 'w')

for morph in ['PSF ']:
    makedir('{0}/{1}'.format(OUTDIR, morph.strip()))

    print(morph)

    """
    Load reference population
    """
    ls_pop, _ = load_legacy(f'tractor/gmmsample/tractor_morph_{morph.strip()}.fits',
                            offsets='legacy_north_south_offset.pkl')
    obs = (ls_pop['nobs_g'] > 0) * (ls_pop['nobs_r'] > 0) * (ls_pop['nobs_z'] > 0)

    ls_str = np.array([ls_pop['lupt_g']-ls_pop['lupt_r'],
                       ls_pop['lupt_r']-ls_pop['lupt_z'],
                       ls_pop['lupt_z']-ls_pop['lupt_w1'],
                       ls_pop['lupt_w1']-ls_pop['lupt_w2'],
                       ]).T

    good = ((np.isinf(ls_str).sum(1) + np.isnan(ls_str).sum(1)) == 0)
    ls_str = ls_str[good,:]

    ls_star_d = gmm_stars.score_samples(ls_str)
    ls_gal_d = gmm_gal.score_samples(ls_str)
    star_prob = np.exp(ls_star_d) / (np.exp(ls_star_d) + np.exp(ls_gal_d))

    parallax_snr = np.abs(ls_pop['parallax'] * np.sqrt(ls_pop['parallax_ivar']))
    likely_star = (ls_pop['type'][good] == 'PSF ') * np.logical_or(parallax_snr[good] > 3, star_prob > 0.9)

    ls_pop = ls_pop[obs * good][np.invert(likely_star)]


    """
    HELP Compilation
    """
    subsample = (help_specz['ANYMASK_OPT'] == False) * (help_specz['t1_z_qual'] >= 3) * (help_specz['z_spec'] <= 5.)
    subsample *= (help_specz['fracflux_g'] < 0.2) * (help_specz['fracflux_r'] < 0.2) * (help_specz['fracflux_z'] < 0.2)
    subsample *= (help_specz['fracmasked_g'] < 0.1) * (help_specz['fracmasked_r'] < 0.1) * (help_specz['fracmasked_z'] < 0.1)
    subsample *= (help_specz['type'] == morph.strip()) * (help_specz['z_spec'] > 0.001)
    #subsample *= (help_specz['dec'] > 32.75)
    subcat = help_specz[subsample]

    Cols = ['ls_id', 'z_spec', 'lupt_g', 'lupterr_g',
            'lupt_r', 'lupterr_r', 'lupt_z', 'lupterr_z',
            'lupt_w1', 'lupterr_w1', 'lupt_w2', 'lupterr_w2',
            'lupt_w3', 'lupterr_w3', 'lupt_w4', 'lupterr_w4']

    subcat.keep_columns(Cols)
    keep = np.invert(np.array(subcat.to_pandas().isna().sum(1), dtype=bool))
    subcat = subcat[keep]

    sz_str = np.array([subcat['lupt_g']-subcat['lupt_r'],
                       subcat['lupt_r']-subcat['lupt_z'],
                       subcat['lupt_z']-subcat['lupt_w1'],
                       subcat['lupt_w1']-subcat['lupt_w2'],
                       ]).T

    good = ((np.isinf(sz_str).sum(1) + np.isnan(sz_str).sum(1)) == 0)
    sz_str = sz_str[good,:]

    sz_star_d = gmm_stars.score_samples(sz_str)
    sz_gal_d = gmm_gal.score_samples(sz_str)
    star_prob = np.exp(sz_star_d) / (np.exp(sz_star_d) + np.exp(sz_gal_d))

    keep = (star_prob < 0.2)
    subcat = subcat[good][keep]

    """
    SDSS Compilation
    """
    subsample = (dr14['ANYMASK_OPT'] == False)
    subsample *= (dr14['fracflux_g'] < 0.2) * (dr14['fracflux_r'] < 0.2) * (dr14['fracflux_z'] < 0.2)
    subsample *= (dr14['fracmasked_g'] < 0.1) * (dr14['fracmasked_r'] < 0.1) * (dr14['fracmasked_z'] < 0.1)
    subsample *= (dr14['type'] == morph) * (dr14['z_spec'] > 0.001)
    subcat2 = dr14[subsample]

    Cols = ['ls_id',
            'z_spec', 'lupt_g', 'lupterr_g',
            'lupt_r', 'lupterr_r', 'lupt_z', 'lupterr_z',
            'lupt_w1', 'lupterr_w1', 'lupt_w2', 'lupterr_w2',
            'lupt_w3', 'lupterr_w3', 'lupt_w4', 'lupterr_w4']

    subcat2.keep_columns(Cols)
    keep = np.invert(np.array(subcat2.to_pandas().isna().sum(1), dtype=bool))
    subcat2 = subcat2[keep]

    """
    VHzQ Compilation
    """
    subsample = (vhzq_specz['ANYMASK_OPT'] == False)
    subsample *= (vhzq_specz['fracflux_g'] < 0.2) * (vhzq_specz['fracflux_r'] < 0.2) * (vhzq_specz['fracflux_z'] < 0.2)
    subsample *= (vhzq_specz['fracmasked_g'] < 0.1) * (vhzq_specz['fracmasked_r'] < 0.1) * (vhzq_specz['fracmasked_z'] < 0.1)
    subsample *= (vhzq_specz['type'] == morph.strip()) * (vhzq_specz['z_spec'] > 0.001)
    subcat3 = vhzq_specz[subsample]

    Cols = ['ls_id', 'z_spec', 'lupt_g', 'lupterr_g',
            'lupt_r', 'lupterr_r', 'lupt_z', 'lupterr_z',
            'lupt_w1', 'lupterr_w1', 'lupt_w2', 'lupterr_w2',
            'lupt_w3', 'lupterr_w3', 'lupt_w4', 'lupterr_w4']

    subcat3.keep_columns(Cols)
    keep = np.invert(np.array(subcat3.to_pandas().isna().sum(1), dtype=bool))
    subcat3 = subcat3[keep]

    merged = vstack([subcat, subcat2, subcat3])

    lsp_x = np.array([ls_pop['lupt_z'],
                      ls_pop['lupt_g']-ls_pop['lupt_r'],
                      ls_pop['lupt_r']-ls_pop['lupt_z'],
                      ls_pop['lupt_z']-ls_pop['lupt_w1'],
                      ]).T

    good = ((np.isinf(lsp_x).sum(1) + np.isnan(lsp_x).sum(1)) == 0)
    lsp_x = lsp_x[good,:]

    gmm_x = np.array([merged['lupt_z'],
                      merged['lupt_g']-merged['lupt_r'],
                      merged['lupt_r']-merged['lupt_z'],
                      merged['lupt_z']-merged['lupt_w1'],
                      ]).T

    good = ((np.isinf(gmm_x).sum(1) + np.isnan(gmm_x).sum(1)) == 0)
    gmm_x = gmm_x[good,:]
    merged = merged[good]

    gmm_ls = GaussianMixture(n_components=NCOMP, #weight_concentration_prior=1e-2,
                             max_iter=500).fit(lsp_x)
    gmm_spec = GaussianMixture(n_components=NCOMP, #weight_concentration_prior=1e-2,
                             max_iter=500).fit(gmm_x)

    # gmm_ls = BayesianGaussianMixture(n_components=20, weight_concentration_prior=1e-2,
    #                                     max_iter=500).fit(lsp_x[ls_morph])
    # gmm_spec = BayesianGaussianMixture(n_components=40, weight_concentration_prior=1e-2,
    #                                     max_iter=500).fit(gmm_x)

    pi_spec = np.exp(gmm_spec.score_samples(gmm_x))
    pi_sample = np.exp(gmm_ls.score_samples(gmm_x))

    weight = (pi_sample+0.001) / (pi_spec+0.001)

    merged['weight'] = np.minimum(weight, 100)
    morph_params['COVARIANCE'] = COV

    if CSL == 'weight':
        morph_params['WEIGHT_COLUMN'] = 'weight'
    else:
        morph_params['WEIGHT_COLUMN'] = ''
        morph_params['WEIGHTING_SCHEME'] = 'uniform'

    bash_script = open('{0}/{1}/merged_sz_subset_{2}_runscript.sh'.format(OUTDIR, morph.strip(), morph.strip()), 'w')

    index = np.arange(len(merged))
    i_train, i_test = train_test_split(index, test_size=0.2)

    train = merged[i_train]
    train_path = '{0}/{1}/merged_sz_subset_train_{2}.txt'.format(OUTDIR, morph.strip(), morph.strip())
    train.write(train_path,
               format='ascii.commented_header', overwrite=True)

    test= merged[i_test]
    test_path = '{0}/{1}/merged_sz_subset_test_{2}.txt'.format(OUTDIR, morph.strip(), morph.strip())
    test.write(test_path,
              format='ascii.commented_header', overwrite=True)

    output_cat =  '{0}/{1}/merged_sz_subset_test_{2}_output.txt'.format(OUTDIR, morph.strip(), morph.strip())
    output_model = '{0}/{1}/merged_sz_subset_{2}_model.dat'.format(OUTDIR, morph.strip(), morph.strip())
    morph_params['TRAINING_CATALOG'] = os.path.abspath(train_path)
    morph_params['PREDICTION_CATALOG'] = os.path.abspath(test_path)
    morph_params['OUTPUT_CATALOG'] = os.path.abspath(output_cat)
    morph_params['MODEL_FILE'] = os.path.abspath(output_model)
    morph_params['NUM_BF'] = int(NTOTBF)

    param_path = '{0}/{1}/merged_sz_subset_{2}.param'.format(OUTDIR, morph.strip(), morph.strip())
    morph_params.write(param_path)
    #bash_script.write('{0} {1}\n'.format(GPZPATH, os.path.abspath(param_path)))
    #all_bash.write('{0} {1}\n'.format(GPZPATH, os.path.abspath(param_path)))

    mmodels = gmm_ls.predict(gmm_x)
    mprob = gmm_ls.predict_proba(gmm_x)

    morph_params['NUM_BF'] = int(NTOTBF / NCOMP)

    for mx in np.arange(gmm_spec.n_components):
        modelcut = (mprob[:,mx] > GMM_THRESHOLD)
        merged['gmm_prob'] = mprob[:,mx]

        index = np.arange(modelcut.sum())
        i_train, i_test = train_test_split(index, test_size=0.2)

        train = merged[modelcut][i_train]
        train_path = '{0}/{1}/merged_sz_subset_m{2}_train_{3}.txt'.format(OUTDIR, morph.strip(), mx, morph.strip())
        train.write(train_path,
                   format='ascii.commented_header', overwrite=True)
        #
        test = merged[modelcut][i_test]
        test_path = '{0}/{1}/merged_sz_subset_m{2}_test_{3}.txt'.format(OUTDIR, morph.strip(), mx, morph.strip())
        test.write(test_path,
                   format='ascii.commented_header', overwrite=True)

        output_cat =  '{0}/{1}/merged_sz_subset_m{2}_test_{3}_output.txt'.format(OUTDIR, morph.strip(), mx, morph.strip())
        output_model = '{0}/{1}/merged_sz_subset_m{2}_{3}_model.dat'.format(OUTDIR, morph.strip(), mx, morph.strip())
        morph_params['TRAINING_CATALOG'] = os.path.abspath(train_path)
        morph_params['PREDICTION_CATALOG'] = os.path.abspath(test_path)
        morph_params['OUTPUT_CATALOG'] = os.path.abspath(output_cat)
        morph_params['MODEL_FILE'] = os.path.abspath(output_model)

        param_path = '{0}/{1}/merged_sz_subset_m{2}_{3}.param'.format(OUTDIR, morph.strip(), mx, morph.strip())
        morph_params.write(param_path)
        bash_script.write('{0} {1}\n'.format(GPZPATH, os.path.abspath(param_path)))
        all_bash.write('{0} {1}\n'.format(GPZPATH, os.path.abspath(param_path)))

    bash_script.close()

morph_params = Params('gpz.sz.dev.param')
morph_params['TRAIN_VALID_RATIO'] = 0.75

for morph in ['DEV ', 'EXP ', 'REX ', 'COMP']:
    makedir('{0}/{1}'.format(OUTDIR, morph.strip()))

    ls_pop, _ = load_legacy(f'tractor/gmmsample/tractor_morph_{morph.strip()}.fits',
                            offsets='legacy_north_south_offset.pkl')

    print(morph)
    """
    HELP Compilation
    """
    subsample = (help_specz['ANYMASK_OPT'] == False) * (help_specz['t1_z_qual'] >= 3)
    subsample *= (help_specz['fracflux_g'] < 0.2) * (help_specz['fracflux_r'] < 0.2) * (help_specz['fracflux_z'] < 0.2)
    subsample *= (help_specz['fracmasked_g'] < 0.1) * (help_specz['fracmasked_r'] < 0.1) * (help_specz['fracmasked_z'] < 0.1)
    subsample *= (help_specz['type'] == morph.strip()) * (help_specz['z_spec'] > 0.001)
    #subsample *= (help_specz['dec'] > 32.75)
    subcat = help_specz[subsample]

    Cols = ['ls_id', 'z_spec', 'lupt_g', 'lupterr_g',
            'lupt_r', 'lupterr_r', 'lupt_z', 'lupterr_z',
            'lupt_w1', 'lupterr_w1', 'lupt_w2', 'lupterr_w2',
            'lupt_w3', 'lupterr_w3', 'lupt_w4', 'lupterr_w4']

    if morph in ['DEV ']:
        subcat['lupt_s'] = subcat['shapedev_r']
        subcat['lupterr_s'] = 1/np.sqrt(subcat['shapedev_r_ivar'])

        Cols.append('lupt_s')
        Cols.append('lupterr_s')

    elif morph in ['EXP ' , 'REX ', 'COMP']:
        subcat['lupt_s'] = subcat['shapeexp_r']
        subcat['lupterr_s'] = 1/np.sqrt(subcat['shapeexp_r_ivar'])

        Cols.append('lupt_s')
        Cols.append('lupterr_s')

    subcat.keep_columns(Cols)
    keep = np.invert(np.array(subcat.to_pandas().isna().sum(1), dtype=bool))
    subcat = subcat[keep]

    """
    SDSS Compilation
    """
    subsample = (dr14['ANYMASK_OPT'] == False)
    subsample *= (dr14['type'] == morph) * (dr14['z_spec'] > 0.001)
    subcat2 = dr14[subsample]

    Cols = ['ls_id', 'z_spec', 'lupt_g', 'lupterr_g',
            'lupt_r', 'lupterr_r', 'lupt_z', 'lupterr_z',
            'lupt_w1', 'lupterr_w1', 'lupt_w2', 'lupterr_w2',
            'lupt_w3', 'lupterr_w3', 'lupt_w4', 'lupterr_w4']

    if morph in ['DEV ']:
        subcat2['lupt_s'] = subcat2['shapedev_r']
        subcat2['lupterr_s'] = 1/np.sqrt(subcat2['shapedev_r_ivar'])

        ls_pop['lupt_s'] = ls_pop['shapedev_r']
        ls_pop['lupterr_s'] = 1/np.sqrt(ls_pop['shapedev_r_ivar'])

        Cols.append('lupt_s')
        Cols.append('lupterr_s')

    elif morph in ['EXP ', 'REX ', 'COMP']:
        subcat2['lupt_s'] = subcat2['shapeexp_r']
        subcat2['lupterr_s'] = 1/np.sqrt(subcat2['shapeexp_r_ivar'])

        ls_pop['lupt_s'] = ls_pop['shapeexp_r']
        ls_pop['lupterr_s'] = 1/np.sqrt(ls_pop['shapeexp_r_ivar'])

        Cols.append('lupt_s')
        Cols.append('lupterr_s')

    subcat2.keep_columns(Cols)
    keep = np.invert(np.array(subcat2.to_pandas().isna().sum(1), dtype=bool))
    subcat2 = subcat2[keep]

    merged = vstack([subcat, subcat2])

    lsp_x = np.array([ls_pop['lupt_z'],
                      ls_pop['lupt_g']-ls_pop['lupt_r'],
                      ls_pop['lupt_r']-ls_pop['lupt_z'],
                      ls_pop['lupt_s']]).T

    obs = ((ls_pop['nobs_g'] > 0) * (ls_pop['nobs_r'] > 0) * (ls_pop['nobs_z'] > 0))

    good = ((np.isinf(lsp_x).sum(1) + np.isnan(lsp_x).sum(1)) == 0)

    if morph in ['EXP ', 'DEV ', 'COMP']:
        bad = np.logical_or((lsp_x[:,0] > 22.) * (np.abs(lsp_x[:,1]) > 2.), (np.abs(lsp_x[:,2]) > 3.))
        obs *= np.invert(bad) * (lsp_x[:,0] >  np.percentile(lsp_x[good,0], 0.1))

    lsp_x = lsp_x[good*obs,:]

    gmm_x = np.array([merged['lupt_z'],
                      merged['lupt_g']-merged['lupt_r'],
                      merged['lupt_r']-merged['lupt_z'],
                      merged['lupt_s']]).T

    good = ((np.isinf(gmm_x).sum(1) + np.isnan(gmm_x).sum(1)) == 0)
    gmm_x = gmm_x[good,:]
    merged = merged[good]

    gmm_ls = GaussianMixture(NCOMP).fit(lsp_x)
    gmm_spec = GaussianMixture(NCOMP).fit(gmm_x)

    pi_spec = np.exp(gmm_spec.score_samples(gmm_x))
    pi_sample = np.exp(gmm_ls.score_samples(gmm_x))

    weight = (pi_sample+0.001) / (pi_spec+0.001)

    merged['weight'] = np.minimum(weight, 100)
    morph_params['COVARIANCE'] = COV

    if CSL == 'weight':
        morph_params['WEIGHT_COLUMN'] = 'weight'
    else:
        morph_params['WEIGHT_COLUMN'] = ''
        morph_params['WEIGHTING_SCHEME'] = 'uniform'

    bash_script = open('{0}/{1}/merged_sz_subset_{2}_runscript.sh'.format(OUTDIR, morph.strip(), morph.strip()), 'w')

    index = np.arange(len(merged))
    i_train, i_test = train_test_split(index, test_size=0.2)

    train = merged[i_train]
    train_path = '{0}/{1}/merged_sz_subset_train_{2}.txt'.format(OUTDIR, morph.strip(), morph.strip())
    train.write(train_path,
               format='ascii.commented_header', overwrite=True)

    test= merged[i_test]
    test_path = '{0}/{1}/merged_sz_subset_test_{2}.txt'.format(OUTDIR, morph.strip(), morph.strip())
    test.write(test_path,
              format='ascii.commented_header', overwrite=True)

    output_cat =  '{0}/{1}/merged_sz_subset_test_{2}_output.txt'.format(OUTDIR, morph.strip(), morph.strip())
    output_model = '{0}/{1}/merged_sz_subset_{2}_model.dat'.format(OUTDIR, morph.strip(), morph.strip())
    morph_params['TRAINING_CATALOG'] = os.path.abspath(train_path)
    morph_params['PREDICTION_CATALOG'] = os.path.abspath(test_path)
    morph_params['OUTPUT_CATALOG'] = os.path.abspath(output_cat)
    morph_params['MODEL_FILE'] = os.path.abspath(output_model)
    morph_params['NUM_BF'] = int(NTOTBF)

    param_path = '{0}/{1}/merged_sz_subset_{2}.param'.format(OUTDIR, morph.strip(), morph.strip())
    morph_params.write(param_path)
    #bash_script.write('{0} {1}\n'.format(GPZPATH, os.path.abspath(param_path)))
    #all_bash.write('{0} {1}\n'.format(GPZPATH, os.path.abspath(param_path)))

    mmodels = gmm_ls.predict(gmm_x)
    mprob = gmm_ls.predict_proba(gmm_x)

    morph_params['NUM_BF'] = int(NTOTBF / NCOMP)

    for mx in np.arange(gmm_ls.n_components):
        modelcut = (mprob[:,mx] > GMM_THRESHOLD)
        merged['gmm_prob'] = mprob[:,mx]
        index = np.arange(modelcut.sum())
        i_train, i_test = train_test_split(index, test_size=0.2)

        train = merged[modelcut][i_train]
        train_path = '{0}/{1}/merged_sz_subset_m{2}_train_{3}.txt'.format(OUTDIR, morph.strip(), mx, morph.strip())
        train.write(train_path,
                   format='ascii.commented_header', overwrite=True)
        #
        test = merged[modelcut][i_test]
        test_path = '{0}/{1}/merged_sz_subset_m{2}_test_{3}.txt'.format(OUTDIR, morph.strip(), mx, morph.strip())
        test.write(test_path,
                   format='ascii.commented_header', overwrite=True)

        output_cat = '{0}/{1}/merged_sz_subset_m{2}_test_{3}_output.txt'.format(OUTDIR, morph.strip(), mx, morph.strip())
        output_model = '{0}/{1}/merged_sz_subset_m{2}_{3}_model.dat'.format(OUTDIR, morph.strip(), mx, morph.strip())
        morph_params['TRAINING_CATALOG'] = os.path.abspath(train_path)
        morph_params['PREDICTION_CATALOG'] = os.path.abspath(test_path)
        morph_params['OUTPUT_CATALOG'] = os.path.abspath(output_cat)
        morph_params['MODEL_FILE'] = os.path.abspath(output_model)

        param_path = '{0}/{1}/merged_sz_subset_m{2}_{3}.param'.format(OUTDIR, morph.strip(), mx, morph.strip())
        morph_params.write(param_path)
        bash_script.write('{0} {1}\n'.format(GPZPATH, os.path.abspath(param_path)))
        all_bash.write('{0} {1}\n'.format(GPZPATH, os.path.abspath(param_path)))

    bash_script.close()
all_bash.close()
