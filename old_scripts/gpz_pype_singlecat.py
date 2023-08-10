import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column, vstack
import astropy.units as u
from astropy.utils.console import human_time
import os
import glob
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from collections import OrderedDict
from utilities import *

from subprocess import call
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")


def star_gal_sep(catalog, gpzdir):
    catalog = full_catalog
    gpzdir = GPZDIR
    gmm_star = load_gmm(f'{gpzdir}/stargal_gmm_star.pkl')
    gmm_qso = load_gmm(f'{gpzdir}/stargal_gmm_qso.pkl')
    gmm_gal = load_gmm(f'{gpzdir}/stargal_gmm_gal.pkl')

    col_arr = np.array([catalog['lupt_g']-catalog['lupt_r'],
                        catalog['lupt_r']-catalog['lupt_z'],
                        catalog['lupt_z']-catalog['lupt_w1'],
                        catalog['lupt_w1']-catalog['lupt_w2']]).T

    good = ((np.isinf(col_arr).sum(1) + np.isnan(col_arr).sum(1)) == 0)

    star_d = gmm_star.score_samples(col_arr[good,:])
    qso_d = gmm_qso.score_samples(col_arr[good,:])
    gal_d = gmm_gal.score_samples(col_arr[good,:])

    star_prob = np.nan*np.ones(len(catalog))
    star_prob[good] = np.exp(star_d) / (np.exp(star_d) + np.exp(gal_d) + np.exp(qso_d))

    parallax_snr = np.abs(catalog['parallax'] * np.sqrt(catalog['parallax_ivar']))
    likely_star = ((catalog['type'] == 'PSF') *
                    np.logical_or(parallax_snr > 3, star_prob > 0.5))

    return star_prob, parallax_snr, likely_star


def fixwise(catalog):
    for i in range(1,5):
        f = np.copy(catalog[f'flux_w{i}'])
        fe = np.copy(catalog[f'flux_ivar_w{i}'])

        deccut = (catalog['dec'] < 32.375)
        catalog[f'flux_w{i}'][deccut] = fe[deccut]
        catalog[f'flux_ivar_w{i}'][deccut] = f[deccut]
    return catalog

# Define functions
def rungpz(command):
    output = call(command, shell=True)
    return output



"""
Global Settings
"""
ls_hem = 'north'

NTOTBF = 500
NCOMP = 10
CSL = 'weight'
GMM_THRESHOLD = 0.5
INCSIZE = 'size'

GPZPATH = '/disk2/kdun/photoz/gpzpp/bin/gpz++'
GPZDIR = 'production/gmm{0}_n{1}_{2}_{3}'.format(GMM_THRESHOLD, NCOMP, CSL, INCSIZE)

TOT_THREADS = 20
MP_THREADS = 1

CLEANFILES = True
WISEFIX = True

input_catalog = 'ancillary/foexsess_lsdr8_photz.txt'
pzpath = 'ancillary/xray/foexsess'
makedir(pzpath)


hpx_start = time.time()

print(f'---------- Catalog ----------')

tractor = Table.read(input_catalog, format='ascii.csv')
if WISEFIX:
    tractor = fixwise(tractor)

full_catalog, _ = load_legacy(tractor)

star_prob, parallax_snr, likely_star = star_gal_sep(full_catalog, GPZDIR)

full_catalog['pstar'] = star_prob
full_catalog['star'] = np.array(likely_star, dtype='bool')

size = full_catalog['shapeexp_r']
size_err = 1/np.sqrt(full_catalog['shapeexp_r_ivar'])

size[full_catalog['type'] == 'DEV'] = full_catalog['shapedev_r'][full_catalog['type'] == 'DEV']
size_err[full_catalog['type'] == 'DEV'] = 1/np.sqrt(full_catalog['shapedev_r_ivar'])[full_catalog['type'] == 'DEV']

full_catalog['lupt_s'] = size
full_catalog['lupterr_s'] = size_err

Cols = ['id', 'release', 'brickid', 'objid',
        'maskbits',
        'fracflux_g', 'fracflux_r', 'fracflux_z',
        'type', 'ra', 'dec', 'pstar', 'star',
        'lupt_g', 'lupterr_g',
        'lupt_r', 'lupterr_r', 'lupt_z', 'lupterr_z',
        'lupt_w1', 'lupterr_w1', 'lupt_w2', 'lupterr_w2',
        'lupt_w3', 'lupterr_w3', 'lupt_w4', 'lupterr_w4',
        'lupt_s', 'lupterr_s', 'ANYMASK_OPT']

morphologies = ['PSF', 'DEV', 'EXP', 'REX', 'COMP']

for im, morph in enumerate(morphologies[:]):
    morph_start = time.time()

    if morph == 'PSF':
        col_arr = np.array([full_catalog['lupt_z'],
                            full_catalog['lupt_g']-full_catalog['lupt_r'],
                            full_catalog['lupt_r']-full_catalog['lupt_z'],
                            full_catalog['lupt_z']-full_catalog['lupt_w1']]).T
    else:
        col_arr = np.array([full_catalog['lupt_z'],
                            full_catalog['lupt_g']-full_catalog['lupt_r'],
                            full_catalog['lupt_r']-full_catalog['lupt_z'],
                            full_catalog['lupt_s']]).T

    good = ((np.isinf(col_arr).sum(1) + np.isnan(col_arr).sum(1)) == 0)

    gmm_morph = load_gmm(f'{GPZDIR}/{morph.strip()}/refpop_gmm_{morph.strip()}.pkl')
    ncomp = gmm_morph.n_components

    morph_set = (full_catalog['type'] == morph)
    models = gmm_morph.predict(col_arr[good*morph_set,:])

    gpz_commands = []

    comps_used = []
    for ic in range(ncomp)[:]:

        subset = full_catalog[np.where(morph_set*good)[0]][models == ic]

        if len(subset) > 0:
            subset.keep_columns(Cols)

            subset['gmmcomp'] = f'{morph[0]}{ic}'
            subset_cat = f'{pzpath}/{morph.strip()}_m{ic}.txt'
            subset.write(subset_cat,
                         format='ascii.commented_header', overwrite=True)

            morph_params = Params(f'{GPZDIR}/{morph.strip()}/merged_sz_subset_m{ic}_{morph.strip()}_gpvc.param')
            morph_params['TRAINING_CATALOG'] = ''
            morph_params['PREDICTION_CATALOG'] = os.path.abspath(subset_cat)
            morph_params['OUTPUT_CATALOG'] = os.path.abspath(f'{pzpath}/{morph.strip()}_m{ic}_output.txt')
            morph_params['VERBOSE'] = 0
            morph_params['N_THREAD'] = int(TOT_THREADS/MP_THREADS)
            morph_params['REUSE_MODEL'] = 1
            morph_params['USE_MODEL_AS_HINT'] = 0

            param_path = f'{pzpath}/{morph.strip()}_m{ic}.param'
            morph_params.write(param_path)

            gpz_commands.append('{0} {1}'.format(GPZPATH, os.path.abspath(param_path)))
            comps_used.append(ic)

    pool = mp.Pool(MP_THREADS)
    gpz_out = pool.map(rungpz, gpz_commands)
    pool.terminate()

    incats = vstack([Table.read(f'{pzpath}/{morph.strip()}_m{ic}.txt',
                                format='ascii.commented_header') for ic in comps_used])
    outcats = vstack([Table.read(f'{pzpath}/{morph.strip()}_m{ic}_output.txt',
                                format='ascii.commented_header', header_start=10) for ic in comps_used])

    merged_cat = incats[Cols]
    merged_cat['gmmcomp'] = incats['gmmcomp']
    merged_cat['zphot'] = outcats['value']
    merged_cat['zphot_err'] = outcats['uncertainty']
    merged_cat.add_column(outcats['var.density'])
    merged_cat.add_column(outcats['var.tr.noise'])
    merged_cat.add_column(outcats['var.in.noise'])

    with open(f'{GPZDIR}/alphas_gmm{GMM_THRESHOLD}_n{NCOMP}_{CSL}_v2.pkl', 'rb') as file:
        alphas_dict = pickle.load(file)
    merged_cat['zphot_err'] *= alphas_mag(merged_cat['lupt_r'], *alphas_dict[morph.strip()])

    merged_cat.write(f'{pzpath}/{morph.strip()}.fits', format='fits', overwrite=True)

    if CLEANFILES:
        tempfiles = glob.glob(f'{pzpath}/{morph.strip()}_m*')
        for t in tempfiles:
            os.remove(t)

    print(f'{morph.strip()}: Nsources = {len(merged_cat)}, Time = {human_time(time.time()-morph_start)}')

print(f'Catalog Complete - Time = {human_time(time.time()-hpx_start)}')
print(f'--------------------------------')

full_output = vstack([Table.read(f'{pzpath}/{morph.strip()}.fits') for
                      morph in morphologies])



# plt.errorbar(full_output[highz*fraccut]['zphot'], full_output[highz*fraccut]['lupt_z'],
#              yerr=full_output[highz*fraccut]['lupterr_z'], xerr=full_output[highz*fraccut]['zphot_err'], fmt='o')
#
# plt.errorbar(full_output[highz*fraccut]['zphot'], full_output[highz*fraccut]['lupt_r']-full_output[highz*fraccut]['lupt_z'],
#              xerr=full_output[highz*fraccut]['zphot_err'], fmt='o')


#
# cols = [
#      't1_ilt_comp_name',
#      'id',
#      'ra',
#      'dec',
#      'type',
#      'maskbits',
#      'lupt_g',
#      'lupterr_g',
#      'lupt_r',
#      'lupterr_r',
#      'fluxerr_z',
#      'lupt_z',
#      'lupterr_z',
#      'lupt_w1',
#      'lupterr_w1',
#      'lupt_w2',
#      'lupterr_w2',
#      'lupt_w3',
#      'lupterr_w3',
#      'lupt_w4',
#      'lupterr_w4',
#      'pstar',
#      'star',
#      'gmmcomp',
#      'zphot',
#      'zphot_err']
