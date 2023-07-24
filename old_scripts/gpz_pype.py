import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column, vstack
import astropy.units as u
from astropy.utils.console import human_time
import os
import glob
import pickle
import time
import healpy
from astropy.coordinates import SkyCoord

from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from collections import OrderedDict
from utilities import *

from subprocess import call
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

"""
Global Settings
"""
ls_hem = 'south'

NTOTBF = 500
NCOMP = 10
CSL = 'weight'
GMM_THRESHOLD = 0.5
INCSIZE = 'size'

GPZPATH = '/disk2/kdun/photoz/gpzpp/bin/gpz++'
GPZDIR = 'production/gmm{0}_n{1}_{2}_{3}'.format(GMM_THRESHOLD, NCOMP, CSL, INCSIZE)

TOT_THREADS = 24
MP_THREADS = 1

CLEANFILES = True

brick_table = Table.read("survey-bricks.fits.gz")

brick_table_north = Table.read("survey-bricks-dr8-north.fits.gz")
brick_table_south = Table.read("survey-bricks-dr8-south.fits.gz")

cond_north = ((brick_table_north['nexp_g'] > 0) *
              (brick_table_north['nexp_r'] > 0) *
              (brick_table_north['nexp_z'] > 0))

cond_south = ((brick_table_south['nexp_g'] > 0) *
              (brick_table_south['nexp_r'] > 0) *
              (brick_table_south['nexp_z'] > 0))

brick_table_north = brick_table_north[cond_north]
brick_table_south = brick_table_south[cond_south]

brick_table_north['hpx_o3_id'] = healpy.ang2pix(2**3, brick_table_north['ra'],
                                                brick_table_north['dec'], lonlat=True)
brick_table_south['hpx_o3_id'] = healpy.ang2pix(2**3, brick_table_south['ra'],
                                                brick_table_south['dec'], lonlat=True)

unique_north = np.unique(brick_table_north['hpx_o3_id'])
unique_south = np.unique(brick_table_south['hpx_o3_id'])

download_names = brick_table["BRICKNAME"]

north_bricks = download_names[np.isin(download_names, brick_table_north["brickname"])]
south_bricks = download_names[np.isin(download_names, brick_table_south["brickname"])]


hpx_ra, hpx_dec = healpy.pix2ang(2**3, unique_south, lonlat=True)
hpx_crd = SkyCoord(hpx_ra, hpx_dec, unit='deg')
hpx_gal = hpx_crd.galactic

lotss_0hr = (hpx_gal.b < 0.) * (hpx_dec > 10.)

hpx_numd = np.concatenate((unique_south[lotss_0hr], unique_south[np.invert(lotss_0hr)]))

#folders = glob.glob(f'tractor/{ls_hem}/hpx_*')
#folders.sort()

#hpx_numd = np.array([int(f.split('_')[-1]) for f in folders])
#hpx_numd.sort()

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
    likely_star = ((catalog['type'] == 'PSF ') *
                    np.logical_or(parallax_snr > 3, star_prob > 0.5))

    return star_prob, parallax_snr, likely_star

# Define functions
def rungpz(command):
    output = call(command, shell=True)
    return output

makedir(f'{GPZDIR}/output')

morphologies = ['DEV ', 'EXP ', 'REX ', 'COMP', 'PSF ']

for i, hpx in enumerate(hpx_numd[112:]):
    hpx_start = time.time()

    print(f'---------- HPX #{hpx} ----------')
    tractor_files = glob.glob(f'tractor/{ls_hem}/hpx_{hpx}/tractor*')
    tractor_files.sort()
    tractor = vstack([Table.read(path, format='fits') for path in tractor_files[:]])

    pzpath = f'{GPZDIR}/output/hpx_{hpx}'
    makedir(pzpath)

    full_catalog, _ = load_legacy(tractor)
    full_catalog = full_catalog[full_catalog['brick_primary']]

    star_prob, parallax_snr, likely_star = star_gal_sep(full_catalog, GPZDIR)

    full_catalog['pstar'] = star_prob
    full_catalog['star'] = np.array(likely_star, dtype='bool')

    size = full_catalog['shapeexp_r']
    size_err = 1/np.sqrt(full_catalog['shapeexp_r_ivar'])

    size[full_catalog['type'] == 'DEV '] = full_catalog['shapedev_r'][full_catalog['type'] == 'DEV ']
    size_err[full_catalog['type'] == 'DEV '] = 1/np.sqrt(full_catalog['shapedev_r_ivar'])[full_catalog['type'] == 'DEV ']

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

    for im, morph in enumerate(morphologies[:]):
        morph_start = time.time()

        if morph == 'PSF ':
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
            subset = full_catalog[morph_set*good][models == ic]
            subset.keep_columns(Cols)
            if len(subset) > 0:
                subset['gmmcomp'] = f'{morph[0]}{ic}'
                subset_cat = f'{pzpath}/hpx_{hpx}_{morph.strip()}_m{ic}.txt'
                subset.write(subset_cat,
                             format='ascii.commented_header', overwrite=True)

                morph_params = Params(f'{GPZDIR}/{morph.strip()}/merged_sz_subset_m{ic}_{morph.strip()}_gpvc.param')
                morph_params['TRAINING_CATALOG'] = ''
                morph_params['PREDICTION_CATALOG'] = os.path.abspath(subset_cat)
                morph_params['OUTPUT_CATALOG'] = os.path.abspath(f'{pzpath}/hpx_{hpx}_{morph.strip()}_m{ic}_output.txt')
                morph_params['VERBOSE'] = 0
                morph_params['N_THREAD'] = int(TOT_THREADS/MP_THREADS)
                morph_params['REUSE_MODEL'] = 1
                morph_params['USE_MODEL_AS_HINT'] = 0

                param_path = f'{pzpath}/hpx_{hpx}_{morph.strip()}_m{ic}.param'
                morph_params.write(param_path)

                gpz_commands.append('{0} {1}'.format(GPZPATH, os.path.abspath(param_path)))
                comps_used.append(ic)

        pool = mp.Pool(MP_THREADS)
        gpz_out = pool.map(rungpz, gpz_commands)
        pool.terminate()

        incats = vstack([Table.read(f'{pzpath}/hpx_{hpx}_{morph.strip()}_m{ic}.txt',
                                    format='ascii.commented_header') for ic in comps_used])
        outcats = vstack([Table.read(f'{pzpath}/hpx_{hpx}_{morph.strip()}_m{ic}_output.txt',
                                    format='ascii.commented_header', header_start=10) for ic in comps_used])

        merged_cat = incats[Cols]
        merged_cat['gmmcomp'] = incats['gmmcomp']
        merged_cat['zphot'] = outcats['value']
        merged_cat['zphot_err'] = outcats['uncertainty']
        merged_cat.add_column(outcats['var.density'])
        merged_cat.add_column(outcats['var.tr.noise'])
        merged_cat.add_column(outcats['var.in.noise'])

        with open(f'{GPZDIR}/alphas_gmm{GMM_THRESHOLD}_n{NCOMP}_{CSL}.pkl', 'rb') as file:
            alphas_dict = pickle.load(file)
        merged_cat['zphot_err'] *= alphas_mag(merged_cat['lupt_r'], *alphas_dict[morph.strip()])

        merged_cat.write(f'{pzpath}/hpx_{hpx}_{morph.strip()}.fits', format='fits', overwrite=True)

        if CLEANFILES:
            tempfiles = glob.glob(f'{pzpath}/hpx_{hpx}_{morph.strip()}_m*')
            for t in tempfiles:
                os.remove(t)

        print(f'{morph.strip()}: Nsources = {len(merged_cat)}, Time = {human_time(time.time()-morph_start)}')

    print(f'HPX Complete - Time = {human_time(time.time()-hpx_start)}')
    print(f'--------------------------------')

    # full_output = vstack([Table.read(f'{pzpath}/hpx_{hpx}_{morph.strip()}.fits') for
    #                       morph in morphologies])
