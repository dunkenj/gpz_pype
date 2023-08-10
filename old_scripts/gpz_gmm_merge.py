import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from collections import OrderedDict

from astropy.table import Table, Column, vstack, join, Row
from astropy.utils.console import ProgressBar
from astropy.coordinates import SkyCoord
import glob
import itertools
from scipy import linalg
from utilities import *


NTOTBF = 500
NCOMP = 10
CSL = 'weight'
GMM_THRESHOLD = 0.5
INCSIZE = 'size'

GPZPATH = '/disk2/kdun/photoz/gpzpp/bin/gpz++'
GPZDIR = 'production/gmm{0}_n{1}_{2}_{3}'.format(GMM_THRESHOLD, NCOMP, CSL, INCSIZE)

folders = glob.glob('production/gmm0.5_n10_weight_size/output_north/hpx_*')
folders.sort()

morphs = ['PSF', 'REX', 'EXP', 'DEV', 'COMP']

with ProgressBar(len(folders)) as bar:
    for i, f in enumerate(folders[:]):
        root, folder = os.path.split(f)
        hpx = int(folder.split('_')[1])
        #print(f'HPX: {hpx}')
        hpx_merged = []

        hpx_dict = OrderedDict()
        hpx_dict['hpx'] = hpx

        for im, morph in enumerate(morphs[:]):
            #print(morph)
            cat = Table.read(f'{root}/hpx_{hpx}/hpx_{hpx}_{morph}.fits')

            with open(f'{GPZDIR}/alphas_gmm{GMM_THRESHOLD}_n{NCOMP}_{CSL}_v2.pkl', 'rb') as file:
                alphas_dict = pickle.load(file)

            new_err = np.sqrt(cat['var.density'] +  cat['var.tr.noise'] +
                              cat['var.in.noise'])
            new_err *= alphas_mag(cat['lupt_r'], *alphas_dict[morph.strip()])

            g_flag = np.logical_or(cat['fracflux_g'] < 0.33, cat['lupterr_g'] > 0.543)
            r_flag = np.logical_or(cat['fracflux_r'] < 0.33, cat['lupterr_r'] > 0.543)
            z_flag = np.logical_or(cat['fracflux_z'] < 0.33, cat['lupterr_z'] > 0.543)
            maskflag = (cat['maskbits'] == 0)

            clean = g_flag * r_flag * z_flag * maskflag
            reliable = ((new_err/(1+cat['zphot']) < 0.2) * clean)

            if morph == 'PSF':
                reliable *= (cat['pstar'] < 0.2)

            hpx_dict[f'{morph.lower()}_all'] = len(cat)
            hpx_dict[f'{morph.lower()}_reliable'] = reliable.sum()
            hpx_dict = dict(hpx_dict)

            cat['zphot_err'] = new_err
            cat['flag_clean'] = np.array(clean, dtype='int')
            cat['flag_qual'] = np.array(reliable, dtype='int')

            hpx_merged.append(cat)

        merged_cat = vstack(hpx_merged)
        merged_cat.sort('id')
        merged_cat.write(f'{root}/north_merged/hpx_{hpx:03}_merged.fits',
                         overwrite=True)

        if i == 0:
            summary_table = Table(names=list(hpx_dict.keys()),
                                  dtype=[np.int64]*len(hpx_dict.keys()))
            summary_table.add_row(vals=list(hpx_dict.values()))
        else:
            summary_table.add_row(vals=list(hpx_dict.values()))
        bar.update()





stop

summary_table.sort('hpx')

total_all = np.array([summary_table[f'{morph.lower()}_all'].sum() for morph
                      in morphs])
total_reliable = np.array([summary_table[f'{morph.lower()}_reliable'].sum()
                           for morph in morphs])

centers = np.array(healpy.pix2ang(2**3, summary_table['hpx'], lonlat=True)).T
corners = np.array([healpy.vec2ang(healpy.boundaries(2**3, [x]).T, lonlat=True)
                    for x in summary_table['hpx']])
vertices = np.append(corners, corners[:,:,0][:,:,None], axis=2)

psf_frac = summary_table['psf_reliable']/summary_table['psf_all']
cols = plt.cm.viridis(psf_frac)

dev_frac = summary_table['exp_reliable']/summary_table['exp_all']
cols = plt.cm.viridis(dev_frac)

ra, dec = healpy.pix2ang(2**3, list(summary_table['hpx']), lonlat=True)
coords = SkyCoord(ra, dec, unit='deg')


Fig, Ax = plt.subplots(1,1)
Ax.plot(np.abs(coords.galactic.b[:-4]),
        (summary_table['psf_reliable']/summary_table['psf_all'])[:-4],
        'o')
Ax.plot(np.abs(coords.galactic.b[:-4]),
        (summary_table['rex_reliable']/summary_table['rex_all'])[:-4],
        'o')
Ax.plot(np.abs(coords.galactic.b[:-4]),
        (summary_table['exp_reliable']/summary_table['exp_all'])[:-4],
        'o')
Ax.plot(np.abs(coords.galactic.b[:-4]),
        (summary_table['dev_reliable']/summary_table['dev_all'])[:-4],
        'o')

Fig = plt.figure(figsize=(11,6))
Ax = Fig.add_subplot(111, projection='mollweide')
Ax.grid(True)
org = 180

for i, hpx in enumerate(summary_table['hpx'][:-4]):
    vert_x = vertices[i][0]
    vert_y = vertices[i][1]

    x = np.remainder(vert_x+360-org,360) # shift RA values
    ind = x>180
    x[ind] -=360    # scale conversion to [-180, 180]
    x=-x    # reverse the scale: East to the left

    cent_x = np.remainder(centers[i][0]+360-org, 360)
    if cent_x > 180:
        cent_x -=360
    cent_x = -cent_x

    if cent_x == 0:
        vert_x = x
        vert_x[vert_x > 300] = 0.
    elif cent_x > 350.:
        vert_x = x
        vert_x[vert_x < 300] = 360.
    else:
        vert_x = x

    Ax.fill(np.radians(vert_x), np.radians(vert_y), color=cols[i], alpha=0.9, zorder=10)
