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

#####
f0 = (3631*u.Jy).to(u.uJy).value
nanomaggy_to_ujy = 10**((23.9-22.5)/2.5)
nanovega_to_ujy_w1 = 10**((23.9-2.699-22.5)/2.5)
nanovega_to_ujy_w2 = 10**((23.9-3.339-22.5)/2.5)

def makedir(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)

def save_gmm(gmm, path):
    with open(path, 'wb') as file:
        pickle.dump(gmm, file)

def load_gmm(path):
    with open(path, 'rb') as file:
        gmm = pickle.load(file)
    return gmm

def alphas_mag(mags, intcp, slope, base = 14.):
    alt_mags = np.clip(mags, base, 35.)
    alphas = intcp + (alt_mags - base)*slope
    return alphas

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

    sweep['id'] = np.array(sweep['release']*1e12 + sweep['brickid']*1e6 + sweep['objid'], dtype='int')
    sweep['ANYMASK_OPT'] = ((sweep['anymask_g'] + sweep['anymask_r'] + sweep['anymask_z']) > 0)
    sweep['brick_primary'].dtype='bool'

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
