import os
import pickle
from collections import OrderedDict

import astropy.units as u
import numpy as np
from astropy.table import Column, Table, vstack

f0 = (3631 * u.Jy).to(u.uJy).value

nanomaggy_to_ujy = 10 ** ((23.9 - 22.5) / 2.5)
nanovega_to_ujy_w1 = 10 ** ((23.9 - 2.699 - 22.5) / 2.5)
nanovega_to_ujy_w2 = 10 ** ((23.9 - 3.339 - 22.5) / 2.5)

import contextlib
import sys
from time import sleep

from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        # sys.stdout = sys.stderr = DummyTqdmFile(orig_out_err[0])
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


def makedir(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)


def save_gmm(gmm, path):
    with open(path, "wb") as file:
        pickle.dump(gmm, file)


def load_gmm(path):
    with open(path, "rb") as file:
        gmm = pickle.load(file)
    return gmm


def alphas_mag(mags, intcp, slope, base=14.0):
    alt_mags = np.clip(mags, base, 35.0)
    alphas = intcp + (alt_mags - base) * slope
    return alphas


def set_gpz_path(path):
    """
    Set the GPz path.
    """
    import os

    if os.path.exists(path):
        os.environ["GPZPATH"] = path
    else:
        raise ValueError("GPz path does not exist.")

    return None


class Params(object):
    """
    Class to read and write gpz++ parameter files.
    """

    def __init__(self, input_path=None):
        """
        Parameters
        ----------
        input_path : str, optional
            Path to parameter file to read.
        """

        if input_path != None:
            self.read(input_path)

    def read(self, file):
        """
        Read parameter file.

        Parameters
        ----------
        file : str
            Path to parameter file to read.

        """

        with open(file, "r") as param:
            lines = param.readlines()
            self.params = OrderedDict()

            for line in lines:
                if line[0] not in ["#", "\n"]:
                    try:
                        parline = line.strip().split("#")[0]
                    except(IndexError):
                        parline = line
                    parts = parline.strip().split("=")
                    self.params[parts[0].strip()] = parts[1].strip()

    def write(self, file=None):
        """
        Write parameter file.

        Parameters
        ----------
        file : str, optional
            Path to parameter file to write.  If not specified, will print to screen.
        """
        if file == None:
            print("No output file specified...")
        else:
            output = open(file, "w")
            for param in self.params.keys():
                if param[0] != "#":
                    output.write(
                        "{0:30s} = {1}\n".format(param, self.params[param])
                    )
                    # str = '%-25s %'+self.formats[param]+'\n'
            #
            output.close()

    def __getitem__(self, param_name):
        """
        Get item from ``params`` dict.

        Parameters
        ----------
        param_name : str
            Name of parameter to get.

        Returns
        -------
        param : str
            Value of parameter.

        """
        if param_name not in self.params.keys():
            print(
                "Column {0} not found.  Check `column_names` attribute.".format(
                    param_name
                )
            )
            return None
        else:
            return self.params[param_name]

    def __setitem__(self, param_name, value):
        self.params[param_name] = value


def flux_to_lupt(flux, fluxerr, b):
    """Calculate asinh magnitudes 'luptitudes' for given fluxes.

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
    lupt = (-2.5 / np.log(10)) * (
        np.arcsinh((flux / f0) / (2 * b)) + np.log(b)
    )
    lupterr = (
        (2.5 / np.log(10))
        * np.abs(fluxerr / flux)
        / np.sqrt(1 + (2 * b / (flux / f0)) ** 2)
    )
    return lupt, lupterr


def basic_lupt_soft(flux, flux_err, unit=u.uJy, f0=3631 * u.Jy, scale=1.05):
    try:
        f0 = f0.to(unit)
    except(u.UnitConversionError):
        raise

    snr = flux / flux_err
    snr_cut = (snr > 4) * (snr < 8)

    if flux_err.unit:
        try:
            rms_err = (np.nanmedian(flux_err[snr_cut])).to(unit)
        except (AttributeError, u.UnitConversionError):
            raise

    else:
        # Assume units equal to value
        rms_err = (np.nanmedian(flux_err[snr_cut]) * unit).to(unit)

    return scale * (rms_err.value / f0.value)
