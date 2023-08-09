import os
import copy
import logging
import numpy as np

# gpz_pype class
from .utilities import Params
from .gmm import GMMbasic

from sklearn.model_selection import train_test_split


class GPz(object):
    """
    Class to run GPz.
    """

    def __init__(self, param_file=None, gpz_path=None, ncpu=None):
        """
        Parameters
        ----------
        param_file : str, optional
            Path to parameter file to read.

        gpz_path : str, optional
            Path to gpz++ installation. If not specified, will look for GPZPATH environment variable.
        
        ncpu : int, optional
            Number of CPUs for gpz++ to use.  If not specified, will use all available CPUs.
        """

        # Read parameter file
        if param_file != None:
            self.read(param_file)

        # Set GPz path
        if gpz_path is not None:
            if os.path.exists(gpz_path):
                self.gpz_path = gpz_path
            else:
                raise ValueError("GPz path does not exist.")
        elif gpz_path is None:
            try:
                self.gpz_path = os.environ['GPZPATH']
            except:
                raise ValueError("GPZPATH environment variable not set.")

        # Set number of CPUs to use
        if ncpu is None:
            self.ncpu = os.cpu_count()

        elif ncpu is not None:
            if ncpu > os.cpu_count():
                logging.warning("Number of CPUs requested is greater than number of available CPUs.  Using all available CPUs.")
                ncpu = os.cpu_count()

            self.ncpu = ncpu

    def read(self, param_file):
        """
        Read parameter file.
        
        Parameters
        ----------
        param_file : str
            Path to parameter file to read.
            
        """

        self.params = Params(param_file)

    def prep_gpz(self, catalog, outdir, basename, label=None,
                mag_prefix='mag_', error_prefix='magerr_', z_col='z_spec',
                weight_col=None, output_min=0, output_max=7, 
                test_fraction=0.2, valid_fraction=0.2, 
                do_iteration=False, iter_cov= 'gpvc', basis_functions=50):
        """
        Prepare gpz++ input catalogues and parameter files 
        for a given input catalogue.
        
        Parameters
        ----------
        catalog : astropy.table.Table
            Catalog to run GPz on.
        
        outdir : str
            Output directory.
        
        basename : str
            Basename for output files.
        
        label : str, optional
            Label to append to output files.

        mag_prefix : str, optional
            Prefix for magnitude columns.  Default is 'mag_'.
        
        error_prefix : str, optional
            Prefix for magnitude error columns.  Default is 'magerr_'.
            
        z_col : str, optional
            Name of spectroscopic redshift column.  Default is 'z_spec'.

        weight_col : str, optional
            Name of weight column.  Default is None.
        
        output_min : float, optional
            Minimum redshift to output.  Default is 0.
        
        output_max : float, optional
            Maximum redshift to output.  Default is 7.

        test_fraction : float, optional
            Fraction of catalog to use for testing.  Default is 0.2.
        
        valid_fraction : float, optional
            Fraction of training set to use for validation.  Default is 0.2.

        do_iteration : bool, optional
            If True, will set up 2nd gpz++ iteration with a more complex covariance. 
            Default is False.

        iter_cov : str, optional
            Covariance type to use for 2nd iteration.  Default is 'gpvc'.

        Returns
        -------
        run_string : str
            Command to run gpz++ for the input catalog.

        file_paths : list
            List of paths to files generated for the input catalog.

        """

        run_params = copy.deepcopy(self.params)

        run_params['FLUX_COLUMN_PREFIX'] = mag_prefix
        run_params['ERROR_COLUMN_PREFIX'] = error_prefix

        filters = [filt[len(mag_prefix):] for filt in catalog.colnames if filt.startswith(mag_prefix)]
        run_params['BANDS'] = f'^{mag_prefix}({"|".join(filters).lower()})$'
        run_params['OUTPUT_COLUMN'] = z_col
        run_params['OUTPUT_MIN'] = output_min
        run_params['OUTPUT_MAX'] = output_max

        if weight_col is not None:
            run_params['WEIGHT_COLUMN'] = weight_col
        else:
            run_params['WEIGHT_COLUMN'] = ''

        if label is None:
            rootname = f'{outdir}/{basename}'
        else:
            rootname = f'{outdir}/{basename}_{label}'

        # Run GPz
        index = np.arange(len(catalog))
        i_train, i_test = train_test_split(index, test_size=test_fraction)

        train_path = f'{rootname}_train.txt'
        catalog[i_train].write(train_path, format='ascii.commented_header', overwrite=True)

        test_path = f'{rootname}_test.txt'
        catalog[i_test].write(test_path, format='ascii.commented_header', overwrite=True)

        output_cat =  f'{rootname}_output.txt'
        output_model = f'{rootname}_model.dat'

        run_params['TRAIN_VALID_RATIO'] =  (1-(test_fraction+valid_fraction)) / (1-test_fraction)
        run_params['TRAINING_CATALOG'] = os.path.abspath(train_path)
        run_params['PREDICTION_CATALOG'] = os.path.abspath(test_path)
        run_params['OUTPUT_CATALOG'] = os.path.abspath(output_cat)
        run_params['MODEL_FILE'] = os.path.abspath(output_model)
        run_params['NUM_BF'] = int(basis_functions)

        param_path = f'{rootname}.param'
        run_params.write(param_path)

        run_string = f'{self.gpz_path} {os.path.abspath(param_path)}'

        if do_iteration:
            run_params['COVARIANCE'] = iter_cov
            run_params['USE_MODEL_AS_HINT'] = '1'
            iter_param_path = f'{rootname}_{iter_cov}.param'
            run_params.write(iter_param_path)

            run_string += '\n'+f'{self.gpz_path} {os.path.abspath(iter_param_path)}'

        file_paths = [train_path, test_path, output_cat, output_model]

        return run_string, file_paths


