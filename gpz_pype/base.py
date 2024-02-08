"""
Python interface to gpz++. 

This module includes the following classes:

- GPz: Core class for preparing and running gpz++ on a given catalog.


"""

import contextlib
import copy
import logging
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np
from astropy.table import Table, vstack
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .gmm import GMMbasic

# gpz_pype class
from .utilities import Params, std_out_err_redirect_tqdm


class DummyFile(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout


class GPz(object):
    """
    Class to run GPz.

    This class is used to prepare and run gpz++ on a given catalog.
    Options for running gpz++ are set in a parameter file, which can
    be read in using the read() method.  Alternatively, options can
    be set directly using the class attributes.

    The run_training() method can be used either to run gpz++ on a
    single catalogue, or to automatically split a catalogue into
    mixture samples for training and testing based on a given
    GMM model (see gpz_pype.gmm.GMMbasic).

    gpz++ can be called directly from within the class or for large
    catalogues, the necessary run commands can be written to a shell
    script.

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
                self.gpz_path = os.environ["GPZPATH"]
            except:
                raise ValueError("GPZPATH environment variable not set.")

        # Set number of CPUs to use
        if ncpu is None:
            self.ncpu = os.cpu_count()

        elif ncpu is not None:
            if ncpu > os.cpu_count():
                logging.warning(
                    "Number of CPUs requested is greater than number of available CPUs.  Using all available CPUs."
                )
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

    def prep_gpz(
        self,
        catalog,
        outdir,
        basename,
        mode="train",
        model=None,
        label=None,
        mag_prefix="mag_",
        error_prefix="magerr_",
        z_col="z_spec",
        weight_col=None,
        output_min=0,
        output_max=7,
        test_fraction=0.2,
        valid_fraction=0.2,
        do_iteration=False,
        iter_cov="gpvc",
        basis_functions=50,
        max_iter = 500,
        tol = 1e-9,
        grad_tol = 1e-5,
    ):
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
        mode : str, optional
            Mode to run GPz in.  Default is 'train'. Other options are 'predict'.
        model : str, optional
            Path to GPz model file.  Default is None.
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
        basis_functions : int, optional
            Number of basis functions to use.  Default is 50.
        max_iter : int, optional
            Maximum number of iterations.  Default is 500.
        tol : float, optional
            Tolerance for convergence.  Default is 1e-9.
        grad_tol : float, optional 
            Tolerance for gradient convergence.  Default is 1e-5.

        Returns
        -------
        run_string : str
            Command to run gpz++ for the input catalog.
        file_paths : list
            List of paths to files generated for the input catalog.

        """

        run_params = copy.deepcopy(self.params)

        run_params["FLUX_COLUMN_PREFIX"] = mag_prefix
        run_params["ERROR_COLUMN_PREFIX"] = error_prefix

        filters = [
            filt[len(mag_prefix) :]
            for filt in catalog.colnames
            if filt.startswith(mag_prefix)
        ]
        run_params["BANDS"] = f'^{mag_prefix}({"|".join(filters).lower()})$'
        run_params["OUTPUT_COLUMN"] = z_col
        run_params["OUTPUT_MIN"] = output_min
        run_params["OUTPUT_MAX"] = output_max

        run_params["MAX_ITER"] = max_iter
        run_params["TOLERANCE"] = tol
        run_params["GRAD_TOLERANCE"] = grad_tol

        if weight_col is not None:
            run_params["WEIGHT_COLUMN"] = weight_col
        else:
            run_params["WEIGHT_COLUMN"] = ""

        if label is None:
            rootname = f"{outdir}/{basename}"
        else:
            rootname = f"{outdir}/{basename}_{label}"

        if mode == "train":
            run_params["REUSE_MODEL"] = "0"

            # Run GPz
            index = np.arange(len(catalog))
            i_train, i_test = train_test_split(index, test_size=test_fraction)

            train_path = f"{rootname}_train.txt"
            catalog[i_train].write(
                train_path, format="ascii.commented_header", overwrite=True
            )

            test_path = f"{rootname}_test.txt"
            catalog[i_test].write(
                test_path, format="ascii.commented_header", overwrite=True
            )

            output_cat = f"{rootname}_output.txt"
            output_model = f"{rootname}_model.dat"

            if os.path.exists(output_model):
                logging.warning(f"Removing existing model file {output_model}")
                os.remove(output_model)
                os.remove(output_cat)

            run_params["TRAIN_VALID_RATIO"] = (
                1 - (test_fraction + valid_fraction)
            ) / (1 - test_fraction)
            run_params["TRAINING_CATALOG"] = os.path.abspath(train_path)
            run_params["PREDICTION_CATALOG"] = os.path.abspath(test_path)
            run_params["OUTPUT_CATALOG"] = os.path.abspath(output_cat)
            run_params["MODEL_FILE"] = os.path.abspath(output_model)
            run_params["NUM_BF"] = int(basis_functions)

        elif mode == "predict":
            output_model = model
            run_params["REUSE_MODEL"] = "1"
            run_params["SAVE_MODEL"] = "0"

            # Run GPz
            predict_path = f"{rootname}_predict.txt"
            catalog.write(
                predict_path, format="ascii.commented_header", overwrite=True
            )

            output_cat = f"{rootname}_prediction_output.txt"

            run_params["TRAINING_CATALOG"] = ""
            run_params["PREDICTION_CATALOG"] = os.path.abspath(predict_path)
            run_params["OUTPUT_CATALOG"] = os.path.abspath(output_cat)
            run_params["MODEL_FILE"] = os.path.abspath(output_model)

        param_path = f"{rootname}.param"
        run_params.write(param_path)

        run_string = f"{self.gpz_path} {os.path.abspath(param_path)}"

        if mode == "train" and do_iteration:
            run_params["COVARIANCE"] = iter_cov
            run_params["USE_MODEL_AS_HINT"] = "1"
            iter_param_path = f"{rootname}_{iter_cov}.param"
            run_params.write(iter_param_path)

            run_string += (
                "\n" + f"{self.gpz_path} {os.path.abspath(iter_param_path)}"
            )

        file_paths = {
            "output_cat": output_cat,
            "output_model": output_model,
            "param": param_path,
        }

        if mode == "train":
            file_paths["test"] = test_path
            file_paths["train"] = train_path
            
        elif mode == "predict":
            file_paths["predict"] = predict_path
            try:
                self.paths["predict"] = predict_path
            except:
                self.paths = {"predict": predict_path}

        if do_iteration:
            file_paths["iter_param"] = iter_param_path

        return run_string, file_paths

    def run_training(
        self,
        catalog,
        outdir,
        basename,
        gmm_output=None,
        bash_script=False,
        mag_prefix="mag_",
        error_prefix="magerr_",
        z_col="z_spec",
        id_col="id",
        weight_col=None,
        output_min=0,
        output_max=7,
        test_fraction=0.2,
        valid_fraction=0.2,
        do_iteration=False,
        iter_cov="gpvc",
        total_basis_functions=100,
        bf_distribution="uniform",
        min_basis_functions=10,
        max_iter = 500,
        tol = 1e-9,
        grad_tol = 1e-5,
        verbose=False,
        **kwargs,
    ):
        """
        Run GPz training and prediction for a given input catalogue.

        Parameters
        ----------
        catalog : astropy.table.Table, str
            Catalog to run GPz on. If str, will read in catalog from file.
        outdir : str
            Output directory.
        basename : str
            Basename for output files.
        gmm_output : astroy.table.Table
            GMM divide and/or weight output catalog.
        bash_script: bool
            If True, output gpz++ commands to a bash script instead of running.
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
        total_basis_functions : int, optional
            Total number of basis functions to use.  Default is 100.
        bf_distribution : str, optional
            Basis function distribution.  Default is 'uniform'.
            Other options are 'sqrt' and 'linear', which will distribute basis functions
            according to the square root and linear of the number of training objects in
            each mixture, respectively.
        min_basis_functions : int, optional
            Minimum number of basis functions per mixture.  Default is 10.
        max_iter : int, optional
            Maximum number of iterations.  Default is 500.
        tol : float, optional
            Tolerance for convergence.  Default is 1e-9.
        grad_tol : float, optional 
            Tolerance for gradient convergence.  Default is 1e-5.
        verbose : bool, optional
            If True, will print out GPz++ output.  Default is False.
        **kwargs : dict, optional
            Additional keyword arguments to pass to Table.read() if catalog is a
            filename.

        Returns
        -------
        merged_output : astropy.table.Table
            Merged output catalog.
        paths : dict
            Dictionary of paths to output files.

        """

        capture_output = not verbose

        if isinstance(catalog, Table):
            catalog = catalog.copy()
        elif isinstance(catalog, str):
            catalog = Table.read(catalog, **kwargs)

        catalog[id_col].name = "id"

        if gmm_output is None:
            run_string, paths = self.prep_gpz(
                catalog,
                outdir,
                basename,
                mode="train",
                mag_prefix=mag_prefix,
                error_prefix=error_prefix,
                z_col=z_col,
                weight_col=weight_col,
                output_min=output_min,
                output_max=output_max,
                test_fraction=test_fraction,
                valid_fraction=valid_fraction,
                do_iteration=do_iteration,
                iter_cov=iter_cov,
                basis_functions=total_basis_functions,
                max_iter = max_iter,
                tol = tol,
                grad_tol = grad_tol,
            )

            if bash_script:
                with open(f"{outdir}/{basename}.sh", "w") as f:
                    f.write(run_string + "\n")

            else:
                run = subprocess.run(
                    run_string,
                    shell=True,
                    capture_output=capture_output,
                )

                if run.returncode != 0 and not verbose:
                    logging.error(
                        f"GPz++ run failed with error: {run.stdout.decode()}"
                    )
                    return None

                elif run.returncode != 0 and verbose:
                    return None

                else:
                    logging.info(f"GPz++ run completed successfully.")

                    incat = Table.read(
                        paths["test"], format="ascii.commented_header"
                    )
                    outcat = Table.read(
                        paths["output_cat"],
                        format="ascii.commented_header",
                        header_start=10,
                    )

                    for col in outcat.colnames[1:]:
                        incat[col] = outcat[col]

                    self.paths = paths

                    return incat, paths

        elif isinstance(gmm_output, Table):
            if weight_col is not None:
                if (
                    weight_col in catalog.colnames
                    and weight_col in gmm_output.colnames
                ):
                    logging.info(
                        "Multiple weight columns found, using weights from input catalog."
                    )
                    weights = catalog[weight_col]

                elif (
                    weight_col in catalog.colnames
                    and weight_col not in gmm_output.colnames
                ):
                    logging.info("Using weights from input catalog.")
                    weights = catalog[weight_col]

                elif (
                    weight_col not in catalog.colnames
                    and weight_col in gmm_output.colnames
                ):
                    logging.info("Using weights from GMM output catalog.")
                    weights = gmm_output[weight_col]

                else:
                    logging.info(
                        "No weight column found, using equal weights."
                    )
                    weights = np.ones(len(catalog))

            catalog[weight_col] = weights

            mixtures = np.sort(np.unique(gmm_output["best"]))
            nmixtures = len(mixtures)

            if nmixtures == 1:
                run_string, paths = self.prep_gpz(
                    catalog,
                    outdir,
                    basename,
                    mag_prefix=mag_prefix,
                    error_prefix=error_prefix,
                    z_col=z_col,
                    weight_col=weight_col,
                    output_min=output_min,
                    output_max=output_max,
                    test_fraction=test_fraction,
                    valid_fraction=valid_fraction,
                    do_iteration=do_iteration,
                    iter_cov=iter_cov,
                    basis_functions=total_basis_functions,
                )
            else:
                output_capture = []
                path_dict = {}

                if bf_distribution == "uniform":
                    nbasis = np.ones(nmixtures) * (total_basis_functions // nmixtures)

                elif bf_distribution == "sqrt":
                    train_sizes = np.array([np.sum(gmm_output[f'm{i}']==True) for i in range(nmixtures)])
                    train_sqrt = np.sqrt(train_sizes)
                    train_sqrt /= np.sum(train_sqrt)

                    shared_basis = total_basis_functions - (nmixtures * min_basis_functions)
                    nbasis = np.round(train_sqrt * shared_basis) + min_basis_functions

                elif bf_distribution == "linear":
                    train_sizes = np.array([np.sum(gmm_output[f'm{i}']==True) for i in range(nmixtures)])
                    train_linear = train_sizes / np.sum(train_sizes)

                    shared_basis = total_basis_functions - (nmixtures * min_basis_functions)
                    nbasis = np.round(train_linear * shared_basis) + min_basis_functions
                else:
                    raise ValueError(f"Unknown basis function distribution {bf_distribution}")

                pbar = tqdm(
                    total=nmixtures,
                    file=sys.stdout,
                    dynamic_ncols=True,
                    desc=f"GPz++ Run",
                )

                with nostdout():
                    for i in range(nmixtures):
                        pbar.set_description(
                            f"GPz++ Run (mixture {i+1}/{nmixtures})"
                        )
                        mixture = mixtures[i]
                        mixture_cat = catalog[gmm_output[f'm{i}']==True]

                        run_string, paths = self.prep_gpz(
                            mixture_cat,
                            outdir,
                            basename,
                            label=f"m{mixture}",
                            mag_prefix=mag_prefix,
                            error_prefix=error_prefix,
                            z_col=z_col,
                            weight_col=weight_col,
                            output_min=output_min,
                            output_max=output_max,
                            test_fraction=test_fraction,
                            valid_fraction=valid_fraction,
                            do_iteration=do_iteration,
                            iter_cov=iter_cov,
                            basis_functions=nbasis[i],
                        )

                        if bash_script:
                            if i == 0:
                                with open(
                                    f"{outdir}/{basename}_m{mixture}.sh", "w"
                                ) as f:
                                    f.write(run_string + "\n")
                            else:
                                with open(
                                    f"{outdir}/{basename}_m{mixture}.sh", "a"
                                ) as f:
                                    f.write(run_string + "\n")

                        else:
                            run = subprocess.run(
                                run_string,
                                shell=True,
                                capture_output=capture_output,
                            )

                            if run.returncode != 0 and not verbose:
                                logging.error(
                                    f"GPz++ run failed with error: {run.stdout.decode()}"
                                )
                                return None

                            elif run.returncode != 0 and verbose:
                                return None

                            output_capture.append(run)
                            path_dict[mixture] = paths

                        pbar.update(1)

                pbar.close()

                if bash_script:
                    logging.info(f"GPz++ run scripts written to {outdir}.")
                    return None

                else:
                    logging.info(f"GPz++ run completed successfully.")

                    merged_output = []

                    for i, mixture in enumerate(mixtures):
                        incat = Table.read(
                            path_dict[mixture]["test"],
                            format="ascii.commented_header",
                        )
                        outcat = Table.read(
                            path_dict[mixture]["output_cat"],
                            format="ascii.commented_header",
                            header_start=10,
                        )

                        for col in outcat.colnames[1:]:
                            incat[col] = outcat[col]

                        merged_output.append(incat)

                    merged_output = vstack(merged_output)
                    merged_output.sort("id")

                    self.paths = path_dict

                    return merged_output, path_dict

    def predict(
        self,
        catalog,
        outdir,
        basename,
        gmm_output=None,
        bash_script=False,
        mag_prefix="mag_",
        error_prefix="magerr_",
        z_col="z_spec",
        id_col="id",
        weight_col=None,
        output_min=0,
        output_max=7,
        verbose=False,
        **kwargs,
    ):
        """
        Run GPz training and prediction for a given input catalogue.

        Parameters
        ----------
        catalog : astropy.table.Table, str
            Catalog to run GPz on. If str, will read in catalog from file.
        outdir : str
            Output directory.
        basename : str
            Basename for output files.
        gmm_output : astroy.table.Table
            GMM divide and/or weight output catalog.
        bash_script: bool
            If True, output gpz++ commands to a bash script instead of running.
        mag_prefix : str, optional
            Prefix for magnitude columns.  Default is 'mag_'.
        error_prefix : str, optional
            Prefix for magnitude error columns.  Default is 'magerr_'.
        z_col : str, optional
            Name of spectroscopic redshift column.  Default is 'z_spec'.
        weight_col : str, optional
            Name of weight column.  Default is None.
        verbose : bool, optional
            If True, will print out GPz++ output.  Default is False.
        **kwargs : dict, optional
            Additional keyword arguments to pass to Table.read()

        Returns
        -------
        incat : astropy.table.Table
            Input catalog with GPz predictions appended.
        paths : dict
            Dictionary of paths to output files.

        """

        capture_output = not verbose

        if isinstance(catalog, Table):
            catalog = catalog.copy()
        elif isinstance(catalog, str):
            catalog = Table.read(catalog, **kwargs)

        catalog[id_col].name = "id"

        rootname = f"{outdir}/{basename}"

        if gmm_output is None:
            model = self.paths["output_model"]

            run_string, paths = self.prep_gpz(
                catalog,
                outdir,
                basename,
                mode="predict",
                model=model,
                mag_prefix=mag_prefix,
                error_prefix=error_prefix,
                z_col=z_col,
                weight_col=weight_col,
                output_min=output_min,
                output_max=output_max,
            )

            if bash_script:
                with open(f"{outdir}/{basename}.sh", "w") as f:
                    f.write(run_string + "\n")

            else:
                run = subprocess.run(
                    run_string,
                    shell=True,
                    capture_output=capture_output,
                )

                if run.returncode != 0 and not verbose:
                    logging.error(
                        f"GPz++ run failed with error: {run.stdout.decode()}"
                    )
                    return None

                elif run.returncode != 0 and verbose:
                    return None

                else:
                    logging.info(f"GPz++ run completed successfully.")

                    incat = Table.read(
                        paths["predict"], format="ascii.commented_header"
                    )
                    outcat = Table.read(
                        paths["output_cat"],
                        format="ascii.commented_header",
                        header_start=10,
                    )

                    for col in outcat.colnames[1:]:
                        incat[col] = outcat[col]

                    return incat, paths

        elif isinstance(gmm_output, Table):
            if weight_col is not None:
                if (
                    weight_col in catalog.colnames
                    and weight_col in gmm_output.colnames
                ):
                    logging.info(
                        "Multiple weight columns found, using weights from input catalog."
                    )
                    weights = catalog[weight_col]

                elif (
                    weight_col in catalog.colnames
                    and weight_col not in gmm_output.colnames
                ):
                    logging.info("Using weights from input catalog.")
                    weights = catalog[weight_col]

                elif (
                    weight_col not in catalog.colnames
                    and weight_col in gmm_output.colnames
                ):
                    logging.info("Using weights from GMM output catalog.")
                    weights = gmm_output[weight_col]

                else:
                    logging.info(
                        "No weight column found, using equal weights."
                    )
                    weights = np.ones(len(catalog))

            catalog[weight_col] = weights

            mixtures = np.sort(np.unique(gmm_output["best"]))
            nmixtures = len(mixtures)

            if nmixtures == 1:
                model = self.paths[0]["output_model"]

                run_string, paths = self.prep_gpz(
                    catalog,
                    outdir,
                    basename,
                    mode="predict",
                    model=model,
                    mag_prefix=mag_prefix,
                    error_prefix=error_prefix,
                    z_col=z_col,
                    weight_col=weight_col,
                    output_min=output_min,
                    output_max=output_max,
                )
            else:
                output_capture = []
                path_dict = {}

                pbar = tqdm(
                    total=nmixtures,
                    file=sys.stdout,
                    dynamic_ncols=True,
                    desc=f"GPz++ Run",
                )

                with nostdout():
                    for i in range(nmixtures):
                        pbar.set_description(
                            f"GPz++ Run (mixture {i+1}/{nmixtures})"
                        )
                        mixture = mixtures[i]
                        mixture_cat = catalog[gmm_output["best"] == mixture]

                        model = self.paths[i]["output_model"]

                        run_string, paths = self.prep_gpz(
                            mixture_cat,
                            outdir,
                            basename,
                            label=f"m{mixture}",
                            mode="predict",
                            model=model,
                            mag_prefix=mag_prefix,
                            error_prefix=error_prefix,
                            z_col=z_col,
                            weight_col=weight_col,
                            output_min=output_min,
                            output_max=output_max,
                        )

                        if bash_script:
                            if i == 0:
                                with open(
                                    f"{outdir}/{basename}_m{mixture}.sh", "w"
                                ) as f:
                                    f.write(run_string + "\n")
                            else:
                                with open(
                                    f"{outdir}/{basename}_m{mixture}.sh", "a"
                                ) as f:
                                    f.write(run_string + "\n")

                        else:
                            run = subprocess.run(
                                run_string,
                                shell=True,
                                capture_output=capture_output,
                            )

                            if run.returncode != 0 and not verbose:
                                logging.error(
                                    f"GPz++ run failed with error: {run.stdout.decode()}"
                                )
                                return None

                            elif run.returncode != 0 and verbose:
                                return None

                            output_capture.append(run)
                            path_dict[mixture] = paths

                        pbar.update(1)

                pbar.close()

                if bash_script:
                    logging.info(f"GPz++ run scripts written to {outdir}.")
                    return None

                else:
                    logging.info(f"GPz++ run completed successfully.")

                    merged_output = []

                    for i, mixture in enumerate(mixtures):
                        incat = Table.read(
                            path_dict[mixture]["predict"],
                            format="ascii.commented_header",
                        )
                        outcat = Table.read(
                            path_dict[mixture]["output_cat"],
                            format="ascii.commented_header",
                            header_start=10,
                        )

                        for col in outcat.colnames[1:]:
                            incat[col] = outcat[col]

                        merged_output.append(incat)

                    merged_output = vstack(merged_output)
                    merged_output.sort("id")

                    return merged_output, path_dict
                
    def save(self, filename):
        """
        Save the GMM model to file.

        Parameters
        ----------
        filename : str
            Name of the file to save the GPz class.

        """
        import pickle

        data = {
            "params": self.params,
            "paths": self.paths,
            "gpz_path": self.gpz_path,
            "ncpu": self.ncpu,
        }

        # Save the GMM models
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load(self, filename):
        """
        Load the GMM model from file.

        Parameters
        ----------
        filename : str
            Name of the file to load the GPz class from.

        """
        import pickle

        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.params = data["params"]
        self.paths = data["paths"]
        self.gpz_path = data["gpz_path"]
        self.ncpu = data["ncpu"]