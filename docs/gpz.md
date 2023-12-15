
# Basic GPz++ usage


## Introduction
For this example notebook we will use the CANDELS COSMOS catalogue as an input. This includes lots of photometry from both ground + space, but we will restrict analysis to just the HST filters for simplicity.

Inputting the catalogue into GPz however requires some additional pre-processing. We start by extracting the general common columns and setting any missing `z_spec` values to `np.nan`:


```python
import numpy as np

# Plotting modules
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 100

# Astropy modules
from astropy.table import Table, join, vstack, hstack
from astropy.coordinates import SkyCoord
import astropy.units as u

# Import required gpz_pype functions
from gpz_pype.utilities import Params, set_gpz_path, basic_lupt_soft, flux_to_lupt
from gpz_pype.base import GPz

candels = Table.read('CANDELS.COSMOS.F160W.Processed.photz.fits', format='fits')

catalogue = candels[['ID', 'RA', 'DEC', 'CLASS_STAR', 'FLAGS', 'EBV', 'z_spec']]

catalogue['z_spec'][catalogue['z_spec'] < 0] = np.nan
```

To further trim the catalogue down, i next want to pull out only the flux/magnitude columns that belong to HST filters. In this case, columns that start with either 'ACS' or 'WFC'. As with `z_spec`, any value set as -99 in the input catalogue (i.e. no data) is converted to `np.nan`:


```python
for col in candels.colnames:
    if col.startswith('ACS') or col.startswith('WFC'):
        inst, filt, ext = col.split('_') # Split column name into 3 parts based on '_'
        col_vals = candels[col]
        col_vals[col_vals < -90] = np.nan
        catalogue[f'{ext}_{filt}'] = col_vals # [FLUX/FLUXERR]_[FILTER]
```

To extract the filters that were found in the catalogue, we reprocess the column names to look for filters. This could have been done above, but since there were both MAG/FLUX columns this would have resulted in duplicates. So this is simpler:


```python
filters = [col.split('_')[1] for col in catalogue.colnames if col.startswith('FLUX_')]
```

### Asinh magnitudes
Extensive past experience has shown that training machine learning using magnitudes can provide significantly better results than using linear flux values. This is particularly true for fields with very high dynamic range.

However, the significant drawback of normal AB magnitudes is that they cannot be used at low S/N or for negative fluxes (consistent with zero). For GPz we therefore make use of asinh magnitudes, which remain real+positive for very low S/N that remain informative for non-detections without the need for additional missing value prediction etc. This is particularly key for high-redshift where the non-detection at bluer wavelengths is critical for the redshift estimate.

Ideally, the softening parameter ($b$) used to derive asinh magnitudes from fluxes + uncertainties will be derived from the local noise on a per-object basis. However, when these are not available the use of a global softening parameter does not significantly impact photo-z results. 

For convenience, `gpz_pype` includes a function for estimating a suitable global softening parameter based on a set of input fluxes+errors (_that are assumed to be representative of the full field_):


```python
# Calculate a softening parameter for each filter in the list of filters derived above:
b_arr = [basic_lupt_soft(catalogue[f'FLUX_{filt}'], catalogue[f'FLUXERR_{filt}']) for filt in filters] 
```

With softening parameters calculated, we can then calculate the asinh magnitudes (also known as 'luptitudes') for each of our filters. These can be calculated using the `flux_to_lupt` function included in `gpz_pype`.


```python
# Make a new catalogue with the relevant key reference columns:
lupt_cols = catalogue[['ID', 'RA', 'DEC', 'CLASS_STAR', 'FLAGS', 'EBV', 'z_spec']]

check_nans = np.zeros(len(catalogue)) # Running counter of null values for each object

for filt, b in zip(filters, b_arr):
    lupt, lupterr = flux_to_lupt(catalogue[f'FLUX_{filt}'], # Input flux (uJy)
                                 catalogue[f'FLUXERR_{filt}'], # Input uncertainty (uJy)
                                 b, # Filter specific softening parameter
                                ) 
    
    lupt_cols[f'lupt_{filt}'] = lupt
    lupt_cols[f'lupterr_{filt}'] = lupterr
    
    check_nans += np.isnan(lupt) # Update nan counter for this filter
```

After filling all the columns, for the purpose of training GPz we want to cut our catalogue down to those sources with values in all the filters we want to use and for which there is a spectroscopic redshift:


```python
good = (check_nans == 0) # 'Good' sources for training are those with 0 NaNs

cat = lupt_cols[good * (np.isnan(lupt_cols['z_spec']) == False)] # Keep only good sources with z_spec
```

The following code assumes that `gpz++` is correctly installed on the system and that it can be run through the command line without issue. The `gpz_pype` classes simply wrap around `gpz++` and streamline the book-keeping required when using more complicated splitting+weighting of training sample.

The path to `gpz++` can be set as a system variable so this step might not be required in future. It can also be input directly into the `GPz` class below, but for safety we can use the convenience function `set_gpz_path` to set this for the current session.

## Running GPz


```python
set_gpz_path('gpzpp/bin/gpz++')
```

The main workhorse of `gpz_pype` is the `GPz` class, it needs to be instantiated with a set of parameters that we can then update manually (and which `GPz` will automatically set for us when necessary).
We can also define the number of CPUs we want `gpz++` to make use of during training:


```python
test = GPz(param_file='gpzpp/example/gpz.param', ncpu=4)
```
To perform GPz training on an input catalogue in gpz++ in our most straight-forward approach, we can use the `.run_training` function from the `GPz` class.

Even in its most straight-forward approach, there are lots of options that must be set. Many of which can be left as their degault value. I have tried to ensure that in-code documentation for `gpz_pype` is relatively complete, so the full set of options available can be seen using the standard python help functionality, e.g.:


```python
help(test.run_training)
```

    Help on method run_training in module gpz_pype.base:
    
    run_training(catalog, outdir, basename, gmm_output=None, bash_script=False, mag_prefix='mag_', error_prefix='magerr_', z_col='z_spec', id_col='id', weight_col=None, output_min=0, output_max=7, test_fraction=0.2, valid_fraction=0.2, do_iteration=False, iter_cov='gpvc', total_basis_functions=100, verbose=False, **kwargs) method of gpz_pype.base.GPz instance
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
        verbose : bool, optional
            If True, will print out GPz++ output.  Default is False.
        **kwargs : dict, optional
            Additional keyword arguments to pass to Table.read() if catalog is a
            filename.
        
        Returns
        -------
    



```python
simple_run, paths = test.run_training(
      cat, # Run training on this input catalogue - can be a path or a Table object
      outdir='test_dir', # Save training/output catalogues in this directory
      basename='candels_cosmos', # Start our output filenames with this
      bash_script=False, # Run GPz directly, don't just write the parameter files + commands to bash
      mag_prefix='lupt_', # Look for these asinh magnitude columns
      error_prefix='lupterr_', # Look for these error columns
      id_col='ID', # ID column to propagate into output files
      total_basis_functions=100, # NUMBF passed to gpz++
      do_iteration=False, # If True, run second iteration with more complex covariance
      verbose=False, # Don't print all of the gpz++ output to screen. Turn on for debugging
    )
```

    WARNING:root:Removing existing model file test_dir/candels_cosmos_model.dat


There are two outputs from `.run_training`:

1. Catalogue: The 'test' subset of the input catalogue, with `gpz++` prediction columns appended:

```python
simple_run.show_in_notebook()
```
2. Output paths: A dictionary containing the relative paths to the various files produced from the gpz++ training - including both input catalogues (in gpz format) and the trained model + associated parameter file:


```python
paths
```

    {'train': 'test_dir/candels_cosmos_train.txt',
     'test': 'test_dir/candels_cosmos_test.txt',
     'output_cat': 'test_dir/candels_cosmos_output.txt',
     'output_model': 'test_dir/candels_cosmos_model.dat',
     'param': 'test_dir/candels_cosmos.param'}



### Plotting outputs
Just for validation purposes, lets plot the predicted photo-$z$ versus spec-$z$ for the test catalogue. We can see that between $1 < z < 3$ the predictions do a decent job, even with just the 4-bands used in training. But at low and high redshifts the performance falls off


```python
# Function x**(1/2)
def forward(x):
    return np.log10(1+x)

def inverse(x):
    return (10**x) - 1

Fig, Ax = plt.subplots(1,1,figsize=(4.5,4.5))

Ax.errorbar(simple_run['z_spec'], simple_run['value'], 
            yerr=simple_run['uncertainty'], 
            fmt='o', ms=3, color='k', ecolor='k', alpha=0.2)
Ax.set_xlim([0, 7])
Ax.set_ylim([0, 7])

Ax.set_yscale('function', functions=(forward, inverse))
Ax.set_xscale('function', functions=(forward, inverse))


Ax.plot([0, 7], [0, 7], '--', color='0.5', lw=2)
Ax.set_xlabel(r'$z_{\rm{spec}}$')
Ax.set_ylabel(r'$z_{\rm{phot}}$')
```
    
![png](figures/gpz_pype%20-%20How-to%20guide_28_2.png)

## Modules
::: gpz_pype.base.GPz
