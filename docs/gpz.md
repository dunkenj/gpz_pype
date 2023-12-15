
# Basic GPz++ usage

The following code assumes that `gpz++` is correctly installed on the system and that it can be run through the command line without issue. The `gpz_pype` classes simply wrap around `gpz++` and streamline the book-keeping required when using more complicated splitting+weighting of training sample.

The path to `gpz++` can be set as a system variable so this step might not be required in future. It can also be input directly into the `GPz` class below, but for safety we can use the convenience function `set_gpz_path` to set this for the current session.


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
