# Gaussian Mixture Model Augmentation

## Introduction

## Example GMM Divide + Cost-sensitive Learning Calculations

The basic functionality for the Gaussian Mixture Model (GMM) sample division and cost-sensitive learning is contained in the `GMMbasic` class. The training of the respective GMMs is done internally within the class, and is effectively a wrapper around the [sci-kit learn/GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture) classes with all its functionality.

The key inputs that we need to define for `GMMbasic` are as follows:

- `X_pop` : The feature array for the reference population.
- `X_train` : The feature array for the training sample.
- `Y_train` : The labels for the training sample (i.e. the spectroscopic redshifts).
- `ncomp` : Number of mixtures used for the GMM model.

For building the mixture models, we need to manually decide on and create a set of features (`X_pop`/`X_train`) with which we want to represent our sample. The 'best' features will be dependent on the input data, the scientific goals etc. These also do not necessarily need to be features used in the GPz training (e.g. luptitudes), or even things derived from luptitudes. 

In the following example, given the limited number of features we will just use a combination of colours and magnitudes. But sizes or morphological information could also be sensible choices.

Since for our example the training set is a subset of the reference population, we can define one set of features and then split:


```python
gmm_features = np.array([lupt_cols['lupt_F606W']-lupt_cols['lupt_F814W'],
                         lupt_cols['lupt_F814W']-lupt_cols['lupt_F125W'],
                         lupt_cols['lupt_F160W']]).T

X_pop = gmm_features[good] # The full reference population 
                           # (i.e. representative of the full sample we would like photo-z predictions for)
X_train = gmm_features[good * (np.isnan(lupt_cols['z_spec']) == False)] # The training subset
```

Instantiating the class with the inputs defined above will build the GMMs:


```python
gmm = GMMbasic(X_pop=X_pop, 
               X_train=X_train, 
               Y_train=cat['z_spec'], 
               ncomp=4) # For larger samples and more features, more mixtures could be appropriate
```

## GMM Weight

Since the cost-sensitive learning (CSL) is the most straight-forward to include in `gpz++`, we will first generate some CSL weights based on the GMMs we have just produced. To do so is as straight-forward as:


```python
weights = gmm.calc_weights(X_train, X_pop, max_weight=100)
```

In principle, we could go straight to running GPz with these inputs. But its first useful to verify that the weights are producing sensible results. And if necessary, to quantitatively compare how well different features perform when trying to match the target distribution.

For the feature set we defined earlier, we can simply plot the distributions of the training sample features, $x_{i}$, before and after weighting is applied:


```python
Fig, Ax = plt.subplots(1,3, figsize=(9,3))

Ax = Ax.flatten()

for i, ax in enumerate(Ax):
    c, bins, _ = ax.hist(X_pop[:,i], density=True, bins=25, 
                      range=np.percentile(X_pop[:,i], [0.1, 99.9]), histtype='step', lw=2,
                        label='All')
    ax.hist(X_train[:,i], density=True, bins=bins, histtype='step', lw=2, color='firebrick',
            label='Training')
    ax.hist(X_train[:,i], density=True, bins=bins, histtype='step', lw=2, color='firebrick', 
            ls='--', weights=weights,
            label='Weighted Training')

    ax.set_ylabel('PDF(x)')
    ax.set_xlabel(f'$x_{i+1}$')

Ax[0].legend(loc='upper right', prop={'size':7})
Fig.tight_layout()
```


    
![png](figures/gpz_pype%20-%20How-to%20guide_36_0.png)
    


Although not perfect, we can see that the fainter training sources have been significantly up-weighted when compared to the original distribution. The colours are also distributed more like those of the full sample.

To include the weights in GPz we can simply add them to our input catalogue and tell GPz to include them in the training:


```python
cat['weights'] = weights # Add the weights to our previous catalogue

weights_run, paths = test.run_training( # All of these as above but with minor changes
      cat, 
      outdir='test_dir', 
      basename='candels_cosmos_weighted', # Change the prefix to keep separate
      bash_script=False,
      mag_prefix='lupt_',
      error_prefix='lupterr_', 
      id_col='ID',
      total_basis_functions=100,
      do_iteration=False, 
      verbose=False,
      weight_col='weights', # Now weight training by this column
    )
```

    WARNING:root:Removing existing model file test_dir/candels_cosmos_weighted_model.dat



```python
Fig, Ax = plt.subplots(1,2,figsize=(9,4))

Ax[0].errorbar(simple_run['z_spec'], simple_run['value'], 
            yerr=simple_run['uncertainty'], 
            fmt='o', ms=3, color='k', ecolor='k', alpha=0.2)

Ax[1].errorbar(weights_run['z_spec'], weights_run['value'], 
            yerr=weights_run['uncertainty'], 
            fmt='o', ms=3, color='steelblue', ecolor='steelblue', alpha=0.4)

labels = ['Simple', 'Weighted']

stats_simple = calcStats(simple_run['z_spec'], simple_run['value'])
stats_weight = calcStats(weights_run['z_spec'], weights_run['value'])

stats = [stats_simple, stats_weight]

for i, ax in enumerate(Ax):
    ax.set_xlim([0, 7])
    ax.set_ylim([0, 7])
    
    ax.set_yscale('function', functions=(forward, inverse))
    ax.set_xscale('function', functions=(forward, inverse))
    ax.plot([0, 7], [0, 7], '--', color='0.5', lw=2)
    ax.set_xlabel(r'$z_{\rm{spec}}$')
    ax.set_ylabel(r'$z_{\rm{phot}}$')
    ax.set_title(labels[i], size=11)
    
    ax.text(0.1, 6.0, f'$\sigma = $ {stats[i][0]:.3f}')
    ax.text(0.1, 5.0, f'OLF = {stats[i][2]:.3f}')


```

    /var/folders/4h/l82f8_jx1476t0k78sdtfq5w0000gn/T/ipykernel_70492/3209683446.py:3: RuntimeWarning: invalid value encountered in log10
      return np.log10(1+x)



    
![png](figures/gpz_pype%20-%20How-to%20guide_39_1.png)
    


At face value, the weights have not resulted in an overall statistical improvement to the photo-$z$ estimates. But by eye, there is definitely some improvement at the higher redshift end. For now, we move onto the GMM-Divide option.

## GMM Divide

Dividing the training sample into different mixtures is as equally simple. For this example with its relatively small sample sizes, the number of training sources that could end up assigned to a mixture could be very small. We therefore will lower the threshold above which a source will be assigned to a mixture to 0.2, approximately saying that the source has at least a 20% chance of belonging to that mixture. (Note the default 0.5 effectively assigns each source to its best match)


```python
train_mixtures = gmm.divide(X_train[:,:], 
                            weight=False, # Do not include CSL weights yet
                            threshold=0.2) # Change the divide threshold

cat['weights'] = 1. # Re-set the weights to be equal

# Save for later
train_mixtures.write('cosmos_mixtures_ex.csv', format='ascii.csv', overwrite=True)
```


```python
gmm_output = Table.read('cosmos_mixtures_ex.csv', format='ascii.csv')

divide_run, paths = test.run_training(cat, outdir='test_dir', 
                         basename='candels_cosmos_divide',
                         gmm_output=gmm_output,
                         bash_script=False,
                         weight_col='weights',
                         mag_prefix='lupt_', 
                         error_prefix='lupterr_', id_col='ID',
                         total_basis_functions=100, do_iteration=False, verbose=False)
```

    GPz++ Run (mixture 1/4):   0%|                                                                                                 | 0/4 [00:00<?, ?it/s]

    WARNING:root:Removing existing model file test_dir/candels_cosmos_divide_m0_model.dat


    GPz++ Run (mixture 2/4):  25%|██████████████████████▎                                                                  | 1/4 [00:01<00:03,  1.08s/it]

    WARNING:root:Removing existing model file test_dir/candels_cosmos_divide_m1_model.dat


    GPz++ Run (mixture 3/4):  50%|████████████████████████████████████████████▌                                            | 2/4 [00:02<00:02,  1.20s/it]

    WARNING:root:Removing existing model file test_dir/candels_cosmos_divide_m2_model.dat


    GPz++ Run (mixture 4/4):  75%|██████████████████████████████████████████████████████████████████▊                      | 3/4 [00:03<00:01,  1.16s/it]

    WARNING:root:Removing existing model file test_dir/candels_cosmos_divide_m3_model.dat


    GPz++ Run (mixture 4/4): 100%|█████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.17s/it]


As above, lets compare the results between the simple and GMM divided runs:


```python
Fig, Ax = plt.subplots(1,2,figsize=(9,4))

Ax[0].errorbar(simple_run['z_spec'], simple_run['value'], 
            yerr=simple_run['uncertainty'], 
            fmt='o', ms=3, color='k', ecolor='k', alpha=0.2)

Ax[1].errorbar(divide_run['z_spec'], divide_run['value'], 
            yerr=divide_run['uncertainty'], 
            fmt='o', ms=3, color='steelblue', ecolor='steelblue', alpha=0.4)

labels = ['No divide', 'Divide']

stats_simple = calcStats(simple_run['z_spec'], simple_run['value'])
stats_divide = calcStats(divide_run['z_spec'], divide_run['value'])

stats = [stats_simple, stats_divide]

for i, ax in enumerate(Ax):
    ax.set_xlim([0, 7])
    ax.set_ylim([0, 7])

    ax.set_yscale('function', functions=(forward, inverse))
    ax.set_xscale('function', functions=(forward, inverse))
    ax.plot([0, 7], [0, 7], '--', color='0.5', lw=2)
    ax.set_xlabel(r'$z_{\rm{spec}}$')
    ax.set_ylabel(r'$z_{\rm{phot}}$')
    ax.set_title(labels[i], size=11)
    
    ax.text(0.1, 6.0, f'$\sigma = $ {stats[i][0]:.3f}')
    ax.text(0.1, 5.0, f'OLF = {stats[i][2]:.3f}')   
```
    
![png](figures/gpz_pype%20-%20How-to%20guide_44_1.png)
    


This time we can see that overall outlier fraction and scatter are now improved. Visually, we can also see that the high-redshift end is also significantly improved. 

Lets now do a run using both the GMM Divide and Weights. This can be done using the divide option, but setting `weight = True` in the options. To prevent confusion, we will also now remove the weight column we added above, since the weights will now be propagated through the catalogue included as `gmm_output`:


```python
train_mixtures2 = gmm.divide(X_train[:,:], 
                            weight=True,
                            threshold=0.2) # Do not include CSL weights yet

if 'weights' in cat.colnames:
    cat.remove_columns(['weights'])  # Weights will be taken from the GMM outputs file now instead
```

Run the gpz training as normal:


```python
combined_run, paths = test.run_training(cat, outdir='test_dir', 
                         basename='candels_cosmos_both',
                         gmm_output=train_mixtures2,
                         bash_script=False,
                         weight_col='weights',
                         mag_prefix='lupt_', 
                         error_prefix='lupterr_', id_col='ID',
                         total_basis_functions=100, do_iteration=False, verbose=False)
```
    GPz++ Run (mixture 4/4): 100%|█████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.15s/it]

```python
Fig, Ax = plt.subplots(1,2,figsize=(9,4))

Ax[0].errorbar(simple_run['z_spec'], simple_run['value'], 
            yerr=simple_run['uncertainty'], 
            fmt='o', ms=3, color='k', ecolor='k', alpha=0.2)

Ax[1].errorbar(combined_run['z_spec'], combined_run['value'], 
            yerr=combined_run['uncertainty'], 
            fmt='o', ms=3, color='steelblue', ecolor='steelblue', alpha=0.4)

labels = ['No divide', 'Divide + Weight']

stats_simple = calcStats(simple_run['z_spec'], simple_run['value'])
stats_combined = calcStats(combined_run['z_spec'], combined_run['value'])

stats = [stats_simple, stats_combined]

for i, ax in enumerate(Ax):
    ax.set_xlim([0, 7])
    ax.set_ylim([0, 7])

    ax.set_yscale('function', functions=(forward, inverse))
    ax.set_xscale('function', functions=(forward, inverse))
    ax.plot([0, 7], [0, 7], '--', color='0.5', lw=2)
    ax.set_xlabel(r'$z_{\rm{spec}}$')
    ax.set_ylabel(r'$z_{\rm{phot}}$')
    ax.set_title(labels[i], size=11)
    
    ax.text(0.1, 6.0, f'$\sigma = $ {stats[i][0]:.3f}')
    ax.text(0.1, 5.0, f'OLF = {stats[i][2]:.3f}')   
```
    
![png](figures/gpz_pype%20-%20How-to%20guide_49_1.png)
    


Compared to the Divide-only run, our combined Divide+Weights is performing slightly worse for the two metrics we have chosen, but is still an improvement over the Weight-only option. So in this case, using only the GMM-Divide would be the optimal approach. But as training sample sizes increase and additional features are added to the GPz training, this might not remain the case.

Additionally, since the spectroscopic sample is obviously biased, the Weight/Divide+Weight runs might actually provide better statistics for the overall galaxy population (at the expense of the brigher sources). Assessing the statistics as a function of e.g. spectroscopic redshift or magnitude can start to make these assessments clearer.


## Code reference
::: gpz_pype.gmm.GMMbasic
