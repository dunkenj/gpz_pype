# gpz_pype
This is a python package for running Gaussian Mixture Model augmented photometric redshift estimation. It is designed to be used with the [gpz++](https://github.com/cschreib/gpzpp) implementation of [GPZ](https://github.com/OxfordML/GPz) (Almosallam et al. 2016a,2016b,2017).

## Features

* A simple python interface for running gpz++ on a dataset.
* Simplified data augmentation through Gaussian Mixture models (GMMs) following the approach presented in [Hatfield et al. (2020)](https://arxiv.org/abs/2009.01952) and [Duncan (2023)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.3662D/abstract).
* Implementation of cost-sensitive learning weights and sample division from GMM inputs.

## Installation

This code assumes a working installation of [gpz++](https://github.com/cschreib/gpzpp). To install the package, clone the repository and run

```bash
make install
```

Or install using pip with

```bash
pip install gpy_pype
```
_(Not yet available.)_

To use the core features of the package, you will need to set the environment variable `GPZPATH` to point to the location of the gpz++ executable. For example, in bash you can do this with
```bash
export GPZPATH=/path/to/gpzpp
```
Alternatively, you can set it in your python script with
```python
import os
os.environ['GPZPATH'] = '/path/to/gpzpp'
```
or using the convenience function
```python
from gpz_pype.utilities import set_gpz_path
set_gpz_path('/path/to/gpzpp')
```

## Usage

Basic usage for running GPz is demonstrated in [gpz](gpz), while the data augmentation features are demonstrated in [gmm](gmm).

# Acknowledgements

The [gpz++](https://github.com/cschreib/gpzpp) C++ version of GPz was developed by [Corentin Schreiber](https://github.com/cschreib/) for the Euclid space mission, with funding from the UK Space Agency, under the supervision of Matt Jarvis.

If you use this code for your own work, please cite this repository and [gpz++](https://github.com/cschreib/gpzpp).
The underlying GPz algorithm is presented in Almosallam et al. (2016a,2016b) where GPz was first introduced, with [Almosallam (2017)](http://www.robots.ox.ac.uk/~parg/pubs/theses/ibrahim_almosallam_thesis.pdf) outlining additional features incorporated into GPz v2.0 that are incorporated into [gpz++](https://github.com/cschreib/gpzpp).