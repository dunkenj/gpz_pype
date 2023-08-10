# gpz_pype

[![Build Status](https://travis-ci.org/COINtoolbox/gpz_pype.svg?branch=master)](https://travis-ci.org/COINtoolbox/gpz_pype)
[![Coverage Status](https://coveralls.io/repos/github/COINtoolbox/gpz_pype/badge.svg?branch=master)](https://coveralls.io/github/COINtoolbox/gpz_pype?branch=master)
[![Documentation Status](https://readthedocs.org/projects/gpz-pype/badge/?version=latest)](https://gpz-pype.readthedocs.io/en/latest/?badge=latest)

This is a python package for running Gaussian Mixture Model augmented photometric redshift estimation. It is designed to be used with the [gpz++](https://github.com/cschreib/gpzpp) implementation of [GPZ](https://github.com/OxfordML/GPz) (Almosallam et al. 2016a,2016b,2017).

## Installation

This code assumes a working installation of [gpz++](https://github.com/cschreib/gpzpp). To install the package, clone the repository and run

```bash
python setup.py install
```

Or install using pip with

```bash
pip install gpy_pype
```

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

# Acknowledgements

The [gpz++](https://github.com/cschreib/gpzpp) C++ version of GPz was developed by [Corentin Schreiber](https://github.com/cschreib/) for the Euclid space mission, with funding from the UK Space Agency, under the supervision of Matt Jarvis.

If you use this code for your own work, please cite this repository and [gpz++](https://github.com/cschreib/gpzpp).
The underlying GPz algorithm is presented in Almosallam et al. (2016a,2016b) where GPz was first introduced, with [Almosallam (2017)](http://www.robots.ox.ac.uk/~parg/pubs/theses/ibrahim_almosallam_thesis.pdf) outlining additional features incorporated into GPz v2.0 that are incorporated into [gpz++](https://github.com/cschreib/gpzpp).