# Osmolytes and protein stability

![pytest](https://github.com/Electrostatics/osmolytes/workflows/pytest/badge.svg)

This code attempts to predict the influence of osmolytes on protein stability, using the methods in:  Auton M, Bolen DW. Predicting the energetics of osmolyte-induced protein folding/unfolding. _Proc Natl Acad Sci_ 102:15065 (2005) https://doi.org/10.1073/pnas.0507053102

Other models may be added in the future.

## Installation

Regardless of how you install the software, we recommend doing so in a virtual environment. 
You can create a virtual environment using, e.g., one of the following:

```
conda create -n osmolytes pip
```
(note that you will need the `pip` package for later steps) or
```
virtualenv -p python3 ~/venvs/osmolytes
```

You will need to activate the virtual environment prior to installing or using software, e.g., with one of the following:

```
conda activate osmolytes
```
or
```
. ~/venvs/osmolytes/bin/activate
```

### Install from PyPI repository

Most users will want to install the software from the [PyPI repository](https://pypi.org/project/osmolytes/0.0.1/) via the following steps:

1. Activate your virtual environment as described above
2. Install the package using
```
pip install osmolytes
```

### Install from source code

Adventurous users and developers may want to install the software from source code via the following steps:

1. Activate your virtual environment as described above
2. Download the source code
```
git clone https://github.com/Electrostatics/osmolytes.git
```
3. Change to the source code directory (e.g., `cd osmolytes`)
4. Install the package from source code using
```
pip install -e .
```
If you are using conda, you may first need to install pip with
```
conda install pip
```

Installing from source also allows testing of the software with 
```
python -m pytest .
```
These tests check the integrity of the software and provide information about the accuracy of the various calculations (particularly, surface area).

## Usage

The main command-line tool installed by this package is the program `mvalue`.

The input to the `mvalue` program is a [PQR file](https://pdb2pqr.readthedocs.io/en/latest/formats/pqr.html) which can be generated by the [PDB2PQR software](https://pdb2pqr.readthedocs.io/en/latest/index.html). 
Most users will want to use the [PDB2PQR web server](http://server.poissonboltzmann.org/).
The `mvalue` program has been tested with PQR files generated by PDB2PQR using the following settings:

* Protonation states assigned by PROPKA
* PARSE force field
* Internal naming scheme

In addition to generating a PQR, the [PROPKA](https://github.com/jensengroup/propka) software generates additional output in the form of `*.propka` files that contain useful predictions about the pH stability of the protein.

The `mvalue` program has several options which can be listed using 
```
mvalue --help
```
The basic usage of `mvalue` is 
```
mvalue 2BU4.pqr
```
where `2BU4.pqr` is the name of an example PQR file.

The `mvalue` program produces output that describes 

* The residue composition of the protein
* The contribution of each residue type (and its surface area) to the *m*-value of the protein for each osmolyte species (at 1M concentration).
* The *m*-value associated with each osmolyte at 1M concentration.

The concept of a *m*-value is described in [Wikipedia](https://en.wikipedia.org/wiki/Equilibrium_unfolding) and many review articles on protein folding and stability.

# Support

Re-implementation of these algorithms was supported by National Institutes of Health grant GM069702.

