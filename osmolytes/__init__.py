"""Osmolytes and protein stability

This code attempts to predict the influence of osmolytes on protein stability,
using the methods in the following paper:

* Auton M, Bolen DW. Predicting the energetics of osmolyte-induced protein
  folding/unfolding. *Proc Natl Acad Sci* 102:15065 (2005)
  https://doi.org/10.1073/pnas.0507053102
"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
