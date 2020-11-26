"""Transfer energy calculation from Auton & Bolen.

From Eqs. 1 and 2 in Supporting Text from:

  Auton M, Bolen DW. Predicting the energetics of osmolyte-induced
  protein folding/unfolding. Proc Natl Acad Sci U S A. 2005 Oct 18;102(42):
  15065-8. doi: 10.1073/pnas.0507053102. Epub 2005 Oct 7. PMID: 16214887;
  PMCID: PMC1257718.
"""
import logging
import pkg_resources
import yaml
import pandas as pd
from osmolytes.pqr import aggregate, count_residues
from osmolytes.sasa import ReferenceModels


_LOGGER = logging.getLogger(__name__)
ENERGY_DICT = yaml.load(
    pkg_resources.resource_stream(__name__, "data/transfer-energies.yaml"),
    Loader=yaml.FullLoader
)


def get_folded_areas(atoms, sas):
    """Get a summary by residue type of folded areas.

    .. todo:: Move to the :mod:`sasa` module.

    :param list(pqr.Atom) atoms:  list of atoms
    :param SolventAccessibleSurface sas: solvent-accessible surface area object
    :returns:  DataFrame with folded sidechain/backbone area by residue type
    :rtype:  pd.DataFrame
    """
    folded_areas = [
        sas.atom_surface_area(iatom) for iatom in range(len(atoms))
    ]
    folded_areas = aggregate(
        atoms,
        folded_areas,
        chain_id=False,
        res_name=True,
        res_num=False,
        sidechain=True,
    ).set_index("res_name")
    folded_areas = folded_areas.rename({"sidechain": "what"}, axis="columns")
    grouper = folded_areas.groupby("what")
    folded_areas = pd.concat(
        [pd.Series(v["value"], name=k) for k, v in grouper], axis="columns"
    ).sort_index()
    return folded_areas


def transfer_energy(atoms, sas):
    """Calculate transfer energy for each osmolyte and amino acid type.

    .. todo:: implement max/min for Creamer model

    :param list(pqr.Atom) atoms:  list of atoms
    :param SolventAccessibleSurface sas: solvent-accessible surface area object
    :returns:  DataFrame with energy contribution by residue type
    :rtype:  pd.DataFrame
    """
    ref_model = ReferenceModels()
    counts = count_residues(atoms)
    folded_areas = get_folded_areas(atoms, sas)
    unfolded_areas = ref_model.molecule_areas(
        atoms, model="auton", how="mean"
    )
    tripeptide_areas = ref_model.molecule_areas(
        atoms, model="tripeptide", how="mean"
    )
    scaled_areas = (unfolded_areas - folded_areas) / tripeptide_areas
    osmolytes = list(ENERGY_DICT["backbone"].keys())
    residue_energies = {}
    bb_energy = {osmolyte: 0.0 for osmolyte in osmolytes}
    for res, area in scaled_areas.T.iteritems():
        sc_energy = {}
        for osmolyte in osmolytes:
            sc_energy[osmolyte] = counts[res] * ENERGY_DICT[res][osmolyte] * area["sidechain"]
            bb_energy[osmolyte] += counts[res] * ENERGY_DICT["backbone"][osmolyte] * area["backbone"]
        residue_energies[res] = sc_energy
    residue_energies["backbone"] = bb_energy
    return pd.DataFrame(residue_energies).T
