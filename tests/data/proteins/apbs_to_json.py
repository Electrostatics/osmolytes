"""Convert APBS SASA output to JSON format."""
import argparse
import json
from osmolytes import pqr


def build_parser():
    """Build argument parser.

    :returns:  argument parser
    :rtype:  argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Convert APBS SASA output to JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pqr", help="path to input PQR file", default="NikR_chains.pqr"
    )
    parser.add_argument(
        "--apbs-output",
        help="path to input APBS output file",
        default="NikR-apbs.out",
    )
    parser.add_argument("json_output", help="path to output JSON file")
    return parser


def parse_apbs_output(apbs_file):
    """Parse APBS output files for SASA information.

    :param file apbs_file:  file-like object with APBS output data
    :returns:  list of per-atom SASAs
    :rtype:  list(float)
    """
    sasa_values = []
    for line in apbs_file:
        line = line.strip()
        if line.startswith("SASA for atom "):
            words = line.split()
            sasa = float(words[4])
            sasa_values.append(sasa)
    return sasa_values


def main():
    """Main driver."""
    parser = build_parser()
    args = parser.parse_args()
    with open(args.pqr, "rt") as pqr_file:
        atoms = pqr.parse_pqr_file(pqr_file)
    with open(args.apbs_output, "rt") as apbs_file:
        sasa_values = parse_apbs_output(apbs_file)
    sasa_dict = {}
    for iatom, atom in enumerate(atoms):
        if atom.chain_id not in sasa_dict:
            sasa_dict[atom.chain_id] = {}
        chain_dict = sasa_dict[atom.chain_id]
        residue_key = f"{atom.res_num} {atom.res_name}"
        if residue_key not in chain_dict:
            chain_dict[residue_key] = 0.0
        chain_dict[residue_key] = chain_dict[residue_key] + sasa_values[iatom]
        sasa_dict[atom.chain_id] = chain_dict
    with open(args.json_output, "wt") as json_file:
        json.dump(sasa_dict, json_file, indent=2)


if __name__ == "__main__":
    main()
