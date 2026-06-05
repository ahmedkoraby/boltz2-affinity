#!/usr/bin/env python3
"""CLI: predict binding affinity from a provided structure with Boltz-2.

Examples
--------
# Auto-extract protein chains/sequences from a multimer CIF, add a SMILES ligand,
# template the structure, and predict affinity:
python predict_affinity.py \
    --structure my_complex.cif \
    --ligand-smiles 'N[C@@H](Cc1ccc(O)cc1)C(=O)O' \
    --force --threshold 5.0 \
    --out results

# Use a CCD ligand and restrain only chain A as the template:
python predict_affinity.py --structure complex.cif --ligand-ccd SAH \
    --template-chains A --out results
"""
from __future__ import annotations

import argparse
import os
import sys

from boltz_affinity import (
    parse_structure, summary,
    ProteinChain, Ligand, TemplateSpec, PocketConstraint,
    build_yaml_dict, write_yaml,
    run_boltz, boltz_available,
    load_affinity, top_structure, report,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", required=True, help="provided CIF/PDB structure")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ligand-smiles")
    g.add_argument("--ligand-ccd")
    ap.add_argument("--ligand-id", default="LIG", help="chain id for the ligand")
    ap.add_argument("--binder-id", default=None,
                    help="ligand id to score (defaults to --ligand-id)")
    ap.add_argument("--template-chains", nargs="*", default=None,
                    help="which protein chains to template (default: all)")
    ap.add_argument("--force", action="store_true", help="restrain backbone to template")
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--no-template", action="store_true",
                    help="do NOT template; just rebuild from sequence (de novo cofold)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cpu", action="store_true", help="run on CPU (slow)")
    ap.add_argument("--no-msa-server", action="store_true")
    ap.add_argument("--cache", default=None)
    args = ap.parse_args(argv)

    parsed = parse_structure(args.structure)
    print(summary(parsed), "\n")
    if not parsed.protein_chains:
        sys.exit("No protein chains found in the structure.")

    proteins = [ProteinChain(c.chain_id, c.sequence) for c in parsed.protein_chains]
    used_ids = {p.id for p in proteins}
    lig_id = args.ligand_id
    if lig_id in used_ids:  # avoid clashing with a protein chain id
        lig_id = next(c for c in "LMNOPQRSTUVWXYZ" if c not in used_ids)
    ligand = Ligand(lig_id, smiles=args.ligand_smiles, ccd=args.ligand_ccd)
    binder_id = args.binder_id or lig_id

    templates = None
    if not args.no_template:
        chain_ids = args.template_chains or [p.id for p in proteins]
        templates = [TemplateSpec(
            cif=os.path.abspath(args.structure),
            chain_id=chain_ids,
            force=args.force,
            threshold=args.threshold if args.force else None,
        )]

    os.makedirs(args.out, exist_ok=True)
    yaml_path = os.path.join(args.out, "input.yaml")
    data = build_yaml_dict(proteins, [ligand], binder_id=binder_id, templates=templates)
    write_yaml(data, yaml_path)
    print(f"Wrote {yaml_path}\n")

    if not boltz_available():
        sys.exit("`boltz` not installed. Run: pip install boltz -U")

    run_boltz(
        yaml_path, args.out,
        use_msa_server=not args.no_msa_server,
        accelerator="cpu" if args.cpu else "gpu",
        cache=args.cache,
    )

    res = load_affinity(args.out, binder_chain=binder_id)
    print("\n" + report(res))
    s = top_structure(args.out)
    if s:
        print(f"\nTop structure: {s}")


if __name__ == "__main__":
    main()
