#!/usr/bin/env python3
"""CLI: predict binding affinity from a provided structure with Boltz-2.

The robust, recommended workflow is to anchor the ligand with a POCKET
constraint (your known binding-site residues) rather than a template. Boltz-2
always re-folds via diffusion, so a template only biases the backbone -- and
its template mmCIF parser is fragile with non-RCSB CIFs (see boltz issues
#300/#451). The pocket constraint is parsed directly from the YAML, never
touches that code path, and guarantees the affinity module sees a real
protein-ligand interface.

Examples
--------
# Recommended: sequence + pocket constraint + affinity
python predict_affinity.py \\
    --structure my_complex.cif \\
    --ligand-smiles 'N[C@@H](Cc1ccc(O)cc1)C(=O)O' \\
    --pocket "A:45,A:48,A:52,A:96" \\
    --out results

# Opt-in template (best effort; may fail to parse on some CIFs):
python predict_affinity.py --structure complex.cif --ligand-ccd SAH \\
    --template --force --threshold 5.0 --pocket "A:45,A:48" --out results
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
    ensure_template_cif,
)


def parse_pocket_contacts(text):
    """Parse 'A:45,A:48,B:12' -> [['A',45],['A',48],['B',12]]."""
    contacts = []
    if not text:
        return contacts
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid pocket residue '{item}'. Expected e.g. A:123.")
        chain, resi = item.split(":", 1)
        chain, resi = chain.strip(), resi.strip()
        if not chain:
            raise ValueError(f"Missing chain ID in pocket residue '{item}'.")
        try:
            resi = int(resi)
        except ValueError:
            raise ValueError(f"Invalid residue number in pocket residue '{item}'.")
        contacts.append([chain, resi])
    return contacts


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

    # Pocket constraint (recommended)
    ap.add_argument("--pocket", default=None,
                    help='comma-separated pocket residues, e.g. "A:45,A:48,A:52"')
    ap.add_argument("--pocket-distance", type=float, default=6.0,
                    help="max ligand-pocket contact distance (A) for the pocket constraint")

    # Template (opt-in, best effort)
    ap.add_argument("--template", action="store_true",
                    help="use the structure as a template (best effort; may fail on some CIFs)")
    ap.add_argument("--template-chains", nargs="*", default=None,
                    help="which protein chains to template (default: all)")
    ap.add_argument("--force", action="store_true", help="restrain backbone to template")
    ap.add_argument("--threshold", type=float, default=5.0)

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
    if lig_id in used_ids:
        lig_id = next(c for c in "LMNOPQRSTUVWXYZ" if c not in used_ids)
    ligand = Ligand(lig_id, smiles=args.ligand_smiles, ccd=args.ligand_ccd)
    binder_id = args.binder_id or lig_id

    # ---- template (opt-in) ----
    templates = None
    if args.template:
        chain_ids = args.template_chains or [p.id for p in proteins]
        template_cif = ensure_template_cif(args.structure)  # sanitized, protein-only
        print(f"[template] using {template_cif}")
        print("[template] NOTE: Boltz-2's template mmCIF parser is fragile; if you "
              "hit a parse error, drop --template and rely on --pocket.")
        templates = [TemplateSpec(
            cif=os.path.abspath(template_cif),
            chain_id=chain_ids,
            force=args.force,
            threshold=args.threshold if args.force else None,
        )]

    # ---- pocket constraint (recommended) ----
    pocket = None
    if args.pocket:
        pocket = PocketConstraint(
            binder=binder_id,
            contacts=parse_pocket_contacts(args.pocket),
            max_distance=args.pocket_distance,
        )
    elif not args.template:
        print("[!] No --pocket and no --template: Boltz will fold de novo and may "
              "place the ligand with no protein contacts, which makes the affinity "
              "module fail. Supplying --pocket is strongly recommended.")

    os.makedirs(args.out, exist_ok=True)
    yaml_path = os.path.join(args.out, "input.yaml")
    data = build_yaml_dict(proteins, [ligand], binder_id=binder_id,
                           templates=templates, pocket=pocket)
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

    try:
        res = load_affinity(args.out, binder_chain=binder_id)
        print("\n" + report(res))
    except FileNotFoundError:
        print("\n[!] No affinity output. Boltz likely skipped the affinity cropper "
              "(no protein-ligand contacts). Add/refine --pocket residues and retry.")
    s = top_structure(args.out)
    if s:
        print(f"\nTop structure: {s}")


if __name__ == "__main__":
    main()
