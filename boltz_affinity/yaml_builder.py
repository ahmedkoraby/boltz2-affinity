"""Build a Boltz-2 input YAML that takes a *provided* structure (CIF) as a
template and requests binding-affinity prediction for a single ligand.

The central idea
----------------
Boltz-2 always runs its diffusion structure module; it does not "score a
pre-built complex" directly. To make affinity prediction respect a structure
you already have (e.g. a multimer CIF from MD, docking, or the PDB), we feed
that structure through the ``templates:`` block. The protein chains in
``sequences:`` are templated against your CIF (optionally *forced* so the
backbone is restrained to it), while the ligand is supplied as SMILES/CCD and
flagged as the affinity ``binder``.

Key constraints of the Boltz-2 affinity head (as of 2026):
  * Affinity is for ONE small-molecule ligand only (no protein-protein,
    no multiple ligands per affinity request).
  * Templates apply to protein chains; ligands are not templated.
  * The affinity module does not explicitly model cofactors, ions, water, or
    multimeric binding partners, so for multimers treat the affinity number as
    "ligand vs. the templated pocket" and read the caveats in the README.
  * CIF templates are recommended; PDB template support has been buggy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union
import yaml


@dataclass
class ProteinChain:
    id: str
    sequence: str
    msa: Optional[str] = None  # path to a precomputed .a3m, or None to use the MSA server


@dataclass
class Ligand:
    id: str
    smiles: Optional[str] = None
    ccd: Optional[str] = None

    def __post_init__(self):
        if bool(self.smiles) == bool(self.ccd):
            raise ValueError(
                f"Ligand {self.id!r} must have exactly one of 'smiles' or 'ccd'."
            )


@dataclass
class TemplateSpec:
    cif: str                                   # path to the provided structure (CIF)
    chain_id: Optional[Union[str, list]] = None        # which YAML chain(s) to template
    template_id: Optional[Union[str, list]] = None     # which chain(s) inside the CIF to use
    force: bool = False                        # restrain backbone toward the template
    threshold: Optional[float] = None          # required if force=True (Angstrom)


@dataclass
class PocketConstraint:
    """Optional but strongly recommended for affinity: pins the binder to the
    real pocket so the affinity head sees the correct interface."""
    binder: str
    contacts: list                  # e.g. [["A", 123], ["A", 145]]  (chain, residue idx, 1-based)
    max_distance: float = 6.0


def build_yaml_dict(
    proteins: list[ProteinChain],
    ligands: list[Ligand],
    binder_id: Optional[str] = None,
    templates: Optional[list[TemplateSpec]] = None,
    pocket: Optional[PocketConstraint] = None,
    version: int = 1,
) -> dict:
    """Return a dict matching the Boltz-2 YAML schema."""
    if not proteins:
        raise ValueError("At least one protein chain is required.")

    sequences: list[dict] = []
    for p in proteins:
        entry: dict = {"id": p.id, "sequence": p.sequence}
        if p.msa:
            entry["msa"] = p.msa
        sequences.append({"protein": entry})

    for lig in ligands:
        entry = {"id": lig.id}
        if lig.smiles:
            entry["smiles"] = lig.smiles
        else:
            entry["ccd"] = lig.ccd
        sequences.append({"ligand": entry})

    out: dict = {"version": version, "sequences": sequences}

    # ---- templates: feed the provided structure ----
    if templates:
        tlist = []
        for t in templates:
            td: dict = {"cif": t.cif}
            if t.chain_id is not None:
                td["chain_id"] = t.chain_id
            if t.template_id is not None:
                td["template_id"] = t.template_id
            if t.force:
                td["force"] = True
                if t.threshold is None:
                    raise ValueError("force=True requires a 'threshold' (Angstrom).")
                td["threshold"] = t.threshold
            tlist.append(td)
        out["templates"] = tlist

    # ---- pocket constraint (optional) ----
    if pocket is not None:
        out["constraints"] = [{
            "pocket": {
                "binder": pocket.binder,
                "contacts": [list(c) for c in pocket.contacts],
                "max_distance": pocket.max_distance,
            }
        }]

    # ---- affinity request ----
    if binder_id is not None:
        lig_ids = {l.id for l in ligands}
        if binder_id not in lig_ids:
            raise ValueError(
                f"binder_id={binder_id!r} is not one of the ligand ids {sorted(lig_ids)}."
            )
        out["properties"] = [{"affinity": {"binder": binder_id}}]

    return out


def write_yaml(data: dict, path: str) -> str:
    """Serialize a YAML dict to disk and return the path."""
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    return path
