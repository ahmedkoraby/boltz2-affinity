"""Read a provided structure (CIF or PDB) and extract the information needed to
build a Boltz-2 input: per-chain one-letter protein sequences and any ligand
(HETATM / non-polymer) residues that could serve as the affinity binder.

Uses gemmi (a hard dependency of boltz, so it is always available alongside it).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import gemmi
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gemmi is required. It ships with boltz; install via `pip install gemmi`."
    ) from exc


@dataclass
class ParsedChain:
    chain_id: str
    kind: str                 # 'protein', 'nucleic', or 'other'
    sequence: str             # one-letter (empty if not a polymer)
    n_residues: int


@dataclass
class ParsedLigand:
    chain_id: str
    resname: str
    n_atoms: int


@dataclass
class ParsedStructure:
    path: str
    chains: list[ParsedChain] = field(default_factory=list)
    ligands: list[ParsedLigand] = field(default_factory=list)

    @property
    def protein_chains(self) -> list[ParsedChain]:
        return [c for c in self.chains if c.kind == "protein"]


def _read_any(path: str) -> "gemmi.Structure":
    """Robustly read a coordinate file. Some CIF/PDB exports (PyMOL, Chimera,
    docking tools) trip gemmi's auto-detection and return 0 models; we retry by
    forcing the format based on the extension and on content."""
    st = gemmi.read_structure(path)
    if len(st) > 0:
        return st

    # Retry by forcing formats.
    fmts = []
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        fmts = [gemmi.CoorFormat.Mmcif, gemmi.CoorFormat.Pdb]
    elif ext in (".pdb", ".ent"):
        fmts = [gemmi.CoorFormat.Pdb, gemmi.CoorFormat.Mmcif]
    else:
        fmts = [gemmi.CoorFormat.Mmcif, gemmi.CoorFormat.Pdb]

    for fmt in fmts:
        try:
            st2 = gemmi.read_structure(path, format=fmt)
            if len(st2) > 0:
                return st2
        except Exception:
            pass

    raise ValueError(
        f"gemmi parsed 0 models from '{path}'. The file may be empty, truncated, "
        "or not a standard coordinate CIF/PDB (e.g. an mmCIF with no _atom_site "
        "coordinates, or a small-molecule/restraint CIF). Try re-exporting as a "
        "standard PDB or mmCIF with coordinates, or check the file isn't empty."
    )


def parse_structure(path: str) -> ParsedStructure:
    """Parse a CIF or PDB file into chains + candidate ligands."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise ValueError(f"'{path}' does not exist or is empty.")

    st = _read_any(path)
    st.setup_entities()
    model = st[0]

    out = ParsedStructure(path=path)
    for chain in model:
        polymer = chain.get_polymer()
        ptype = polymer.check_polymer_type() if len(polymer) else None

        if ptype in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            kind = "protein"
        elif ptype in (
            gemmi.PolymerType.Dna,
            gemmi.PolymerType.Rna,
            gemmi.PolymerType.DnaRnaHybrid,
        ):
            kind = "nucleic"
        else:
            kind = "other"

        seq = polymer.make_one_letter_sequence() if len(polymer) else ""
        seq = "".join(ch for ch in seq if ch.isalpha())

        if len(polymer):
            out.chains.append(
                ParsedChain(
                    chain_id=chain.name,
                    kind=kind,
                    sequence=seq,
                    n_residues=len(polymer),
                )
            )

        for res in chain.get_ligands():
            out.ligands.append(
                ParsedLigand(chain_id=chain.name, resname=res.name, n_atoms=len(res))
            )

    # Fallback: if gemmi found no polymers at all (rare; happens when entities
    # aren't set up well), treat every chain with >1 standard residue as protein.
    if not out.chains:
        for chain in model:
            seq = ""
            for res in chain:
                code = gemmi.find_tabulated_residue(res.name)
                if code and code.is_amino_acid():
                    seq += gemmi.find_tabulated_residue(res.name).one_letter_code.upper()
            if seq:
                out.chains.append(
                    ParsedChain(chain.name, "protein", seq, len(seq))
                )

    return out


def summary(parsed: ParsedStructure) -> str:
    lines = [f"Structure: {parsed.path}", "Chains:"]
    if not parsed.chains:
        lines.append("  (no polymer chains detected)")
    for c in parsed.chains:
        preview = (c.sequence[:40] + "...") if len(c.sequence) > 40 else c.sequence
        lines.append(f"  [{c.chain_id}] {c.kind:8s} {c.n_residues:>4d} res  {preview}")
    if parsed.ligands:
        lines.append("Ligand / non-polymer residues:")
        for lg in parsed.ligands:
            lines.append(f"  [{lg.chain_id}] {lg.resname:>4s}  ({lg.n_atoms} atoms)")
    else:
        lines.append("Ligand / non-polymer residues: none detected")
    return "\n".join(lines)
