"""Read a provided structure (CIF or PDB) and extract the information needed to
build a Boltz-2 input: per-chain one-letter protein sequences and any ligand
(HETATM / non-polymer) residues that could serve as the affinity binder.

Uses gemmi (a hard dependency of boltz, so it is always available alongside it).
"""
from __future__ import annotations

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


def parse_structure(path: str) -> ParsedStructure:
    """Parse a CIF or PDB file into chains + candidate ligands."""
    st = gemmi.read_structure(path)
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
        # gemmi marks gaps/unknowns; keep only standard one-letter codes for the
        # YAML sequence (Boltz wants the bare sequence).
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

        # non-polymer residues = candidate ligands
        for res in chain.get_ligands():
            out.ligands.append(
                ParsedLigand(
                    chain_id=chain.name,
                    resname=res.name,
                    n_atoms=len(res),
                )
            )

    return out


def summary(parsed: ParsedStructure) -> str:
    lines = [f"Structure: {parsed.path}", "Chains:"]
    for c in parsed.chains:
        preview = (c.sequence[:40] + "...") if len(c.sequence) > 40 else c.sequence
        lines.append(
            f"  [{c.chain_id}] {c.kind:8s} {c.n_residues:>4d} res  {preview}"
        )
    if parsed.ligands:
        lines.append("Ligand / non-polymer residues:")
        for lg in parsed.ligands:
            lines.append(
                f"  [{lg.chain_id}] {lg.resname:>4s}  ({lg.n_atoms} atoms)"
            )
    else:
        lines.append("Ligand / non-polymer residues: none detected")
    return "\n".join(lines)
