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


def _one_letter(resname: str) -> Optional[str]:
    info = gemmi.find_tabulated_residue(resname)
    if info and info.is_amino_acid():
        return info.one_letter_code.upper()
    return None


def _manual_atom_site_parse(path: str) -> Optional[ParsedStructure]:
    """Last-resort parser: read the _atom_site loop directly from the CIF and
    build chains/sequences ourselves. Handles files whose coordinate records are
    present but which gemmi's Structure builder returns as 0 models."""
    try:
        doc = gemmi.cif.read(path)
    except Exception:
        return None

    for block in doc:
        def col(*names):
            for n in names:
                c = block.find_loop("_atom_site." + n)
                if c and len(c):
                    return list(c)
            return None

        comp  = col("label_comp_id", "auth_comp_id")
        chain = col("auth_asym_id", "label_asym_id")
        seqid = col("auth_seq_id", "label_seq_id")
        group = col("group_PDB")
        if not (comp and chain and seqid):
            continue

        n = len(comp)
        group = group or ["ATOM"] * n
        clean = gemmi.cif.as_string

        # ordered unique residues per chain
        seen, residues = set(), []   # residues: (chain, seqid, comp, is_het)
        atom_counts: dict = {}
        for i in range(n):
            ch = clean(chain[i]); rid = clean(seqid[i]); rn = clean(comp[i])
            grp = clean(group[i]).upper() if i < len(group) else "ATOM"
            key = (ch, rid, rn)
            atom_counts[key] = atom_counts.get(key, 0) + 1
            if key not in seen:
                seen.add(key)
                residues.append((ch, rid, rn, grp == "HETATM"))

        out = ParsedStructure(path=path)
        by_chain: dict = {}
        for ch, rid, rn, is_het in residues:
            by_chain.setdefault(ch, []).append((rid, rn, is_het))

        for ch, reslist in by_chain.items():
            seq = "".join(filter(None, (_one_letter(rn) for _, rn, het in reslist if not het)))
            n_res = sum(1 for _, _, het in reslist if not het)
            if seq:
                out.chains.append(ParsedChain(ch, "protein", seq, n_res))
            for rid, rn, het in reslist:
                if het and _one_letter(rn) is None and rn not in ("HOH", "WAT"):
                    out.ligands.append(
                        ParsedLigand(ch, rn, atom_counts.get((ch, rid, rn), 0))
                    )
        if out.chains:
            return out
    return None


def parse_structure(path: str) -> ParsedStructure:
    """Parse a CIF or PDB file into chains + candidate ligands."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise ValueError(f"'{path}' does not exist or is empty.")

    try:
        st = _read_any(path)
    except ValueError:
        manual = _manual_atom_site_parse(path)
        if manual is not None and manual.chains:
            return manual
        raise
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
