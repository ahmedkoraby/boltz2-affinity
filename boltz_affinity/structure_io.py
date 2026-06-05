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


def rebuild_structure_from_atom_site(path: str):
    """Build a gemmi.Structure from the raw _atom_site loop (coordinates + names).
    Used when gemmi can read the columns but won't assemble a model itself."""
    try:
        doc = gemmi.cif.read(path)
    except Exception:
        return None
    clean = gemmi.cif.as_string

    for block in doc:
        def col(*names):
            for n in names:
                c = block.find_loop("_atom_site." + n)
                if c and len(c):
                    return list(c)
            return None

        comp  = col("label_comp_id", "auth_comp_id")
        asym  = col("auth_asym_id", "label_asym_id")
        seqid = col("auth_seq_id", "label_seq_id")
        aname = col("label_atom_id", "auth_atom_id")
        elem  = col("type_symbol")
        x = col("Cartn_x"); y = col("Cartn_y"); z = col("Cartn_z")
        group = col("group_PDB")
        if not (comp and asym and seqid and aname and x and y and z):
            continue

        n = len(comp)
        group = group or ["ATOM"] * n
        st = gemmi.Structure()
        model = gemmi.Model("1")
        chains: dict = {}
        cur_key = None
        cur_res = None

        for i in range(n):
            ch_id = clean(asym[i]); rid = clean(seqid[i]); rn = clean(comp[i])
            an = clean(aname[i]).strip('"')
            if ch_id not in chains:
                chains[ch_id] = gemmi.Chain(ch_id)
                cur_key = None
            chain = chains[ch_id]
            key = (ch_id, rid, rn)
            if key != cur_key:
                res = gemmi.Residue()
                res.name = rn
                try:
                    res.seqid = gemmi.SeqId(rid)
                except Exception:
                    res.seqid = gemmi.SeqId(str(len(chain) + 1))
                res.het_flag = "H" if (group[i] and clean(group[i]).upper() == "HETATM") else "A"
                chain.add_residue(res)
                cur_res = chain[-1]
                cur_key = key
            atom = gemmi.Atom()
            atom.name = an
            try:
                atom.element = gemmi.Element(clean(elem[i])) if elem else gemmi.Element(an[0])
            except Exception:
                atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(clean(x[i])), float(clean(y[i])), float(clean(z[i])))
            cur_res.add_atom(atom)

        for ch in chains.values():
            model.add_chain(ch)
        st.add_model(model)
        if len(st) and len(st[0]):
            st.setup_entities()
            return st
    return None


def ensure_template_cif(path: str) -> str:
    """Return the path to a Boltz-readable mmCIF for templating.
    - readable CIF  -> returned as-is
    - readable PDB  -> converted to CIF
    - unreadable    -> rebuilt from _atom_site coordinates and written as CIF
    """
    ext = os.path.splitext(path)[1].lower()
    st = None
    try:
        st = gemmi.read_structure(path)
    except Exception:
        st = None

    if st is not None and len(st) > 0:
        if ext in (".cif", ".mmcif"):
            return path
        st.setup_entities()
        out = os.path.splitext(path)[0] + ".cif"
        st.make_mmcif_document().write_file(out)
        return out

    rb = rebuild_structure_from_atom_site(path)
    if rb is not None and len(rb) > 0:
        out = os.path.splitext(path)[0] + "_rebuilt.cif"
        rb.make_mmcif_document().write_file(out)
        return out

    raise ValueError(
        f"Could not produce a Boltz-readable template CIF from '{path}'. "
        "Try uploading a standard PDB instead, or disable templating."
    )
