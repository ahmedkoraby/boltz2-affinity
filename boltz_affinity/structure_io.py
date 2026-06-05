"""Read a provided structure (CIF or PDB) and prepare Boltz-2 inputs:
per-chain one-letter protein sequences, candidate ligands, and a sanitized
template CIF that Boltz-2's template parser will accept.
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


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _one_letter(resname: str) -> Optional[str]:
    info = gemmi.find_tabulated_residue(resname)
    if info and info.is_amino_acid():
        return info.one_letter_code.upper()
    return None


def _read_any(path: str) -> "gemmi.Structure":
    """Robustly read a coordinate file; retry by forcing format on failure."""
    st = gemmi.read_structure(path)
    if len(st) > 0:
        return st

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
        "or not a standard coordinate CIF/PDB."
    )


def _atom_site_columns(block):
    """Return a dict of _atom_site columns we care about (or None)."""
    def col(*names):
        for n in names:
            c = block.find_loop("_atom_site." + n)
            if c and len(c):
                return list(c)
        return None
    return {
        "comp":  col("label_comp_id", "auth_comp_id"),
        "asym":  col("auth_asym_id", "label_asym_id"),
        "seqid": col("auth_seq_id", "label_seq_id"),
        "aname": col("label_atom_id", "auth_atom_id"),
        "elem":  col("type_symbol"),
        "x": col("Cartn_x"), "y": col("Cartn_y"), "z": col("Cartn_z"),
        "group": col("group_PDB"),
    }


def _manual_atom_site_parse(path: str) -> Optional[ParsedStructure]:
    """Build chains/sequences directly from the _atom_site loop."""
    try:
        doc = gemmi.cif.read(path)
    except Exception:
        return None
    clean = gemmi.cif.as_string

    for block in doc:
        c = _atom_site_columns(block)
        if not (c["comp"] and c["asym"] and c["seqid"]):
            continue
        n = len(c["comp"])
        group = c["group"] or ["ATOM"] * n

        out = ParsedStructure(path=path)
        by_chain: dict = {}
        atom_counts: dict = {}
        seen = set()
        for i in range(n):
            ch = clean(c["asym"][i]); rid = clean(c["seqid"][i]); rn = clean(c["comp"][i])
            het = clean(group[i]).upper() == "HETATM" if i < len(group) else False
            key = (ch, rid, rn)
            atom_counts[key] = atom_counts.get(key, 0) + 1
            if key not in seen:
                seen.add(key)
                by_chain.setdefault(ch, []).append((rid, rn, het))

        for ch, reslist in by_chain.items():
            seq = "".join(filter(None, (_one_letter(rn) for _, rn, het in reslist if not het)))
            n_res = sum(1 for _, _, het in reslist if not het)
            if seq:
                out.chains.append(ParsedChain(ch, "protein", seq, n_res))
            for rid, rn, het in reslist:
                if het and _one_letter(rn) is None and rn not in ("HOH", "WAT"):
                    out.ligands.append(ParsedLigand(ch, rn, atom_counts.get((ch, rid, rn), 0)))
        if out.chains:
            return out
    return None


def rebuild_structure_from_atom_site(path: str):
    """Construct a gemmi.Structure from the raw _atom_site loop (with xyz)."""
    try:
        doc = gemmi.cif.read(path)
    except Exception:
        return None
    clean = gemmi.cif.as_string

    for block in doc:
        c = _atom_site_columns(block)
        if not (c["comp"] and c["asym"] and c["seqid"] and c["aname"]
                and c["x"] and c["y"] and c["z"]):
            continue
        n = len(c["comp"])
        group = c["group"] or ["ATOM"] * n
        st = gemmi.Structure()
        model = gemmi.Model("1")
        chains: dict = {}
        cur_key = None
        cur_res = None
        for i in range(n):
            ch_id = clean(c["asym"][i]); rid = clean(c["seqid"][i]); rn = clean(c["comp"][i])
            an = clean(c["aname"][i]).strip('"')
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
                res.het_flag = "H" if clean(group[i]).upper() == "HETATM" else "A"
                chain.add_residue(res)
                cur_res = chain[-1]
                cur_key = key
            atom = gemmi.Atom()
            atom.name = an
            try:
                atom.element = gemmi.Element(clean(c["elem"][i])) if c["elem"] else gemmi.Element(an[0])
            except Exception:
                atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(clean(c["x"][i])), float(clean(c["y"][i])), float(clean(c["z"][i])))
            cur_res.add_atom(atom)
        for ch in chains.values():
            model.add_chain(ch)
        st.add_model(model)
        if len(st) and len(st[0]):
            st.setup_entities()
            return st
    return None


def _sanitize_protein_template(st):
    """Protein-only structure, each chain renumbered contiguously from 1.
    Boltz-2's template parser indexes into the entity sequence by residue
    position, so gaps / insertion codes / hetero residues cause
    `IndexError: list index out of range`. Stripping to amino acids and
    renumbering 1..N avoids that."""
    new = gemmi.Structure()
    model = gemmi.Model("1")
    src = st[0]
    for chain in src:
        nc = gemmi.Chain(chain.name)
        idx = 0
        for res in chain:
            if _one_letter(res.name) is None:
                continue
            idx += 1
            nr = gemmi.Residue()
            nr.name = res.name
            nr.seqid = gemmi.SeqId(str(idx))
            nr.het_flag = "A"
            seen_atoms = set()
            for atom in res:
                if atom.name in seen_atoms:
                    continue
                seen_atoms.add(atom.name)
                na = gemmi.Atom()
                na.name = atom.name
                na.element = atom.element
                na.pos = atom.pos
                na.occ = atom.occ or 1.0
                na.b_iso = atom.b_iso
                nr.add_atom(na)
            nc.add_residue(nr)
        if len(nc):
            model.add_chain(nc)
    new.add_model(model)
    new.setup_entities()
    try:
        new.assign_label_seq_id()
    except Exception:
        pass
    return new


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse_structure(path: str) -> ParsedStructure:
    """Parse a CIF or PDB into chains + candidate ligands (robust to odd files)."""
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
        elif ptype in (gemmi.PolymerType.Dna, gemmi.PolymerType.Rna,
                       gemmi.PolymerType.DnaRnaHybrid):
            kind = "nucleic"
        else:
            kind = "other"
        seq = polymer.make_one_letter_sequence() if len(polymer) else ""
        seq = "".join(ch for ch in seq if ch.isalpha())
        if len(polymer):
            out.chains.append(ParsedChain(chain.name, kind, seq, len(polymer)))
        for res in chain.get_ligands():
            out.ligands.append(ParsedLigand(chain.name, res.name, len(res)))

    if not out.chains:
        manual = _manual_atom_site_parse(path)
        if manual is not None and manual.chains:
            return manual
    return out


def ensure_template_cif(path: str) -> str:
    """Return a Boltz-readable, sanitized mmCIF path for templating.
    Always emits a protein-only, contiguously-numbered CIF."""
    st = None
    try:
        st = gemmi.read_structure(path)
    except Exception:
        st = None
    if not (st is not None and len(st) > 0):
        st = rebuild_structure_from_atom_site(path)
    if not (st is not None and len(st) > 0):
        raise ValueError(
            f"Could not read coordinates from '{path}'. Upload a standard PDB, "
            "or disable templating."
        )
    clean = _sanitize_protein_template(st)
    if len(clean) == 0 or len(clean[0]) == 0:
        raise ValueError(f"No protein residues found in '{path}' to build a template.")
    out = os.path.splitext(path)[0] + "_template.cif"
    clean.make_mmcif_document().write_file(out)
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
