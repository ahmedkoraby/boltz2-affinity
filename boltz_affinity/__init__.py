"""boltz_affinity: feed a structure (CIF, incl. multimers) to Boltz-2 and
predict binding affinity for a ligand."""
from .yaml_builder import (
    ProteinChain,
    Ligand,
    TemplateSpec,
    PocketConstraint,
    build_yaml_dict,
    write_yaml,
)
from .structure_io import ensure_template_cif, rebuild_structure_from_atom_site, parse_structure, summary, ParsedStructure
from .runner import run_boltz, boltz_available
from .analysis import (
    load_affinity,
    load_confidence,
    top_structure,
    report,
    AffinityResult,
)

__all__ = [
    "ProteinChain", "Ligand", "TemplateSpec", "PocketConstraint",
    "build_yaml_dict", "write_yaml",
    "parse_structure", "summary", "ParsedStructure",
    "run_boltz", "boltz_available",
    "load_affinity", "load_confidence", "top_structure", "report",
    "AffinityResult",
]
__version__ = "0.1.0"
