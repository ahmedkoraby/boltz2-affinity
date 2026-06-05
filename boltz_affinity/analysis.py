"""Locate and interpret Boltz-2 outputs (affinity + confidence + structure).

affinity_pred_value interpretation (per the Boltz docs):
    value = log10(IC50) with IC50 in micromolar (uM).
    -> IC50_uM = 10 ** value           (lower value = tighter binder)
    -> pIC50   = 6 - value             (standard pIC50, IC50 in molar)
    -> dG_kcal_per_mol approx = (6 - value) * 1.364   (non-standard; see README)

affinity_probability_binary in [0,1]: probability the ligand is a binder
(use for hit-vs-decoy discrimination, not for SAR ranking).

The ensemble reports `*_value` (averaged) plus `*_value1` / `*_value2` from the
two affinity sub-models.
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class AffinityResult:
    binder_chain: Optional[str]
    pred_value: float                 # log10(IC50 uM), averaged
    probability_binary: float
    pred_value1: Optional[float] = None
    pred_value2: Optional[float] = None
    probability_binary1: Optional[float] = None
    probability_binary2: Optional[float] = None

    # derived
    ic50_uM: float = float("nan")
    pic50: float = float("nan")
    dg_kcal_per_mol: float = float("nan")

    def __post_init__(self):
        self.ic50_uM = 10.0 ** self.pred_value
        self.pic50 = 6.0 - self.pred_value
        self.dg_kcal_per_mol = (6.0 - self.pred_value) * 1.364


def _find_prediction_dir(out_dir: str) -> str:
    cands = glob.glob(os.path.join(out_dir, "**", "predictions", "*"), recursive=True)
    cands = [c for c in cands if os.path.isdir(c)]
    if not cands:
        raise FileNotFoundError(f"No predictions/<name> directory under {out_dir}")
    return cands[0]


def load_affinity(out_dir: str, binder_chain: Optional[str] = None) -> AffinityResult:
    pred_dir = _find_prediction_dir(out_dir)
    matches = glob.glob(os.path.join(pred_dir, "affinity_*.json"))
    if not matches:
        raise FileNotFoundError(
            f"No affinity_*.json in {pred_dir}. Did the YAML include a "
            "`properties: - affinity: binder: <id>` block?"
        )
    with open(matches[0]) as fh:
        d = json.load(fh)
    return AffinityResult(
        binder_chain=binder_chain,
        pred_value=float(d["affinity_pred_value"]),
        probability_binary=float(d["affinity_probability_binary"]),
        pred_value1=d.get("affinity_pred_value1"),
        pred_value2=d.get("affinity_pred_value2"),
        probability_binary1=d.get("affinity_probability_binary1"),
        probability_binary2=d.get("affinity_probability_binary2"),
    )


def load_confidence(out_dir: str) -> dict:
    pred_dir = _find_prediction_dir(out_dir)
    matches = sorted(glob.glob(os.path.join(pred_dir, "confidence_*.json")))
    if not matches:
        return {}
    with open(matches[0]) as fh:
        return json.load(fh)


def top_structure(out_dir: str) -> Optional[str]:
    pred_dir = _find_prediction_dir(out_dir)
    for pat in ("*_model_0.cif", "*_model_0.pdb", "*.cif", "*.pdb"):
        m = sorted(glob.glob(os.path.join(pred_dir, pat)))
        if m:
            return m[0]
    return None


def report(res: AffinityResult) -> str:
    lines = [
        "=== Boltz-2 affinity ===",
        f"binder chain            : {res.binder_chain}",
        f"affinity_pred_value     : {res.pred_value:.3f}   (log10 IC50 in uM)",
        f"  -> IC50               : {res.ic50_uM:.3g} uM",
        f"  -> pIC50              : {res.pic50:.2f}",
        f"  -> dG (approx)        : {res.dg_kcal_per_mol:.2f} kcal/mol  (non-standard)",
        f"affinity_probability    : {res.probability_binary:.3f}   (P(binder), 0..1)",
    ]
    if res.pred_value1 is not None:
        lines.append(
            f"  ensemble values       : {res.pred_value1:.3f} / {res.pred_value2:.3f}"
        )
    lines.append(
        "Use probability_binary for hit-vs-decoy; pred_value for SAR / lead-opt."
    )
    return "\n".join(lines)
