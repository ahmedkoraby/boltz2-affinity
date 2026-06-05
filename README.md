# Boltz-2 Affinity From Structure

Predict **binding affinity from a structure you already have** — including
**multimer CIFs** — with [Boltz-2](https://github.com/jwohlwend/boltz), while
keeping a one-click affinity toggle.

The original [boltz2-notebook](https://github.com/AtharvaTilewale/boltz2-notebook)
folds a protein from *sequence* and then predicts affinity. This repo solves the
other direction: **"I have a structure (often a multimer). Use it, and give me
the affinity for this ligand."**

---

## How it works (and an important nuance)

Boltz-2 does not score a fully pre-built complex directly — it always runs its
diffusion structure module. To make affinity respect a structure you provide, we
feed that structure through Boltz-2's **`templates:`** block:

- Every protein chain in your structure is **templated** against your CIF.
- Setting `force: true` + a `threshold` (Å) **restrains the predicted backbone**
  to your structure, so you get affinity on a pose derived from *your* coordinates
  rather than a fresh fold.
- The ligand is supplied as **SMILES or CCD** and flagged as the affinity
  `binder`.
- Optionally, a **pocket constraint** pins the ligand to the real binding site —
  recommended, because Boltz-2's affinity head is only reliable when the pocket
  and interface are correct.

```yaml
version: 1
sequences:
  - protein: { id: A, sequence: MVTPE... }   # auto-extracted from your CIF
  - protein: { id: B, sequence: MVTPE... }   # multimer: as many chains as you have
  - ligand:  { id: C, smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O' }
templates:
  - cif: my_complex.cif
    chain_id: [A, B]
    force: true
    threshold: 5.0
properties:
  - affinity: { binder: C }
```

### Affinity limitations you should know
- Affinity is for **one small-molecule ligand only** — no protein–protein
  affinity, no multiple ligands per request.
- Templates apply to **protein chains**; ligands are not templated.
- The affinity head **does not explicitly model cofactors, ions, water, or
  multimeric partners**. For a multimer, read the number as *"ligand vs. the
  templated pocket"* and validate against orthogonal methods (your MM/GBSA,
  ABFE, or MST data).
- Use **CIF** templates; PDB template support has been buggy upstream.
- Recent Boltz-2 builds reject ligands with **≥128 atoms** for affinity.

---

## Quick start

### Colab notebook (one-click affinity)
Open `Boltz2_Affinity_From_Structure.ipynb` in Google Colab, set the runtime to
a GPU, upload your CIF, set the ligand, and run. The notebook auto-detects chains
and sequences, builds the YAML, runs Boltz-2, and shows the affinity dashboard +
3D pose.

### Local / Compute Canada
```bash
pip install -r requirements.txt          # installs boltz, gemmi, etc.

python predict_affinity.py \
    --structure my_complex.cif \
    --ligand-smiles 'N[C@@H](Cc1ccc(O)cc1)C(=O)O' \
    --force --threshold 5.0 \
    --out results
```
On SLURM (Narval/Rorqual), request the GPU in your job script and either allow
outbound network for the first-run weight download or point `--cache` at a
pre-seeded `~/.boltz` cache.

### As a library
```python
from boltz_affinity import (parse_structure, ProteinChain, Ligand, TemplateSpec,
                            build_yaml_dict, write_yaml, run_boltz,
                            load_affinity, report)

parsed = parse_structure("my_complex.cif")
proteins = [ProteinChain(c.chain_id, c.sequence) for c in parsed.protein_chains]
lig = Ligand("L", smiles="N[C@@H](Cc1ccc(O)cc1)C(=O)O")
data = build_yaml_dict(proteins, [lig], binder_id="L",
                       templates=[TemplateSpec("my_complex.cif",
                                  chain_id=[p.id for p in proteins],
                                  force=True, threshold=5.0)])
write_yaml(data, "results/input.yaml")
run_boltz("results/input.yaml", "results", use_msa_server=True)
print(report(load_affinity("results", binder_chain="L")))
```

---

## Interpreting the output

| field | meaning | use for |
|-------|---------|---------|
| `affinity_pred_value` | `log10(IC50)`, IC50 in µM (lower = tighter) | SAR / lead-opt ranking |
| `affinity_probability_binary` | P(ligand is a binder), 0–1 | hit-vs-decoy discrimination |

Conversions (handled for you in `analysis.report`):
`IC50_µM = 10**value`, `pIC50 = 6 − value`, `ΔG ≈ (6 − value)·1.364 kcal/mol`
(the ΔG form is non-standard; use with caution when comparing to other tools).

Outputs land in `results/.../predictions/<name>/`:
`*_model_0.cif`, `confidence_*.json`, `affinity_*.json`, `pae/pde/plddt_*.npz`.

---

## Repo layout
```
boltz2-affinity-from-structure/
├── Boltz2_Affinity_From_Structure.ipynb   # Colab notebook (main deliverable)
├── predict_affinity.py                     # CLI for local/cluster runs
├── boltz_affinity/                          # importable library
│   ├── structure_io.py    # parse chains/sequences/ligands from CIF/PDB (gemmi)
│   ├── yaml_builder.py    # build the Boltz-2 YAML (templates + affinity + pocket)
│   ├── runner.py          # `boltz predict` wrapper
│   └── analysis.py        # parse + interpret affinity / confidence outputs
├── examples/affinity_from_template.yaml
├── requirements.txt
└── LICENSE
```

## Credits
Built on [Boltz-2](https://github.com/jwohlwend/boltz) (Passaro, Wohlwend et al.,
2025) and inspired by the Colab UX of
[boltz2-notebook](https://github.com/AtharvaTilewale/boltz2-notebook). MIT licensed.
