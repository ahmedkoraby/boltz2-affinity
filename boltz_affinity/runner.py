"""Thin wrapper around the ``boltz predict`` CLI."""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional


def boltz_available() -> bool:
    return shutil.which("boltz") is not None


def run_boltz(
    yaml_path: str,
    out_dir: str,
    use_msa_server: bool = True,
    devices: int = 1,
    accelerator: str = "gpu",
    diffusion_samples: int = 1,
    recycling_steps: Optional[int] = None,
    sampling_steps: Optional[int] = None,
    output_format: str = "mmcif",
    cache: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    stream: bool = True,
) -> str:
    """Run Boltz-2 on a YAML file. Returns the output directory.

    Notes
    -----
    * ``--use_msa_server`` lets Boltz fetch MSAs automatically (needed for
      protein chains that don't carry an ``msa:`` path). Disable for offline /
      air-gapped clusters where you precompute MSAs.
    * On Compute Canada-style SLURM nodes, set ``accelerator='gpu'`` and request
      the GPU in your job script; the model weights download on first run, so
      ensure outbound network or a pre-seeded cache (``--cache``).
    """
    if not boltz_available():
        raise RuntimeError(
            "`boltz` CLI not found. Install with: pip install boltz -U"
        )

    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "boltz", "predict", yaml_path,
        "--out_dir", out_dir,
        "--accelerator", accelerator,
        "--devices", str(devices),
        "--diffusion_samples", str(diffusion_samples),
        "--output_format", output_format,
    ]
    if use_msa_server:
        cmd.append("--use_msa_server")
    if recycling_steps is not None:
        cmd += ["--recycling_steps", str(recycling_steps)]
    if sampling_steps is not None:
        cmd += ["--sampling_steps", str(sampling_steps)]
    if cache:
        cmd += ["--cache", cache]
    if extra_args:
        cmd += list(extra_args)

    print("Running:", " ".join(cmd), flush=True)
    if stream:
        proc = subprocess.run(cmd)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout)
        print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"boltz predict failed (exit {proc.returncode}).")
    return out_dir
