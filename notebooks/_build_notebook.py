"""Generate notebooks/latency_probe_colab.ipynb.

The notebook is checked in, so this only needs rerunning when the cells change.
Kept ASCII-only on purpose: notebook JSON travels through git, Colab, and browsers,
and a stray smart-quote or em dash is not worth the encoding risk.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "latency_probe_colab.ipynb")


def _lines(text):
    """Split into ipynb `source` form: every line keeps its trailing newline except
    the last. Jupyter concatenates the array elements verbatim, so stripping the
    newlines collapses the whole cell onto one line - which is a syntax error for
    code and a wall of run-together prose for markdown."""
    return text.splitlines(keepends=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text.strip())}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _lines(src.strip("\n"))}


CELLS = [
    md("""
# Latency probe: cross-hardware run

Runs this project's own `benchmarks/latency_probe.py` on a Colab GPU, so the
numbers are directly comparable to a local run. The methodology is not
reimplemented here - this notebook only clones, installs, runs, and exports.

**Before running:** Runtime > Change runtime type > T4 GPU.

Why a T4 specifically: it has tensor cores, and the development machine
(GTX 1650 Ti) does not, even though both report CUDA compute capability 7.5 -
Nvidia removed the tensor cores from the GTX 16-series. Locally, fp16 gave no
speedup and was sometimes slower. If fp16 helps on a T4, that isolates the cause
instead of leaving it an unexplained curiosity.

Runtime: roughly 10 minutes, most of it installing audiocraft.
"""),

    md("## 1. Confirm the GPU"),
    code("""
import torch

if not torch.cuda.is_available():
    raise RuntimeError("No GPU attached. Runtime > Change runtime type > T4 GPU, then rerun.")

props = torch.cuda.get_device_properties(0)
capability = torch.cuda.get_device_capability(0)

print(f"GPU        : {torch.cuda.get_device_name(0)}")
print(f"VRAM       : {props.total_memory / 1e9:.1f} GB")
print(f"Capability : {capability[0]}.{capability[1]}")
print(f"Torch      : {torch.__version__}")
"""),

    md("""
## 2. Clone and install

Installing `audiocraft` may prompt a runtime restart. If it does, restart and
rerun from this cell - the clone is idempotent.
"""),
    code("""
import os

if not os.path.isdir("Brain-Music-Therapy"):
    !git clone -q https://github.com/kirthankulkarni-bit/Brain-Music-Therapy.git

%pip install -q audiocraft
"""),

    md("""
## 3. Run the probe

Sections A and B measure the analysis path and DSP compute. No GPU is involved in
those, so they should roughly match the local run - they are included as a control.
Section C is the GPU-dependent part and is the reason for this notebook.
"""),
    code("""
import re

import torch

label = "colab-" + re.sub(r"[^a-z0-9]+", "-", torch.cuda.get_device_name(0).lower()).strip("-")
out = f"benchmarks/latency_{label}.json"
print(f"label: {label}")

!cd Brain-Music-Therapy && python benchmarks/latency_probe.py --label "$label" --durations 4 8 --trials 3 --out "$out"
"""),

    md("## 4. Results"),
    code("""
import json

import pandas as pd

results = json.load(open(f"Brain-Music-Therapy/{out}"))
hw = results["hardware"]

print(f"{hw['gpu_name']}  |  capability {hw['compute_capability']}"
      f"  |  tensor cores: {hw['has_tensor_cores']}")
print(f"end-to-end worst case: {results['end_to_end_worst_case_s']:.1f} s")
print()

pd.DataFrame(results["musicgen"])[
    ["precision", "duration_s", "median_generation_s", "realtime_factor", "faster_than_realtime"]
]
"""),

    md("""
## 5. Compare fp16 against fp32

The single number this notebook exists to produce. On the GTX 1650 Ti the speedup
was about 1.0x or worse; a tensor-core GPU should be clearly above 1.0x.
"""),
    code("""
rows = {(r["precision"], r["duration_s"]): r["median_generation_s"] for r in results["musicgen"]}
durations = sorted({d for _, d in rows})

print(f"{'duration':>9} {'fp32':>9} {'fp16':>9} {'speedup':>9}")
print("-" * 39)
for d in durations:
    fp32, fp16 = rows.get(("fp32", d)), rows.get(("fp16", d))
    if fp32 and fp16:
        print(f"{d:>8.0f}s {fp32:>8.2f}s {fp16:>8.2f}s {fp32 / fp16:>8.2f}x")
"""),

    md("""
## 6. Export

Download the JSON and commit it to `benchmarks/` next to the local result. Each
machine writes its own file, so the comparison table in the paper is built from
whichever files are present.
"""),
    code("""
from google.colab import files

files.download(f"Brain-Music-Therapy/{out}")
"""),
]

NOTEBOOK = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": CELLS,
}


def stub_magics(line):
    """Turn IPython magics into `pass`, keeping indentation, so cells can be compiled."""
    match = re.match(r"^(\s*)([!%].*)$", line)
    return f"{match.group(1)}pass  # {match.group(2)}" if match else line


def main():
    for i, cell in enumerate(CELLS):
        source = "".join(cell["source"])  # exactly how Jupyter reassembles it
        source.encode("ascii")  # fail loudly rather than shipping mojibake

        # Every line but the last must carry its newline, or the cell collapses.
        for line in cell["source"][:-1]:
            assert line.endswith("\n"), f"cell {i}: source line missing trailing newline"
        assert "\n" in source or len(cell["source"]) == 1, f"cell {i}: collapsed to one line"

        if cell["cell_type"] == "code":
            compile("".join(stub_magics(l) for l in source.splitlines(keepends=True)),
                    "<cell>", "exec")

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(NOTEBOOK, fh, indent=1)
        fh.write("\n")

    with open(OUT, encoding="utf-8") as fh:
        loaded = json.load(fh)

    print(f"wrote {OUT}")
    print(f"  {len(loaded['cells'])} cells, all ASCII, all code cells compile")


if __name__ == "__main__":
    main()
