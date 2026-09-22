"""
stage_arxiv_figures.py - copy the figures arxiv_short.tex includes into docs/arxiv_figures.

WHY THIS EXISTS

The short paper includes four of the eight figures make_figures.py produces, and arXiv
wants the source tree it compiles to be self-contained. Copying them by hand is exactly
the step that goes stale: make_figures.py regenerates a panel, docs/figures updates, and
the copy the paper actually compiles against does not - so the submitted PDF shows a
figure no longer produced by the code, with nothing reporting it.

Filenames are hyphenated on the way across. Underscores in an \\includegraphics argument
are a well-known way to lose an afternoon to a LaTeX error, and there is no LaTeX on the
machine this was written on to discover that with.

    python scripts/stage_arxiv_figures.py
"""

from __future__ import annotations

import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The figures arxiv_short.tex includes. Keep in step with it; the check below is what
# tells you when it has drifted.
FIGURES = (
    "fig0_alpha_validation.png",
    "fig0b_alpha_validation_af78.png",
    "fig2_autocorrelation.png",
    "fig6_estimator_tradeoff.png",
)


def main() -> int:
    src = os.path.join(_ROOT, "docs", "figures")
    dst = os.path.join(_ROOT, "docs", "arxiv_figures")
    tex = os.path.join(_ROOT, "docs", "arxiv_short.tex")
    os.makedirs(dst, exist_ok=True)

    source = open(tex, encoding="utf-8").read() if os.path.isfile(tex) else ""
    missing = []

    for name in FIGURES:
        staged = name.replace("_", "-")
        origin = os.path.join(src, name)
        if not os.path.isfile(origin):
            missing.append(name)
            continue
        shutil.copy2(origin, os.path.join(dst, staged))
        print(f"  {name} -> arxiv_figures/{staged}")
        if source and staged not in source:
            print(f"  NOTE: {staged} is staged but arxiv_short.tex does not include it.")

    # The other direction, which is the one that breaks a compile rather than merely
    # wasting a file: the paper asks for something nothing staged.
    for line in source.splitlines():
        if "\\includegraphics" not in line or "{" not in line:
            continue
        wanted = line.rsplit("{", 1)[-1].split("}")[0]
        if not os.path.isfile(os.path.join(dst, wanted)):
            print(f"  MISSING: arxiv_short.tex includes {wanted}, which is not staged.")
            missing.append(wanted)

    if missing:
        print(f"\n{len(missing)} problem(s). Run scripts/make_figures.py first.")
        return 1

    print(f"\n{len(FIGURES)} figures staged in docs/arxiv_figures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
