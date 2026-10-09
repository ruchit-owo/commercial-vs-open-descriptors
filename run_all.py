"""Reproduce the whole study from a clean checkout.

Runs the phases in order: curate, features, main comparison, matched budget,
practical cost, then figures. Each phase is a script under experiments/ that
reads config.yaml and writes into data/ or results/. Re-running is safe:
features are cached, and a phase that finds its outputs already present skips the
expensive work unless --force is given.

The alvaDesc export (phase 02) needs the commercial licence; every later phase
works from the cached feature CSVs and does not.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PHASES = [
    ("01_curate", "download and standardise every endpoint"),
    ("02_features", "compute alvaDesc first, then the open representations"),
    ("03_main_comparison", "the leak-free nested-CV comparison"),
    ("04_matched_budget", "rerun with every representation reduced to the same k"),
    ("05_practical_cost", "timing, failure and missingness accounting"),
    ("06_figures", "per-endpoint and paired-difference figures"),
]

EXPERIMENTS = Path(__file__).parent / "experiments"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="PHASE",
        help="run only these phases, e.g. --only 03_main_comparison",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute even when a phase's outputs already exist",
    )
    args = parser.parse_args()

    phases = PHASES
    if args.only:
        wanted = set(args.only)
        phases = [p for p in PHASES if p[0] in wanted]
        if not phases:
            print(f"No matching phase in {args.only}", file=sys.stderr)
            return 2

    for name, description in phases:
        script = EXPERIMENTS / f"{name}.py"
        print(f"\n=== {name}: {description} ===", flush=True)
        cmd = [sys.executable, str(script)]
        if args.force:
            cmd.append("--force")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Phase {name} failed (exit {result.returncode}).", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
