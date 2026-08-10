#!/usr/bin/env python3
"""Shrink posterior .npz files ~10x for transfer, losslessly for the website.

Roughly 90 per cent of every ``posterior_*.npz`` is the posterior-predictive
spectrum stored per draw:

    extra/pp_spec_line_pp   (ndraw, nwave) float64   ~15 MB
    extra/pp_spec_pp        (ndraw, nwave) float32   ~ 8 MB

The site never uses the individual draws - ``figures.fig_sed`` only ever plots
the 16/50/84 percentiles of (continuum + lines). This script replaces those two
arrays with a single

    extra/pp_spec_pct       (3, nwave) float32       ~180 kB

holding exactly those percentiles of the summed spectrum, and casts
``pp_spec_wave`` to float32. EVERYTHING ELSE IS COPIED BYTE-FOR-BYTE, including
``samples/theta`` (the corner plots need the full chain), ``pp_phot_pp``,
``pp_lines_pp`` and the nested-sampling arrays.

Typical result: 25 MB -> ~2 MB per galaxy.

Run it ON THE CLUSTER, then rsync only the thinned directory:

    python thin_posteriors.py --in results_fesc --out results_fesc_thin
    # then, from the laptop:
    rsync -avh --info=progress2 --partial \
        tursa:.../results_fesc_thin/  ~/Desktop/PhD/Tursa/Tursa/jades_full/results_fesc/

The build pipeline reads either form: ``figures.py`` uses ``pp_spec_pct`` when
present and falls back to the full arrays otherwise, so thinned and un-thinned
files can sit side by side in the same results directory.

Only numpy is required. Existing outputs are skipped, so the run is resumable.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DROP = ("extra/pp_spec_pp", "extra/pp_spec_line_pp")
PCTS = (16.0, 50.0, 84.0)


def thin_one(args):
    src, dst, force = args
    src, dst = Path(src), Path(dst)
    if dst.exists() and not force:
        return ("skip", src.name, 0, 0)
    try:
        with np.load(src, allow_pickle=True) as d:
            keys = list(d.files)
            out = {}
            cont = d["extra/pp_spec_pp"] if "extra/pp_spec_pp" in keys else None
            line = d["extra/pp_spec_line_pp"] if "extra/pp_spec_line_pp" in keys else None
            for k in keys:
                if k in DROP:
                    continue
                v = d[k]
                if k == "extra/pp_spec_wave":
                    v = np.asarray(v, dtype=np.float32)
                out[k] = v
            if cont is not None:
                tot = np.asarray(cont, dtype=np.float64)
                if line is not None and np.shape(line) == np.shape(cont):
                    tot = tot + np.asarray(line, dtype=np.float64)
                if tot.ndim == 2:
                    pct = np.percentile(tot, PCTS, axis=0)
                else:                      # already thinned or single draw
                    pct = np.atleast_2d(tot)
                out["extra/pp_spec_pct"] = pct.astype(np.float32)
        tmp = dst.with_suffix(".tmp.npz")
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(tmp, **out)              # uncompressed: fast, rsync -z handles the rest
        os.replace(tmp, dst)
        return ("ok", src.name, src.stat().st_size, dst.stat().st_size)
    except Exception as e:                # noqa: BLE001 - report and continue
        return ("fail", f"{src.name}: {e}", 0, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, type=Path,
                    help="directory holding posterior_*.npz")
    ap.add_argument("--out", dest="dst", required=True, type=Path,
                    help="output directory for the thinned copies")
    ap.add_argument("--glob", default="posterior_*.npz")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true",
                    help="rewrite outputs that already exist")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N files (for a trial run)")
    a = ap.parse_args()

    files = sorted(a.src.glob(a.glob))
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit(f"no files matching {a.glob} in {a.src}")
    a.dst.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(a.dst / p.name), a.force) for p in files]

    n_ok = n_skip = n_fail = 0
    tot_in = tot_out = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (status, name, sin, sout) in enumerate(ex.map(thin_one, tasks, chunksize=4), 1):
            if status == "ok":
                n_ok += 1
                tot_in += sin
                tot_out += sout
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                print(f"  [fail] {name}")
            if i % 200 == 0 or i == len(tasks):
                shrink = (tot_in / tot_out) if tot_out else 0.0
                print(f"  [{i}/{len(tasks)}] ok={n_ok} skip={n_skip} fail={n_fail} "
                      f"| {tot_in/1e9:.1f} GB -> {tot_out/1e9:.1f} GB (x{shrink:.1f})",
                      flush=True)
    print(f"\ndone: {n_ok} written, {n_skip} skipped, {n_fail} failed -> {a.dst}")
    if tot_out:
        print(f"size: {tot_in/1e9:.1f} GB -> {tot_out/1e9:.1f} GB "
              f"({100*tot_out/tot_in:.1f}% of the original)")


if __name__ == "__main__":
    main()
