#!/usr/bin/env python3
"""Derive the manifold column store from the MIST/MILES fits.

    python build/build_manifold.py

Source
    ~/Desktop/PhD/Tools/SFMS/mist_run/results_sfms12_bursty/summary/
        sfms12_extract_bursty.csv      3190 galaxies x 140 columns
        ids_gold_mist_bursty.txt       the gold sub-sample

These are the CERIDWEN fits described in SFMS/PAPER_HANDOVER.md: MIST/MILES,
Chabrier, 12-bin non-parametric SFH with a rising-main-sequence prior, tree
``bursty``.  They replace the earlier BPASS 2.2 catalogue that the rest of the
site still serves, so the two are NOT interchangeable -- see the note the page
puts on the drill-down link.

Output (site/data/manifold/)
    manifest.json     dims (key, label, unit, center, scale, lo, hi, clo, chi),
                      colour options, categorical codes, per-run row counts
    <run>.f32         Float32 column-major, [n_cols][n_rows], raw values; the
                      page standardises with center/scale so tooltips can show
                      physical units
    <run>.meta.json   ids, categorical codes, sky positions

3190 galaxies x ~22 columns is ~280 kB, so there is no case for the
quantisation and byte-shuffling a 10^5-point dataset would need.
"""
from __future__ import annotations

import csv
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
OUT = SITE / "data" / "manifold"
def _find_src():
    """Locate the MIST/MILES summary directory.

    $SFMS_SUMMARY wins; otherwise try the usual laptop layout and the same
    tree reached through a mounted home, so the script runs unchanged from a
    sandbox that does not share the Mac's home directory.
    """
    import os
    cands = []
    env = os.environ.get("SFMS_SUMMARY")
    if env:
        cands.append(Path(env))
    rel = "Desktop/PhD/Tools/SFMS/mist_run/results_sfms12_bursty/summary"
    cands.append(Path.home() / rel)
    cands.append(Path.home() / "mnt" / rel.replace("Desktop/", ""))
    cands.append(ROOT.parent / "Tools/SFMS/mist_run/results_sfms12_bursty/summary")
    for c in cands:
        if (c / "sfms12_extract_bursty.csv").is_file():
            return c
    raise SystemExit(
        "could not find sfms12_extract_bursty.csv; set $SFMS_SUMMARY to the "
        "directory that holds it. Tried:\n  " + "\n  ".join(str(c) for c in cands))


SRC = _find_src()
EXTRACT = SRC / "sfms12_extract_bursty.csv"
GOLD = SRC / "ids_gold_mist_bursty.txt"

RUN = "sfms12_bursty"
RUN_LABEL = "MIST/MILES · bursty"

# (key, csv field, label, unit, log10 of the field?)
DIMS = [
    ("zred",       "zred",       "redshift",                "z",          False),
    ("logmass",    "logmass",    "log M★",             "M☉",    False),
    ("logsfr10",   "logsfr10",   "log SFR₁₀",     "M☉/yr", False),
    ("logsfr100",  "logsfr100",  "log SFR₁₀₀", "M☉/yr", False),
    ("logssfr10",  "logssfr10",  "log sSFR₁₀",    "1/yr",       False),
    ("burst",      "burst",      "burstiness b",            "dex",        False),
    ("Z",          "Z",          "log Z★",             "",           False),
    ("gas_logz",   "gas_logz",   "log Z_gas/Z☉",       "",           False),
    ("gas_logu",   "gas_logu",   "log U",                   "",           False),
    ("tau_dust",   "diffuse_tau_kc", "τ_diffuse",      "",           False),
    ("MUV",        "MUV",        "M_UV",                    "mag",        False),
    ("beta_uv",    "beta_uv",    "β_UV",               "",           False),
]
# available as colour / filter, but not as a projection axis
EXTRA = [
    ("chi2_nu",    "chi2_nu",    "log χ²_ν", "",    True),
    ("t50",        "t50_myr",    "log t₅₀",       "Myr", True),
    ("EW_Ha",      "EW_Ha",      "log EW(Hα)",         "Å", True),
    ("EW_O3Hb",    "EW_O3Hb",    "log EW([O III]+Hβ)", "Å", True),
    ("oh_te",      "oh_te",      "12 + log(O/H) [T_e]",     "",    False),
    ("ebv_balmer", "ebv_balmer", "E(B−V) Balmer",      "mag", False),
    ("dust_index", "diffuse_dust_index", "dust index",      "",    False),
    ("n_lines_det", "n_lines_det", "detected lines",        "",    False),
    ("z_spec",     "z_spec",     "z_spec",                  "",    False),
]
FLAG_CODE = {"A": 0, "B": 1, "C": 2}          # 3 = none (absent in this run)


def fnum(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def column(rows, field, is_log):
    out = []
    for r in rows:
        v = fnum(r.get(field, ""))
        if is_log and math.isfinite(v):
            v = math.log10(v) if v > 0 else math.nan
        out.append(v)
    return out


def robust_scale(vals):
    """Centre on the median, scale by half the 2-98 percentile span.

    A standard deviation would let a handful of railed objects (chi2_nu -> 342,
    beta_uv -> 7.2) compress every real galaxy into the middle of the plot.
    lo/hi are the true extremes; clo/chi are the percentiles a colour ramp
    must span.
    """
    g = sorted(v for v in vals if math.isfinite(v))
    if len(g) < 10:
        return 0.0, 1.0, 0.0, 1.0, 0.0, 1.0

    def q(p):
        return g[min(len(g) - 1, max(0, int(round(p * (len(g) - 1)))))]
    lo, hi, med = q(0.02), q(0.98), q(0.5)
    return med, (hi - lo) / 2.0 or 1.0, g[0], g[-1], lo, hi


def main():
    if not EXTRACT.is_file():
        raise SystemExit(f"not found: {EXTRACT}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(EXTRACT.open()))
    n = len(rows)
    gold = {l.strip() for l in GOLD.read_text().split() if l.strip()} \
        if GOLD.is_file() else set()

    cols, meta_dims, meta_extra = [], [], []
    for key, field, label, unit, is_log in DIMS:
        v = column(rows, field, is_log)
        nf = sum(1 for x in v if math.isfinite(x))
        if nf < 0.5 * n:
            print(f"  dim {key}: only {nf}/{n} finite - dropped")
            continue
        c, s, lo, hi, clo, chi = robust_scale(v)
        cols.append(v)
        meta_dims.append(dict(key=key, label=label, unit=unit, log=is_log,
                              center=c, scale=s, lo=lo, hi=hi,
                              clo=clo, chi=chi, n=nf))
    for key, field, label, unit, is_log in EXTRA:
        v = column(rows, field, is_log)
        nf = sum(1 for x in v if math.isfinite(x))
        if nf == 0:
            print(f"  extra {key}: no finite values - dropped")
            continue
        c, s, lo, hi, clo, chi = robust_scale(v)
        cols.append(v)
        meta_extra.append(dict(key=key, label=label, unit=unit, log=is_log,
                               center=c, scale=s, lo=lo, hi=hi,
                               clo=clo, chi=chi, n=nf))

    buf = bytearray()
    for v in cols:
        buf += struct.pack(f"<{n}f", *v)
    (OUT / f"{RUN}.f32").write_bytes(buf)

    n_gold = sum(1 for r in rows if r["galaxy_id"] in gold)
    meta = {
        "run": RUN, "n": n,
        "id":   [int(float(r["galaxy_id"])) for r in rows],
        "ra":   [fnum(r.get("ra", "")) for r in rows],
        "dec":  [fnum(r.get("dec", "")) for r in rows],
        "flag": [FLAG_CODE.get((r.get("zflag") or "").strip(), 3) for r in rows],
        "gold": [1 if r["galaxy_id"] in gold else 0 for r in rows],
        "tier": [r.get("tier") or "" for r in rows],
    }
    (OUT / f"{RUN}.meta.json").write_text(json.dumps(meta, separators=(",", ":")))

    manifest = {
        "site_title": "JADES SED manifold",
        "site_subtitle": ("MIST/MILES fits of 3 ≤ z ≤ 9 JADES galaxies "
                          "— scaling relations as rotating projections"),
        "provenance": ("CERIDWEN, MIST/MILES, Chabrier IMF, 12-bin "
                       "non-parametric SFH with a rising-main-sequence prior "
                       "(tree 'bursty'); NIRCam DR5 photometry + NIRSpec DR4 "
                       "lines."),
        "runs": [dict(name=RUN, label=RUN_LABEL, n=n, n_gold=n_gold,
                      dims=[d["key"] for d in meta_dims],
                      extra=[d["key"] for d in meta_extra])],
        "dims": meta_dims, "extra": meta_extra, "flag_code": FLAG_CODE,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # The previous BPASS column store would otherwise keep being served.
    # Move rather than delete: deletion is refused over the device bridge,
    # and keeping the old columns costs nothing.
    stale = sorted(OUT.glob("*_ns.*"))
    if stale:
        old_dir = OUT / "_superseded_bpass"
        old_dir.mkdir(exist_ok=True)
        for f in stale:
            f.replace(old_dir / f.name)
        print(f"  moved {len(stale)} superseded BPASS file(s) -> "
              f"{old_dir.name}/")

    print(f"\n  {RUN}: {n} rows ({n_gold} gold) x {len(cols)} columns "
          f"({len(buf)/1024:.0f} kB)")
    print(f"  dims : {', '.join(d['key'] for d in meta_dims)}")
    print(f"  extra: {', '.join(d['key'] for d in meta_extra)}")
    print(f"\nwrote {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
