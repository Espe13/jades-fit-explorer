#!/usr/bin/env python3
"""Build the static JADES fit-explorer site from Prospector posterior files.

Usage:
    python build_site.py --config config.yaml [--jobs 8] [--force] [--no-figures]

For each run listed in config.yaml this script:
  * scans the results directory for posterior npz files,
  * generates the four per-galaxy figures (corner, SED, line ratios, KL),
  * writes per-galaxy JSON detail files and a per-run catalog.json,
  * merges in rows from the master FITS catalogue and (optionally) the
    run's summary CSV.

Re-running is incremental: figures are only regenerated when the npz is
newer than the existing figure (or with --force). Adding a new fit run is
just a new entry in config.yaml + re-run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figures
from fits_lite import read_fits_table
from priors import priors_from_config

FIG_KINDS = ("corner", "sed", "lines", "kl")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.str_, str)):
        return str(x)
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return jsonable(x.item())
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def q163(x):
    q16, q50, q84 = np.percentile(np.asarray(x, float), [16, 50, 84])
    return float(q16), float(q50), float(q84)


def load_summary_csv(path, id_col="galaxy_id"):
    """Load a summary CSV into {galaxy_id: row_dict} (values as float where possible)."""
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                gid = int(float(row[id_col]))
            except (KeyError, ValueError):
                continue
            clean = {}
            for k, v in row.items():
                if v is None or v == "":
                    clean[k] = None
                    continue
                try:
                    fv = float(v)
                    clean[k] = fv if np.isfinite(fv) else None
                except ValueError:
                    clean[k] = v
            rows[gid] = clean
    return rows


def load_master_catalog(cfg):
    """Load the master FITS catalogue into {id: {col: value}}."""
    path = os.path.expanduser(cfg.get("catalog_fits") or "")
    if not path or not os.path.exists(path):
        print(f"  [warn] catalogue FITS not found: {path} - galaxy pages will "
              "lack coordinates / z_spec")
        return {}, []
    tbl = read_fits_table(path)
    id_col = cfg.get("catalog_id_column", "ID")
    want = cfg.get("catalog_columns") or [c for c in tbl if c != id_col]
    ids = np.asarray(tbl[id_col])
    ids = np.where(np.isfinite(ids.astype(float)), ids, -1).astype(np.int64) \
        if ids.dtype.kind == "f" else ids.astype(np.int64)
    out = {}
    for i, gid in enumerate(ids):
        out[int(gid)] = {c: jsonable(tbl[c][i]) for c in want if c in tbl}
    return out, [c for c in want if c in tbl]


# ----------------------------------------------------------------------------
# per-galaxy processing
# ----------------------------------------------------------------------------

def agn_flags(lines):
    """Narrow-line AGN diagnostics from observed line fluxes.

    Criteria: classical BPT (Kewley+01), R3S2/VO87 (Kewley+01),
    R3O1 (Mazzolari+25 eq. 3), the three [O III]4363 diagnostics of
    Mazzolari+24 (O3Hg vs O32 / Ne3O2 / O33), and [Ne IV]2424 detection
    (Mazzolari+25). Lines must be detected at S/N >= 3 (>= 5 for NeIV).
    Being above a demarcation is sufficient, not necessary, for AGN.
    """
    L = {l["name"]: l for l in lines}

    def f(name, snmin=3.0):
        l = L.get(name)
        if l and l.get("obs") and l.get("err") and l["obs"] > 0 \
           and l["err"] > 0 and l["obs"] >= snmin * l["err"]:
            return l["obs"]
        return None

    def lg(a, b):
        return math.log10(a / b) if a and b else None

    ha, hb, hg = f("Ba-alpha 6563"), f("Ba-beta 4861"), f("Ba-gamma 4341")
    o3, o3a = f("[O III] 5007"), f("[O III] 4363")
    n2f, o1f = f("[N II] 6584"), f("[O I] 6300")
    s2f = (f("[S II] 6716") or 0) + (f("[S II] 6731") or 0) or None
    o2f, ne3 = f("[O II] 3726"), f("[Ne III] 3869")  # [O II] = blended doublet
    tags = []
    R3 = lg(o3, hb)
    N2, S2, O1 = lg(n2f, ha), lg(s2f, ha), lg(o1f, ha)
    if R3 is not None and N2 is not None and (
            N2 >= 0.47 or R3 > 0.61 / (N2 - 0.47) + 1.19):
        tags.append("BPT (Kewley+01)")
    if R3 is not None and S2 is not None and (
            S2 >= 0.32 or R3 > 0.72 / (S2 - 0.32) + 1.30):
        tags.append("R3S2/VO87 (Kewley+01)")
    if R3 is not None and O1 is not None and (
            O1 >= 0.15 or R3 > 2.5 + 2.65 / (O1 - 0.15)):
        tags.append("R3O1 (Mazzolari+25)")
    Y = lg(o3a, hg)
    if Y is not None:
        X = lg(o3, o2f)
        if X is not None and Y > (0.55 * X - 0.95 if X > 0.84
                                  else 0.1 * X - 0.57):
            tags.append("O3Hg-O32 (Mazzolari+24)")
        X = lg(ne3, o2f)
        if X is not None and Y > (0.48 * X - 0.42 if X > -0.07
                                  else 0.2 * X - 0.44):
            tags.append("O3Hg-Ne3O2 (Mazzolari+24)")
        X = lg(o3, o3a)
        if X is not None and Y > -1.1 * X + 1.47:
            tags.append("O3Hg-O33 (Mazzolari+24)")
    if f("Blnd 2424", snmin=5.0):
        tags.append("[Ne IV]2424 detected (Mazzolari+25)")
    return tags


def process_galaxy(task):
    """Worker: build figures + detail JSON for one galaxy. Returns catalog row."""
    (npz_path, gid, run, out_data, out_assets, asset_url, cutout_url, snap_url,
     cat_row, sum_row, prior_spec, fitsmap_tpl, corner_dpi, force,
     no_figures, refigure) = task
    t0 = time.time()
    post = figures.load_posterior(npz_path)
    ex = post["extra"]
    names = post["param_names"]
    theta = post["theta"]
    # per-galaxy redshift prior: dist 'zspec_gauss' resolves from the
    # catalogue z_Spec / z_Spec_flag (A/B: sigma_ab*(1+z); C: sigma_c*(1+z);
    # otherwise uniform over [lo, hi]) - mirrors run_jades.decide_zred_treatment
    zspec_spec = (prior_spec or {}).get("zred")
    if zspec_spec and zspec_spec.get("dist") == "zspec_gauss":
        prior_spec = dict(prior_spec)
        lo = float(zspec_spec.get("lo", 0.0))
        hi = float(zspec_spec.get("hi", 20.0))
        zs = (cat_row or {}).get("z_Spec")
        fl = str((cat_row or {}).get("z_Spec_flag") or "").strip().upper()
        try:
            zs = float(zs)
        except (TypeError, ValueError):
            zs = float("nan")
        if fl in ("A", "B", "C") and np.isfinite(zs) and zs > 0:
            sig = float(zspec_spec.get("sigma_ab", 0.01) if fl in ("A", "B")
                        else zspec_spec.get("sigma_c", 0.02)) * (1.0 + zs)
            prior_spec["zred"] = {"dist": "normal", "mean": zs, "sigma": sig,
                                  "lo": lo, "hi": hi}
        else:
            prior_spec["zred"] = {"dist": "uniform", "lo": lo, "hi": hi}
    priors = priors_from_config(prior_spec)

    fig_dir = Path(out_assets)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {k: fig_dir / f"{gid}_{k}.webp" for k in FIG_KINDS}

    src_mtime = os.path.getmtime(npz_path)

    def stale(p, kind=None):
        if kind is not None and kind in refigure:
            return True
        png = Path(str(p).replace(".webp", ".png"))
        tgt = p if p.exists() else (png if png.exists() else None)
        return force or tgt is None or os.path.getmtime(tgt) < src_mtime

    have_lines = figures.has_lines(post)
    if not no_figures:
        if stale(fig_paths["corner"], "corner"):
            figures.fig_corner(post, fig_paths["corner"], dpi=corner_dpi)
        if stale(fig_paths["sed"], "sed"):
            figures.fig_sed(post, fig_paths["sed"])
        if have_lines and stale(fig_paths["lines"], "lines"):
            figures.fig_lines(post, fig_paths["lines"])
        if stale(fig_paths["kl"], "kl"):
            figures.fig_kl(post, fig_paths["kl"], priors=priors)

    # ---- parameter table (sampled params + derived samples) ----------------
    kl = {r["name"]: r for r in figures.kl_per_parameter(post, priors=priors)}
    params = []
    for i, name in enumerate(names):
        q16, q50, q84 = q163(theta[:, i])
        params.append({
            "name": name, "label": figures.param_label(name),
            "q16": q16, "q50": q50, "q84": q84,
            "kl_bits": jsonable(kl.get(name, {}).get("kl_bits")),
            "kl_fallback": bool(kl.get(name, {}).get("fallback", False)),
        })
    derived = []
    d = np.load(npz_path, allow_pickle=True)
    for key in d.files:
        if key.startswith("samples/") and key[8:] not in names and key != "samples/theta":
            arr = np.asarray(d[key], float)
            if arr.ndim == 1 and arr.size == theta.shape[0]:
                q16, q50, q84 = q163(arr)
                derived.append({"name": key[8:], "q16": q16, "q50": q50, "q84": q84})

    # ---- photometry / lines tables -----------------------------------------
    filts = [str(f) for f in ex["phot_filters"]]
    wl = [figures.filter_pivot_um(f) for f in filts]
    obs_p = np.asarray(ex["obs_phot"], float) * figures.MAGGIE_TO_NJY
    err_p = np.asarray(ex["obs_phot_err"], float) * figures.MAGGIE_TO_NJY
    pp = np.asarray(ex["pp_phot_pp"], float) * figures.MAGGIE_TO_NJY
    p16, p50, p84 = np.percentile(pp, [16, 50, 84], axis=0)
    photometry = [{
        "filter": f, "wl_um": jsonable(w),
        "obs_njy": jsonable(o), "err_njy": jsonable(e),
        "pred_njy": jsonable(m), "pred_lo": jsonable(lo), "pred_hi": jsonable(hi),
        "chi": jsonable((o - m) / e) if e else None,
    } for f, w, o, e, m, lo, hi in zip(filts, wl, obs_p, err_p, p50, p16, p84)]

    lines = []
    if have_lines:
        lnames = [str(l) for l in ex["line_names"]]
        lwl = np.asarray(ex["line_wavelengths"], float)
        obs_l = np.asarray(ex["obs_lines"], float)
        err_l = np.asarray(ex["obs_lines_err"], float)
        lpp = np.asarray(ex["pp_lines_pp"], float)
        l16, l50, l84 = np.percentile(lpp, [16, 50, 84], axis=0)
        lines = [{
            "name": n, "wl_A": jsonable(w),
            "obs": jsonable(o), "err": jsonable(e),
            "pred": jsonable(m), "pred_lo": jsonable(lo), "pred_hi": jsonable(hi),
            "ratio": jsonable(m / o) if o else None,
            "chi": jsonable((o - m) / e) if e else None,
        } for n, w, o, e, m, lo, hi in zip(lnames, lwl, obs_l, err_l, l50, l16, l84)]

    # ---- goodness of fit ----------------------------------------------------
    gof = {"logZ": jsonable(ex.get("logZ")), "logZ_err": jsonable(ex.get("logZ_err"))}
    chi_p = np.array([p["chi"] for p in photometry if p["chi"] is not None])
    chi_l = np.array([l["chi"] for l in lines if l["chi"] is not None])
    gof["chi2_phot_med"] = jsonable(np.sum(chi_p ** 2)) if chi_p.size else None
    gof["chi2_lines_med"] = jsonable(np.sum(chi_l ** 2)) if chi_l.size else None
    if sum_row:
        for k in ("chi2_phot", "chi2_lines", "chi2_total", "chi2_nu",
                  "n_data", "n_params", "dof", "logZ", "logZ_err"):
            if sum_row.get(k) is not None:
                gof[k] = sum_row[k]

    # summary-derived quantities (fesc, MUV, nion, ...) when a CSV exists
    summary_quants = []
    if sum_row:
        skip = {"galaxy_id", "redshift", "z_Spec", "wall_time_s", "n_steps"}
        for k, v in sum_row.items():
            if k in skip or k.startswith(("e_", "E_")) or v is None:
                continue
            if k.startswith("jwst_") or k.startswith("hst_"):
                continue
            if isinstance(v, float):
                summary_quants.append({
                    "name": k, "value": v,
                    "err_lo": sum_row.get(f"e_{k}"), "err_hi": sum_row.get(f"E_{k}"),
                })

    ra = cat_row.get("RA_1") if cat_row else None
    dec = cat_row.get("DEC_1") if cat_row else None
    fitsmap_url = None
    if fitsmap_tpl and ra is not None and dec is not None:
        fitsmap_url = fitsmap_tpl.format(ra=ra, dec=dec)

    zred_i = names.index("zred") if "zred" in names else None
    zred = q163(theta[:, zred_i]) if zred_i is not None else (None, None, None)

    detail = {
        "id": gid, "run": run,
        "model": jsonable(post.get("model")), "engine": jsonable(post.get("engine")),
        "n_samples": int(theta.shape[0]),
        "wall_time_s": jsonable(post.get("wall_time_s")),
        "z_input": jsonable(ex.get("redshift")),
        "zred": {"q16": zred[0], "q50": zred[1], "q84": zred[2]},
        "catalog": cat_row or {},
        "fitsmap_url": fitsmap_url,
        "agn": agn_flags(lines),
        "params": params, "derived": derived,
        "photometry": photometry, "lines": lines,
        "gof": gof, "summary": summary_quants,
        # only link figures whose file actually exists, so a --no-figures
        # build of new galaxies never publishes dead image links
        "figures": {**{k: f"{asset_url}/{gid}_{k}" for k in FIG_KINDS
                       if (k != "lines" or have_lines) and
                       (fig_paths[k].exists() or
                        fig_paths[k].with_suffix(".png").exists())},
                    **({"cutout": cutout_url} if cutout_url else {}),
                    **({"fitsmap": snap_url} if snap_url else {})},
        "npz_file": os.path.basename(npz_path),
    }
    gal_dir = Path(out_data) / "galaxies"
    gal_dir.mkdir(parents=True, exist_ok=True)
    with open(gal_dir / f"{gid}.json", "w") as f:
        json.dump(jsonable(detail), f)

    # ---- catalogue row -------------------------------------------------------
    def pq(name):
        for p in params:
            if p["name"] == name:
                return p["q50"], p["q16"], p["q84"]
        for p in derived:
            if p["name"] == name:
                return p["q50"], p["q16"], p["q84"]
        return None, None, None

    def plain_line(lab):
        rep = {"Ba-alpha": "Hα", "Ba-beta": "Hβ", "Ba-gamma": "Hγ",
               "Ba-delta": "Hδ", "Pa-alpha": "Paα", "Pa-beta": "Paβ",
               "Pa-gamma": "Paγ", "Pa-delta": "Paδ"}
        out = str(lab)
        for k, v in rep.items():
            out = out.replace(k, v)
        return re.sub(r"(\d+)\.\d+A", r"\1", out)

    row = {"id": gid,
           "ra": jsonable(ra), "dec": jsonable(dec),
           "tier": (cat_row or {}).get("TIER"),
           "z_spec": (cat_row or {}).get("z_Spec"),
           "z_flag": (cat_row or {}).get("z_Spec_flag"),
           "agn": agn_flags(lines),
           "logZ_evidence": gof.get("logZ")}
    # key emission lines for the catalogue-wide predicted-vs-observed figures
    _LINEKEYS = {"[O III] 5007": "oiii", "Ba-alpha 6563": "ha", "Ba-beta 4861": "hb"}
    for l in lines:
        short = _LINEKEYS.get(l["name"])
        if short:
            row[f"{short}_obs"] = l["obs"]
            row[f"{short}_err"] = l["err"]
            row[f"{short}_pred"] = l["pred"]
            row[f"{short}_pred_lo"] = l["pred_lo"]
            row[f"{short}_pred_hi"] = l["pred_hi"]
    # per-line / per-filter pulls (pred - obs)/sigma for the sample-wide
    # pull-distribution figures
    lc = {plain_line(l["name"]): jsonable(round(-l["chi"], 4))
          for l in lines if l.get("chi") is not None}
    pc = {p["filter"].replace("jwst_", "").upper(): jsonable(round(-p["chi"], 4))
          for p in photometry if p.get("chi") is not None}
    if lc:
        row["line_chi"] = lc
    if pc:
        row["phot_chi"] = pc
    for key, col in (("zred", "zred"), ("logmass", "logmass"), ("Z", "logzsol"),
                     ("gas_logz", "gas_logz"),
                     ("gas_logu", "gas_logu"), ("SFR10", "sfr10"),
                     ("SFR100", "sfr100"), ("frac_obrun", "frac_obrun")):
        v50, v16, v84 = pq(key)
        row[col] = v50
        if v50 is not None:
            row[col + "_lo"], row[col + "_hi"] = v16, v84
    if sum_row:
        for k in ("chi2_nu", "fesc", "MUV", "xion"):
            if sum_row.get(k) is not None:
                row[k] = sum_row[k]
    print(f"    {run}/{gid}  ({time.time()-t0:.1f}s)")
    return jsonable(row)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def load_te_metallicity_map(cfg):
    """{Unique_ID: (oh, lo, hi)} from the external MZR release, if configured."""
    ov = cfg.get("overlays") or {}
    path = os.path.expanduser(ov.get("mzr_fits", "") or "")
    if not path or not os.path.exists(path):
        return {}
    t = read_fits_table(path)
    oh_col = next((c for c in t if "log(O/H)" in c and not c.startswith(("e_", "E_"))), None)
    uid_col = next((c for c in t if "unique" in c.lower()), None)
    if not (oh_col and uid_col):
        return {}
    e_lo = t.get(f"e_{oh_col}")
    e_hi = t.get(f"E_{oh_col}")
    out = {}
    for i, u in enumerate(t[uid_col]):
        oh = float(t[oh_col][i])
        if not np.isfinite(oh):
            continue
        lo = oh - float(e_lo[i]) if e_lo is not None and np.isfinite(e_lo[i]) else None
        hi = oh + float(e_hi[i]) if e_hi is not None and np.isfinite(e_hi[i]) else None
        out[str(u)] = (oh, lo, hi)
    return out


def build_run(run_cfg, cfg, master, te_map, out_root, jobs, force, no_figures,
              refigure=frozenset()):
    name = run_cfg["name"]
    results_dir = Path(os.path.expanduser(run_cfg["results_dir"]))
    pattern = run_cfg.get("pattern", r"posterior_jades_(?P<id>\d+)_.*\.npz$")
    rx = re.compile(pattern)
    if not results_dir.is_dir():
        print(f"  run '{name}': results_dir does not exist yet ({results_dir})"
              " - skipped")
        return None
    files = sorted(p for p in results_dir.iterdir()
                   if p.suffix == ".npz" and rx.search(p.name))
    print(f"  run '{name}': {len(files)} posterior files in {results_dir}")
    if not files:
        return None

    sum_rows = {}
    scsv = run_cfg.get("summary_csv")
    if scsv and os.path.exists(os.path.expanduser(scsv)):
        sum_rows = load_summary_csv(os.path.expanduser(scsv))
        print(f"    summary CSV: {len(sum_rows)} rows")
    elif scsv:
        print(f"    [warn] summary CSV not found: {scsv}")

    out_data = Path(out_root) / "data" / "runs" / name
    prior_spec = {**(cfg.get("priors") or {}), **(run_cfg.get("priors") or {})}
    fitsmap_tpl = run_cfg.get("fitsmap_url", cfg.get("fitsmap_url"))
    corner_dpi = int((cfg.get("figures") or {}).get("corner_dpi", 110))

    # asset placement: local by default; sharded / external for large samples
    assets_output = Path(os.path.expanduser(
        cfg.get("assets_output", str(Path(out_root) / "assets"))))
    shards = int(cfg.get("asset_shards", 1))
    base_url = cfg.get("asset_base_url", "assets")
    cutout_root = Path(out_root) / "assets" / "cutouts"

    tasks = []
    for p in files:
        gid = int(rx.search(p.name).group("id"))
        shard = gid % shards
        sub = ((f"shard{shard}/" if shards > 1 else "") + name
               if "{run}" not in base_url else name)
        out_assets = assets_output / sub
        if "{run}" in base_url:
            # one repo PER RUN, e.g. https://user.github.io/jades-assets-{run}
            # the run name is already in the host path, so do not repeat it
            asset_url = base_url.format(shard=shard, run=name).rstrip("/")
        elif "{" in base_url:
            # hash-sharded, e.g. https://user.github.io/jades-assets-{shard}
            asset_url = base_url.format(shard=shard, run=name).rstrip("/") + "/" + name
        else:
            asset_url = base_url.rstrip("/") + "/" + sub
        cut = cutout_root / f"{gid}.webp"
        cutout_url = f"assets/cutouts/{gid}" if (cut.exists() or
                     cut.with_suffix(".png").exists()) else None
        snap = Path(out_root) / "assets" / "fitsmap" / f"{gid}.webp"
        snap_url = f"assets/fitsmap/{gid}" if snap.exists() else None
        tasks.append((str(p), gid, name, str(out_data), str(out_assets),
                      asset_url, cutout_url, snap_url, master.get(gid),
                      sum_rows.get(gid), prior_spec, fitsmap_tpl,
                      corner_dpi, force, no_figures, refigure))

    if jobs > 1:
        with mp.Pool(jobs) as pool:
            rows = pool.map(process_galaxy, tasks)
    else:
        rows = [process_galaxy(t) for t in tasks]
    rows = [r for r in rows if r]
    rows.sort(key=lambda r: r["id"])

    # cross-match Te-based metallicities from the external MZR release
    if te_map:
        n_x = 0
        for r in rows:
            u = str((master.get(r["id"]) or {}).get("Unique_ID", ""))
            if u in te_map:
                oh, lo, hi = te_map[u]
                r["oh_te"], r["oh_te_lo"], r["oh_te_hi"] = oh, lo, hi
                n_x += 1
        print(f"    Te-metallicity crossmatch: {n_x}/{len(rows)} galaxies")

    out_data.mkdir(parents=True, exist_ok=True)
    with open(out_data / "catalog.json", "w") as f:
        json.dump({"run": name, "rows": rows}, f)

    return {
        "name": name,
        "label": run_cfg.get("label", name),
        "description": run_cfg.get("description", ""),
        "model_description": run_cfg.get("model_description", ""),
        # flatten one level so a shared YAML anchor list can be mixed with
        # run-specific rows
        "priors_table": [e for entry in run_cfg.get("priors_table", [])
                         for e in (entry if entry and isinstance(entry[0], list)
                                   else [entry])],
        "n_galaxies": len(rows),
        "ids": [r["id"] for r in rows],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true",
                    help="regenerate figures even if up to date")
    ap.add_argument("--refigure", default="",
                    help="comma list of figure kinds to force-regenerate even "
                         "when up to date (corner,sed,lines,kl) - e.g. "
                         "--refigure kl after changing the priors config")
    ap.add_argument("--no-figures", action="store_true",
                    help="only rebuild JSON data, skip figure generation")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="only build these run names")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # resolve a relative `output:` against the CONFIG FILE's directory, never
    # the current working directory (".resolve()" first would defeat the check)
    out_root = Path(os.path.expanduser(cfg.get("output", "../site")))
    if not out_root.is_absolute():
        out_root = (Path(args.config).resolve().parent / out_root).resolve()
    print(f"output: {out_root}")

    print("loading master catalogue ...")
    master, _ = load_master_catalog(cfg)
    print(f"  {len(master)} catalogue rows")
    te_map = load_te_metallicity_map(cfg)
    if te_map:
        print(f"  {len(te_map)} Te-metallicity entries from MZR release")

    run_metas = []
    for run_cfg in cfg["runs"]:
        if args.runs and run_cfg["name"] not in args.runs:
            continue
        meta = build_run(run_cfg, cfg, master, te_map, out_root, args.jobs,
                         args.force, args.no_figures,
                         frozenset(k.strip() for k in
                                   (args.refigure or "").split(",") if k.strip()))
        if meta:
            run_metas.append(meta)

    # merge with pre-existing runs.json so partial builds don't drop runs
    runs_path = out_root / "data" / "runs.json"
    existing = {}
    if runs_path.exists():
        with open(runs_path) as f:
            existing = {r["name"]: r for r in json.load(f).get("runs", [])}
    for m in run_metas:
        existing[m["name"]] = m
    runs_out = {
        "site_title": cfg.get("site_title", "JADES fit explorer"),
        "site_subtitle": cfg.get("site_subtitle", ""),
        "runs": list(existing.values()),
    }
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_path, "w") as f:
        json.dump(runs_out, f)

    # optional external overlay datasets for the catalogue-wide figures
    ov = cfg.get("overlays") or {}
    if ov.get("mzr_fits"):
        path = os.path.expanduser(ov["mzr_fits"])
        if os.path.exists(path):
            t = read_fits_table(path)
            # expected columns: 12+log(O/H), log(Mstar), z_Spec (Curti/Isobe-style release)
            oh_col = next((c for c in t if "log(O/H)" in c and not
                           c.startswith(("e_", "E_"))), None)
            m_col = next((c for c in t if "log(Mstar)" in c and not
                          c.startswith(("e_", "E_"))), None)
            z_col = next((c for c in t if c.lower().startswith("z")), None)
            if oh_col and m_col:
                good = np.isfinite(t[oh_col]) & np.isfinite(t[m_col])
                mzr = {
                    "label": ov.get("mzr_label", "MZR release"),
                    "logm": jsonable(np.round(t[m_col][good], 4)),
                    "oh": jsonable(np.round(t[oh_col][good], 4)),
                    "z": jsonable(np.round(t[z_col][good], 4)) if z_col else None,
                }
                od = out_root / "data" / "overlays"
                od.mkdir(parents=True, exist_ok=True)
                with open(od / "mzr.json", "w") as f:
                    json.dump(mzr, f)
                print(f"  overlay: {good.sum()} MZR points from {os.path.basename(path)}")
        else:
            print(f"  [warn] overlays.mzr_fits not found: {path}")

    # cross-run galaxy index (which runs contain each galaxy)
    gal_index = {}
    for r in runs_out["runs"]:
        for gid in r["ids"]:
            gal_index.setdefault(str(gid), []).append(r["name"])
    with open(out_root / "data" / "galaxy_index.json", "w") as f:
        json.dump(gal_index, f)

    print(f"done: {len(runs_out['runs'])} run(s) in {runs_path}")


if __name__ == "__main__":
    main()
