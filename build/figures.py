"""Per-galaxy diagnostic figures from Prospector posterior .npz files.

Generates four figures per (galaxy, run):
  corner  - posterior corner plot of all sampled parameters
  sed     - observed vs posterior-predicted photometry + model spectrum band
  lines   - predicted / observed emission-line flux ratios
  kl      - per-parameter information gain KL(posterior || prior) in bits

All figures are written as WebP (falls back to PNG if Pillow lacks WebP).
Only numpy + matplotlib are required.
"""
from __future__ import annotations

import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
COL_OBS = "#333333"      # observed data: near-black
COL_MODEL = "#3465ad"    # model / predicted: blue
COL_BAND = "#a8c4e8"     # posterior band fill
COL_ACCENT = "#b5442d"   # highlight (bad points, reference lines)
COL_GRID = "#dddddd"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#666666",
    "axes.labelcolor": "#222222",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
})

_corner_cmap = LinearSegmentedColormap.from_list(
    "cornerblue", ["#ffffff", "#c9d9ef", "#6f97c9", "#2c5991", "#16334f"]
)

MAGGIE_TO_NJY = 3631.0e9  # 1 maggie = 3631 Jy

# Greek/pretty labels for common Prospector parameters
_LABELS = {
    "logmass": r"$\log\,M_\star/\mathrm{M}_\odot$",
    "Z": r"$\log\,Z_\star/\mathrm{Z}_\odot$",
    "zred": r"$z$",
    "gas_logz": r"$\log\,Z_\mathrm{gas}/\mathrm{Z}_\odot$",
    "gas_logu": r"$\log\,U$",
    "diffuse_tau_kc": r"$\tau_\mathrm{diff}$",
    "diffuse_dust_index": r"$n_\mathrm{dust}$",
    "tau_bc_fraction": r"$\tau_\mathrm{BC}/\tau_\mathrm{diff}$",
    "frac_obrun": r"$f_\mathrm{obrun}$",
    "eline_scaling": r"$s_\mathrm{eline}$",
    "fesc": r"$f_\mathrm{esc}$",
}


def param_label(name: str) -> str:
    m = re.match(r"logsfr_ratios\[(\d+)\]", name)
    if m:
        return rf"$r_{{{m.group(1)}}}$"
    return _LABELS.get(name, name.replace("_", " "))


def prettify_line(label: str) -> str:
    """Compact emission-line label for axis ticks."""
    lab = str(label)
    lab = lab.replace("Ba-alpha", r"H$\alpha$").replace("Ba-beta", r"H$\beta$")
    lab = lab.replace("Ba-gamma", r"H$\gamma$").replace("Ba-delta", r"H$\delta$")
    lab = lab.replace("Pa-alpha", r"Pa$\alpha$").replace("Pa-beta", r"Pa$\beta$")
    lab = lab.replace("Pa-gamma", r"Pa$\gamma$").replace("Pa-delta", r"Pa$\delta$")
    lab = re.sub(r"(\d+)\.\d+A", r"\1", lab)      # 4685.64A -> 4685
    lab = re.sub(r"(\d)\.(\d+)um", lambda m: m.group(1) + "." + m.group(2) + r"$\mu$m", lab)
    return lab


def filter_pivot_um(name: str) -> float:
    """Approximate pivot wavelength (micron) from a filter name like jwst_f356w."""
    m = re.search(r"f(\d{3})(w|m|n|lp|w2)?$", name.lower())
    if not m:
        return np.nan
    val = int(m.group(1))
    if name.lower().startswith(("jwst", "nircam", "miri")):
        return val / 100.0
    return val / 1000.0  # HST convention: f435w -> 0.435 um


# ----------------------------------------------------------------------------
# npz loading
# ----------------------------------------------------------------------------

def load_posterior(path):
    """Load a posterior npz into a convenient dict."""
    d = np.load(path, allow_pickle=True)
    out = {"path": str(path)}
    out["param_names"] = [str(p) for p in d["param_names"]]
    out["theta"] = np.asarray(d["samples/theta"], dtype=float)
    ex = {}
    for k in d.files:
        if k.startswith("extra/"):
            ex[k[6:]] = d[k]
    out["extra"] = ex
    for k in ("galaxy_id", "model", "engine", "wall_time_s", "n_steps"):
        if k in d.files:
            out[k] = d[k].item() if d[k].shape == () else d[k]
    return out


# ----------------------------------------------------------------------------
# Corner plot
# ----------------------------------------------------------------------------

def fig_corner(post, out_path, dpi=110):
    theta = post["theta"]
    names = post["param_names"]
    n = theta.shape[1]
    size = max(1.05 * n, 6.0)
    fig, axes = plt.subplots(n, n, figsize=(size, size))
    lims = [np.percentile(theta[:, i], [0.5, 99.5]) for i in range(n)]
    lims = [(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)) for lo, hi in lims]

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_axis_off()
                continue
            if i == j:
                x = theta[:, i]
                ax.hist(x, bins=40, range=lims[i], color=COL_BAND,
                        edgecolor=COL_MODEL, linewidth=0.4,
                        histtype="stepfilled", density=True)
                q16, q50, q84 = np.percentile(x, [16, 50, 84])
                for q, ls in ((q16, ":"), (q50, "--"), (q84, ":")):
                    ax.axvline(q, color=COL_OBS, lw=0.7, ls=ls)
                ax.set_yticks([])
            else:
                H, xe, ye = np.histogram2d(
                    theta[:, j], theta[:, i], bins=35,
                    range=[lims[j], lims[i]])
                Hs = _smooth2d(H)
                ax.pcolormesh(xe, ye, Hs.T, cmap=_corner_cmap, rasterized=True)
                _contours(ax, Hs, xe, ye)
                ax.set_ylim(lims[i])
            ax.set_xlim(lims[j])
            ax.tick_params(labelsize=6, length=2, width=0.5)
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(param_label(names[j]), fontsize=7)
                for lb in ax.get_xticklabels():
                    lb.set_rotation(45)
            if j > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(param_label(names[i]), fontsize=7)

    fig.subplots_adjust(hspace=0.06, wspace=0.06,
                        left=0.05, right=0.99, bottom=0.05, top=0.99)
    _save(fig, out_path, dpi=dpi)


def _smooth2d(H, passes=2):
    """Cheap separable smoothing (avoids a scipy dependency in hot loop)."""
    K = np.array([0.25, 0.5, 0.25])
    for _ in range(passes):
        H = np.apply_along_axis(lambda r: np.convolve(r, K, mode="same"), 0, H)
        H = np.apply_along_axis(lambda r: np.convolve(r, K, mode="same"), 1, H)
    return H


def _contours(ax, H, xe, ye):
    Hf = H.flatten()
    idx = np.argsort(Hf)[::-1]
    csum = np.cumsum(Hf[idx])
    csum /= csum[-1]
    levels = []
    for frac in (0.393, 0.865):  # 1-sigma, 2-sigma in 2D
        cut = Hf[idx][np.searchsorted(csum, frac)]
        levels.append(cut)
    levels = sorted(set(levels))
    if len(levels) < 1:
        return
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    ax.contour(xc, yc, H.T, levels=levels, colors="#16334f",
               linewidths=0.5, alpha=0.8)


# ----------------------------------------------------------------------------
# SED figure
# ----------------------------------------------------------------------------

def fig_sed(post, out_path, dpi=150):
    ex = post["extra"]
    filts = [str(f) for f in ex["phot_filters"]]
    wl = np.array([filter_pivot_um(f) for f in filts])
    obs = np.asarray(ex["obs_phot"], float) * MAGGIE_TO_NJY
    err = np.asarray(ex["obs_phot_err"], float) * MAGGIE_TO_NJY
    pp = np.asarray(ex["pp_phot_pp"], float) * MAGGIE_TO_NJY  # (ndraw, nband)
    p16, p50, p84 = np.percentile(pp, [16, 50, 84], axis=0)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.0, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    # posterior spectrum band
    spec_top = None
    if "pp_spec_pp" in ex and "pp_spec_wave" in ex:
        wave_um = np.asarray(ex["pp_spec_wave"], float) / 1e4
        lo_w = max(0.3, 0.75 * np.nanmin(wl))
        hi_w = min(30.0, 1.6 * np.nanmax(wl))
        sel = (wave_um > lo_w) & (wave_um < hi_w)
        if sel.sum() > 10:
            # pp_spec_pp is the CONTINUUM ONLY; the nebular emission lines are
            # stored separately in pp_spec_line_pp. The photometry prediction
            # (pp_phot_pp) already contains both, so the plotted spectrum must
            # add them too or it sits below the predicted photometry in every
            # line-dominated band.
            spec_raw = np.asarray(ex["pp_spec_pp"], float)[:, sel]
            if "pp_spec_line_pp" in ex:
                lines_raw = np.asarray(ex["pp_spec_line_pp"], float)[:, sel]
                if lines_raw.shape == spec_raw.shape:
                    spec_raw = spec_raw + lines_raw
            spec = spec_raw * _spec_to_njy(spec_raw, obs)
            w = wave_um[sel]
            s16, s50, s84 = np.percentile(spec, [16, 50, 84], axis=0)
            ax.fill_between(w, s16, s84, color=COL_BAND, alpha=0.55, lw=0,
                            label="model spectrum (16–84%)", zorder=1)
            ax.plot(w, s50, color=COL_MODEL, lw=0.7, alpha=0.9, zorder=2)
            spec_top = float(np.nanmax(s84)) if np.isfinite(s84).any() else None

    # predicted photometry
    ax.errorbar(wl, p50, yerr=[p50 - p16, p84 - p50], fmt="s",
                ms=6, mfc="white", mec=COL_MODEL, mew=1.4,
                ecolor=COL_MODEL, elinewidth=1.0, capsize=0, zorder=3,
                label="predicted photometry")
    # observed photometry. Bands whose catalogue error exceeds the flux are
    # non-detections (the Kron error blows up where mosaic coverage is
    # marginal); drawing them as error bars would span the whole panel on a
    # log axis, so show them as upper limits instead. They carry essentially
    # no weight in the likelihood either way.
    # Reproduce the fit's own mask exactly: run_jades.py flags a band when
    # |flux| / sigma < phot_ul_snr_threshold, evaluated AFTER the error-floor
    # inflation (i.e. on the values stored here), and switches that band to a
    # one-sided chi^2. A threshold of 0 means the run used two-sided Gaussians
    # throughout, in which case nothing is drawn as an upper limit.
    ul_thr = float(ex.get("phot_ul_snr_threshold", 0.0) or 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(err > 0, np.abs(obs) / err, np.inf)
    nd = (snr < ul_thr) | (obs <= 0)   # masked in the fit, or unplottable on a log axis
    det = ~nd
    ax.errorbar(wl[det], obs[det], yerr=err[det], fmt="o", ms=5,
                mfc=COL_OBS, mec=COL_OBS, ecolor=COL_OBS, elinewidth=1.0,
                capsize=0, zorder=4, label="observed photometry")

    ax.set_xscale("log")
    ax.set_yscale("log")
    pos = obs[det & (obs > 0)] if det.any() else obs[obs > 0]
    if pos.size:
        phot_top = 8 * max(pos.max(), p84.max())
        # leave room for the emission-line peaks, but never let a single very
        # strong line squash the photometry (cap at 30x the photometric range)
        top = phot_top
        if spec_top and np.isfinite(spec_top):
            top = min(max(phot_top, 1.35 * spec_top), 30 * max(pos.max(), p84.max()))
        bot = 0.2 * pos.min()
        ax.set_ylim(bot, top)
        # 2-sigma upper limits for the non-detections, clipped into the panel
        if nd.any():
            ul = np.where(err > 0, np.maximum(obs, 0) + 2 * err, np.nan)[nd]
            ul = np.clip(ul, bot * 1.6, top * 0.88)
            ax.errorbar(wl[nd], ul, yerr=0.28 * ul, uplims=True, fmt="none",
                        ecolor=COL_OBS, elinewidth=1.0, alpha=0.75, zorder=4,
                        label=f"non-detection (2σ upper limit, S/N < {ul_thr:g})")
    ax.set_ylabel(r"$F_\nu$ [nJy]")
    ax.legend(loc="lower right", fontsize=8)
    zred = float(np.median(post["theta"][:, post["param_names"].index("zred")])) \
        if "zred" in post["param_names"] else float(ex.get("redshift", np.nan))
    ax.text(0.02, 0.96, rf"$z = {zred:.3f}$", transform=ax.transAxes,
            va="top", fontsize=9, color="#222222")

    # residual panel: chi = (obs - pred) / sigma
    chi = (obs - p50) / np.where(err > 0, err, np.nan)
    axr.axhline(0, color=COL_GRID, lw=1)
    axr.axhspan(-1, 1, color="#f0f0f0", zorder=0)
    axr.plot(wl[det], chi[det], "o", ms=4.5, color=COL_MODEL)
    if nd.any():   # non-detections: open symbols, they constrain nothing
        axr.plot(wl[nd], chi[nd], "o", ms=4.5, mfc="white", mec=COL_MODEL,
                 mew=1.0)
    bad = (np.abs(chi) > 3) & det
    if bad.any():
        axr.plot(wl[bad], chi[bad], "o", ms=4.5, color=COL_ACCENT)
    lim = max(3.5, np.nanmax(np.abs(chi)) * 1.15) if np.isfinite(chi).any() else 3.5
    axr.set_ylim(-lim, lim)
    axr.set_ylabel(r"$\chi$")
    axr.set_xlabel(r"observed wavelength [$\mu$m]")

    ticks = sorted(set(np.round(wl, 2)))
    axr.set_xticks(ticks)
    axr.set_xticklabels([f"{t:g}" for t in ticks], rotation=45, fontsize=7)
    axr.minorticks_off()
    _save(fig, out_path, dpi=dpi)


def _spec_to_njy(spec_raw, obs_njy):
    """Detect the model-spectrum flux unit and return the conversion to nJy.

    Prospector outputs are commonly stored either in maggies or in
    erg/s/cm^2/Hz. We pick whichever conversion lands the spectrum's
    median closest (in log space) to the observed photometry median.
    """
    med_spec = float(np.median(spec_raw[spec_raw > 0])) if np.any(spec_raw > 0) else np.nan
    med_obs = float(np.median(obs_njy[obs_njy > 0])) if np.any(obs_njy > 0) else np.nan
    candidates = {
        "maggies": MAGGIE_TO_NJY,
        "cgs_fnu": 1e32,   # erg/s/cm^2/Hz -> nJy
        "njy": 1.0,
    }
    if not (np.isfinite(med_spec) and np.isfinite(med_obs)):
        return MAGGIE_TO_NJY
    best = min(candidates.values(),
               key=lambda c: abs(np.log10(med_spec * c) - np.log10(med_obs)))
    return best


# ----------------------------------------------------------------------------
# Emission-line ratio figure
# ----------------------------------------------------------------------------

def has_lines(post):
    ex = post["extra"]
    return ("line_names" in ex and "pp_lines_pp" in ex
            and np.asarray(ex["line_names"]).size > 0)


def fig_lines(post, out_path, dpi=150):
    if not has_lines(post):
        return False
    ex = post["extra"]
    names = [prettify_line(l) for l in ex["line_names"]]
    obs = np.asarray(ex["obs_lines"], float)
    err = np.asarray(ex["obs_lines_err"], float)
    pp = np.asarray(ex["pp_lines_pp"], float)  # (ndraw, nline)
    p16, p50, p84 = np.percentile(pp, [16, 50, 84], axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p50 / obs
        rlo = p16 / obs
        rhi = p84 / obs
        obs_frac = np.abs(err / obs)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6.5, 0.42 * len(names)), 4.0))

    ok = np.isfinite(ratio) & (obs > 0)
    # y-range from the data (keep the log axis sane when a ratio is extreme)
    vals = np.concatenate([ratio[ok], rlo[ok & (rlo > 0)], rhi[ok]])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    ylo = min(10 ** np.floor(np.log10(max(vals.min(), 1e-6))), 0.1) if vals.size else 0.1
    yhi = max(10 ** np.ceil(np.log10(vals.max())), 10.0) if vals.size else 10.0

    # observational uncertainty band around ratio = 1 (clipped for log axis)
    band_lo = np.clip(1 - obs_frac, ylo, None)
    band_hi = 1 + obs_frac
    ax.axhline(1.0, color="#888888", lw=1.0, zorder=1)
    ax.bar(x, band_hi - band_lo, bottom=band_lo, width=0.72,
           color="#ececec", edgecolor="none", zorder=0,
           label=r"obs. uncertainty ($\pm1\sigma$)")

    ax.errorbar(x[ok], ratio[ok],
                yerr=[np.maximum(ratio - rlo, 0)[ok], np.maximum(rhi - ratio, 0)[ok]],
                fmt="o", ms=5, color=COL_MODEL, ecolor=COL_MODEL,
                elinewidth=1.0, capsize=0, zorder=3, label="predicted / observed")
    # flag strong outliers
    with np.errstate(invalid="ignore"):
        sig = np.abs(p50 - obs) / np.where(err > 0, err, np.inf)
    bad = ok & (sig > 3)
    if bad.any():
        ax.plot(x[bad], ratio[bad], "o", ms=5, color=COL_ACCENT, zorder=4,
                label=r"$|\Delta| > 3\sigma$")

    ax.set_yscale("log")
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("predicted / observed flux")
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.14), fontsize=7, ncol=3)
    _save(fig, out_path, dpi=dpi)


# ----------------------------------------------------------------------------
# KL (information gain) per parameter
# ----------------------------------------------------------------------------

def kl_per_parameter(post, priors=None, bins=40):
    """KL(posterior || prior) per parameter, in bits.

    ``priors`` maps parameter name -> object with .pdf(x) and .support()
    (see priors.py). Parameters without a spec fall back to a uniform
    prior over the sampled range (flagged with fallback=True).
    """
    from priors import UniformPrior, student_t_prior_from_extra

    theta = post["theta"]
    names = post["param_names"]
    ex = post["extra"]
    priors = dict(priors or {})

    # SFH ratio priors are recorded in the file itself
    st = student_t_prior_from_extra(ex)
    results = []
    for i, name in enumerate(names):
        x = theta[:, i]
        prior = priors.get(name)
        fallback = False
        if prior is None and name.startswith("logsfr_ratios") and st is not None:
            prior = st
        if prior is None:
            lo, hi = float(np.min(x)), float(np.max(x))
            pad = 1e-3 * (hi - lo) if hi > lo else 1e-6
            prior = UniformPrior(lo - pad, hi + pad)
            fallback = True

        lo_s, hi_s = prior.support()
        lo_b = max(float(np.min(x)), lo_s)
        hi_b = min(float(np.max(x)), hi_s)
        if hi_b <= lo_b:
            results.append({"name": name, "kl_bits": np.nan, "fallback": fallback})
            continue
        hist, edges = np.histogram(x, bins=bins, range=(lo_b, hi_b), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        q = np.array([prior.pdf(c) for c in centers])
        mask = (hist > 0) & (q > 0)
        kl_nats = float(np.sum(hist[mask] * np.log(hist[mask] / q[mask]) * width))
        results.append({"name": name,
                        "kl_bits": kl_nats / np.log(2.0),
                        "fallback": fallback})
    return results


def fig_kl(post, out_path, priors=None, dpi=150):
    res = kl_per_parameter(post, priors=priors)
    names = [param_label(r["name"]) for r in res]
    vals = np.array([r["kl_bits"] for r in res])
    fallback = np.array([r["fallback"] for r in res])

    order = np.argsort(np.nan_to_num(vals, nan=-1))[::-1]
    names = [names[i] for i in order]
    vals = vals[order]
    fallback = fallback[order]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.2, max(3.0, 0.24 * len(names))))
    colors = [COL_BAND if fb else COL_MODEL for fb in fallback]
    ax.barh(y, np.nan_to_num(vals), height=0.62, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("information gain  KL(posterior ‖ prior)  [bits]")
    ax.axvline(0, color="#999999", lw=0.8)
    for yi, v in zip(y, vals):
        if np.isfinite(v):
            ax.text(v + 0.02 * max(np.nanmax(vals), 1), yi, f"{v:.2f}",
                    va="center", fontsize=6.5, color="#444444")
    if fallback.any():
        ax.text(0.98, 0.02,
                "light bars: uniform prior over sampled range (no prior spec)",
                transform=ax.transAxes, ha="right", fontsize=6.5, color="#777777")
    ax.set_xlim(0, max(np.nanmax(vals) * 1.18, 0.5))
    fig.tight_layout()
    _save(fig, out_path, dpi=dpi)


# ----------------------------------------------------------------------------

def _save(fig, out_path, dpi=150):
    out_path = str(out_path)
    kw = {"dpi": dpi, "bbox_inches": "tight"}
    if out_path.endswith(".webp"):
        kw["pil_kwargs"] = {"quality": 82, "method": 4}
    try:
        fig.savefig(out_path, **kw)
    except (ValueError, KeyError):
        # Pillow without WebP support -> fall back to PNG
        fallback = re.sub(r"\.webp$", ".png", out_path)
        fig.savefig(fallback, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
