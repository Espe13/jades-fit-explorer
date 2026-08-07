#!/usr/bin/env python3
"""RGB cutouts with aperture + slit overlays for the fit-explorer site.

For each galaxy (RA/Dec from the master catalogue) this extracts a cutout
from the same NIRCam mosaics your fitsmap is built from, composes an RGB
image (asinh stretch), and overlays

  * the Kron aperture ellipse (A_KRON / B_KRON / THETA_KRON from the
    master catalogue), and
  * optionally the NIRSpec MSA shutter/slit footprints, read from a CSV.

Configuration lives in config.yaml under ``cutouts:``, e.g.::

    cutouts:
      size_arcsec: 3.0
      kron_units: arcsec        # or 'pixel' if A_KRON/B_KRON are in pixels
      fields:
        - name: goods-s
          # any TIER containing this substring uses these mosaics
          tier_match: goods-s
          r: /path/to/goods-s/F444W_mosaic.fits
          g: /path/to/goods-s/F200W_mosaic.fits
          b: /path/to/goods-s/F090W_mosaic.fits
        - name: goods-n
          tier_match: goods-n
          r: /path/to/goods-n/F444W_mosaic.fits
          g: /path/to/goods-n/F200W_mosaic.fits
          b: /path/to/goods-n/F090W_mosaic.fits
      slits_csv: /path/to/slits.csv   # optional

``slits_csv`` columns: id, ra, dec, pa_deg, width_arcsec, height_arcsec
(one row per shutter; several rows per galaxy id draw several rectangles;
pa_deg is measured from North through East).

Run it standalone (writes site/assets/cutouts/<id>.webp):

    python cutouts.py --config config.yaml [--ids 2150 490] [--force]

The site build picks the images up automatically: if
``assets/cutouts/<id>.webp`` exists, the galaxy page shows it alongside
the fitsmap link. astropy is used when available; otherwise the built-in
numpy FITS reader + TAN WCS handles standard mosaics.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fits_lite import read_fits_table  # noqa: E402

APER_COLOR = "#7fd4a0"
SLIT_COLOR = "#f2c14e"


# ----------------------------------------------------------------------------
# image access (astropy if present, else fits_lite)
# ----------------------------------------------------------------------------

class MosaicImage:
    """A FITS mosaic opened lazily with a sky<->pixel transform."""

    def __init__(self, path, hdu=None):
        self.path = os.path.expanduser(str(path))
        self._astropy = None
        try:
            from astropy.io import fits as apfits
            from astropy.wcs import WCS as APWCS
            hdul = apfits.open(self.path, memmap=True)
            k = hdu if hdu is not None else self._first_image_hdu(hdul)
            self.data = hdul[k].data
            self._wcs = APWCS(hdul[k].header).celestial
            self._astropy = True
        except ImportError:
            from fits_lite import read_fits_image, TanWCS
            k = hdu if hdu is not None else 0
            try:
                self.data, hdr = read_fits_image(self.path, hdu=k)
            except ValueError:
                self.data, hdr = read_fits_image(self.path, hdu=k + 1)
            self._wcs = TanWCS(hdr)
            self._astropy = False

    @staticmethod
    def _first_image_hdu(hdul):
        for i, h in enumerate(hdul):
            if h.data is not None and getattr(h.data, "ndim", 0) >= 2:
                return i
        raise ValueError("no image HDU found")

    def sky2pix(self, ra, dec):
        if self._astropy:
            x, y = self._wcs.world_to_pixel_values(ra, dec)
            return float(x), float(y)
        x, y = self._wcs.sky2pix(ra, dec)
        return float(np.asarray(x).ravel()[0]), float(np.asarray(y).ravel()[0])

    @property
    def pixel_scale_arcsec(self):
        if self._astropy:
            from astropy.wcs.utils import proj_plane_pixel_scales
            return float(np.mean(proj_plane_pixel_scales(self._wcs)) * 3600.0)
        return self._wcs.pixel_scale_arcsec

    def cutout(self, ra, dec, size_arcsec):
        """Return (2-D array, x_center, y_center, pixscale) or None if off-image."""
        x, y = self.sky2pix(ra, dec)
        ps = self.pixel_scale_arcsec
        r = int(np.ceil(size_arcsec / 2.0 / ps))
        ny, nx = self.data.shape[-2:]
        if not (0 <= x < nx and 0 <= y < ny):
            return None
        x0, x1 = int(round(x)) - r, int(round(x)) + r + 1
        y0, y1 = int(round(y)) - r, int(round(y)) + r + 1
        pad = [max(0, -y0), max(0, y1 - ny), max(0, -x0), max(0, x1 - nx)]
        cut = np.asarray(self.data[max(0, y0):min(ny, y1),
                                   max(0, x0):min(nx, x1)], dtype=float)
        if any(pad):
            cut = np.pad(cut, ((pad[0], pad[1]), (pad[2], pad[3])),
                         constant_values=np.nan)
        cx = x - x0
        cy = y - y0
        return cut, cx, cy, ps


# ----------------------------------------------------------------------------
# hips2fits backend: fetch a WCS-calibrated cutout from a HiPS survey
# ----------------------------------------------------------------------------

HIPS2FITS = ("https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
             "?hips={hips}&ra={ra}&dec={dec}&fov={fov}&width={npix}"
             "&height={npix}&projection=TAN&format=fits")


def fetch_hips_image(hips_id, ra, dec, size_arcsec, pixscale_arcsec, cache_dir):
    """Download a cutout FITS from the CDS hips2fits service (cached).

    Returns a MosaicImage over the downloaded file. ``hips_id`` is a HiPS
    identifier, e.g. 'CDS/P/JWST/JADES/GOODS-S/F444W' or an ESA/other HiPS
    URL id - check https://aladin.cds.unistra.fr/hips/list for JWST surveys.
    """
    import urllib.request
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = hips_id.replace("/", "_").replace(":", "_")
    npix = int(np.ceil(size_arcsec / pixscale_arcsec))
    # oversize slightly so the overlay geometry never clips at the edge
    fov = (npix * pixscale_arcsec) / 3600.0
    fname = cache_dir / f"{safe}_{ra:.6f}_{dec:.6f}_{npix}.fits"
    if not fname.exists():
        url = HIPS2FITS.format(hips=urllib.parse.quote(hips_id, safe=""),
                               ra=ra, dec=dec, fov=fov, npix=npix)
        urllib.request.urlretrieve(url, fname)
    return MosaicImage(str(fname))


# ----------------------------------------------------------------------------
# RGB composition
# ----------------------------------------------------------------------------

def asinh_rgb(chans, Q=8.0, pct=99.7, nsigma_floor=10.0):
    """Noise-referenced Lupton-style asinh RGB from up to three 2-D arrays.

    Each channel is background-subtracted (median) and the common scale is
    the largest of {per-channel pct-percentile residual, nsigma_floor x MAD
    sigma}, so pure-noise fields stay dark instead of saturating, and a
    common vmax across channels preserves colour.
    """
    res, sigs, vmaxs = [], [], []
    for c in chans:
        a = np.where(np.isfinite(c), c, 0.0).astype(float)
        med = np.median(a)
        sig = 1.4826 * np.median(np.abs(a - med)) + 1e-30
        r = a - med
        res.append(r)
        sigs.append(sig)
        vmaxs.append(max(float(np.nanpercentile(r, pct)), nsigma_floor * sig))
    vmax = max(vmaxs)
    stack = np.clip(np.sum(res, axis=0) / len(res), 0, None) / vmax
    I = np.clip(stack, 1e-9, None)
    f = np.arcsinh(Q * I) / (I * np.arcsinh(Q))
    rgb = np.dstack([np.clip(np.clip(r, 0, None) / vmax * f, 0, 1) for r in res])
    return rgb


# ----------------------------------------------------------------------------
# overlays + rendering
# ----------------------------------------------------------------------------

def render_cutout(rgb, cx, cy, ps, out_path, gid,
                  kron=None, slits=(), size_arcsec=3.0, dpi=150):
    """kron: (a_arcsec, b_arcsec, theta_deg) semi-axes; slits: list of dicts."""
    npix = rgb.shape[0]
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.imshow(rgb, origin="lower", interpolation="nearest")

    if kron is not None:
        a, b, th = kron
        ax.add_patch(Ellipse((cx, cy), 2 * a / ps, 2 * b / ps, angle=th,
                             fill=False, edgecolor=APER_COLOR, lw=1.3))
    for s in slits:
        w = s["width_arcsec"] / ps
        h = s["height_arcsec"] / ps
        # position angle from North (y axis) through East; imshow x is
        # typically -RA so the rotation sign matches standard orientation
        sx, sy = s.get("_x", cx), s.get("_y", cy)
        rect = Rectangle((sx - w / 2, sy - h / 2), w, h,
                         angle=s.get("pa_deg", 0.0),
                         rotation_point="center",
                         fill=False, edgecolor=SLIT_COLOR, lw=1.1, ls="-")
        ax.add_patch(rect)

    # scale bar: 1"
    bar = 1.0 / ps
    x0, y0 = npix * 0.06, npix * 0.06
    ax.plot([x0, x0 + bar], [y0, y0], color="white", lw=1.6)
    ax.text(x0 + bar / 2, y0 + npix * 0.02, '1"', color="white",
            ha="center", fontsize=8)
    ax.text(0.96, 0.95, str(gid), color="white", ha="right", va="top",
            transform=ax.transAxes, fontsize=9)
    ax.set_xlim(-0.5, npix - 0.5)
    ax.set_ylim(-0.5, npix - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#666666")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    kw = {"dpi": dpi}
    if str(out_path).endswith(".webp"):
        kw["pil_kwargs"] = {"quality": 88, "method": 4}
    fig.savefig(out_path, **kw)
    plt.close(fig)


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def load_slits(path):
    slits = {}
    if not path or not os.path.exists(os.path.expanduser(path)):
        return slits
    with open(os.path.expanduser(path), newline="") as f:
        for row in csv.DictReader(f):
            gid = int(float(row["id"]))
            slits.setdefault(gid, []).append({
                "ra": float(row["ra"]), "dec": float(row["dec"]),
                "pa_deg": float(row.get("pa_deg", 0) or 0),
                "width_arcsec": float(row.get("width_arcsec", 0.20) or 0.20),
                "height_arcsec": float(row.get("height_arcsec", 0.46) or 0.46),
            })
    return slits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ccfg = cfg.get("cutouts")
    if not ccfg:
        sys.exit("no `cutouts:` section in config - nothing to do")

    out_root = Path(os.path.expanduser(cfg.get("output", "../site")))
    if not out_root.is_absolute():
        out_root = (Path(args.config).parent / out_root).resolve()
    out_dir = out_root / "assets" / "cutouts"
    out_dir.mkdir(parents=True, exist_ok=True)

    size = float(ccfg.get("size_arcsec", 3.0))
    kron_units = ccfg.get("kron_units", "pixel")
    kron_scale = float(ccfg.get("kron_scale", 1.0))
    slits = load_slits(ccfg.get("slits_csv"))

    # master catalogue
    tbl = read_fits_table(os.path.expanduser(cfg["catalog_fits"]))
    idc = cfg.get("catalog_id_column", "ID")
    ids_all = np.asarray(tbl[idc], float).astype(np.int64)

    # figure out which galaxies we need: those present in the site data
    wanted = set(args.ids) if args.ids else None
    if wanted is None:
        import json
        gi = out_root / "data" / "galaxy_index.json"
        if gi.exists():
            with open(gi) as f:
                wanted = {int(k) for k in json.load(f)}
        else:
            wanted = set(ids_all.tolist())

    # optional backend: pre-cut per-galaxy multi-band FITS stamps, one folder
    # per galaxy named by (zero-padded) NIRSpec ID - e.g. the create_cutouts.py
    # output of F. D'Eugenio: <stamps_dir>/000619/000619_F200W_cutout.fits
    stamps_dir = ccfg.get("stamps_dir")
    stamps_dir = Path(os.path.expanduser(stamps_dir)) if stamps_dir else None
    stamps_bands = ccfg.get("stamps_bands",
                            {"r": "F444W", "g": "F200W", "b": "F090W"})
    ns_all = (np.asarray(tbl["NIRSpec_ID"], float)
              if "NIRSpec_ID" in tbl else None)

    def stamp_images(i, gid):
        """Return {r/g/b: MosaicImage} from a stamp folder, or None."""
        if stamps_dir is None:
            return None
        cand_ids = []
        if ns_all is not None and np.isfinite(ns_all[i]):
            cand_ids.append(int(ns_all[i]))
        cand_ids.append(gid)
        folder = None
        for cid in cand_ids:
            for name in (f"{cid:06d}", str(cid)):
                if (stamps_dir / name).is_dir():
                    folder = stamps_dir / name
                    break
            if folder:
                break
        if folder is None:
            return None
        imgs = {}
        for c, band in stamps_bands.items():
            hit = None
            for stem in (folder.name, str(int(folder.name))):
                p = folder / f"{stem}_{band}_cutout.fits"
                if p.exists():
                    hit = p
                    break
            if hit is None:  # any file for this band
                g = sorted(folder.glob(f"*_{band}_*.fits"))
                hit = g[0] if g else None
            if hit is None:
                return None  # missing band: let other backends try
            imgs[c] = MosaicImage(str(hit))
        return imgs

    # open mosaics lazily per field
    fields = ccfg.get("fields", [])
    opened = {}

    def field_for(tier):
        t = (tier or "").lower()
        for fld in fields:
            if fld.get("tier_match", "") .lower() in t:
                return fld
        return fields[0] if fields else None

    n_done = n_skip = 0
    for i, gid in enumerate(ids_all):
        gid = int(gid)
        if gid not in wanted:
            continue
        out = out_dir / f"{gid}.webp"
        if out.exists() and not args.force:
            n_skip += 1
            continue
        ra = float(tbl["RA_1"][i]) if "RA_1" in tbl else None
        dec = float(tbl["DEC_1"][i]) if "DEC_1" in tbl else None
        tier = str(tbl["TIER"][i]) if "TIER" in tbl else ""
        if ra is None or not np.isfinite(ra):
            continue
        imgs = stamp_images(i, gid)
        key = "stamps"
        fld = None
        if imgs is None:
            fld = field_for(tier)
            if fld is None:
                if stamps_dir is not None:
                    continue  # stamps-only config: skip galaxies without stamps
                sys.exit("cutouts.fields is empty - add mosaic paths to config")
            key = fld["name"]
        hips_ps = float(ccfg.get("hips_pixel_scale", 0.03))
        cache = Path(__file__).parent / ".hips_cache"
        if imgs is not None:
            pass  # stamps backend already provided the images
        elif any(str(fld.get(c, "")).startswith("hips:") for c in ("r", "g", "b")):
            # per-galaxy fetch from hips2fits; sources may mix hips: and files
            imgs = {}
            for c in ("r", "g", "b"):
                src = fld.get(c)
                if not src:
                    continue
                if str(src).startswith("hips:"):
                    try:
                        imgs[c] = fetch_hips_image(str(src)[5:], ra, dec,
                                                   size * 1.2, hips_ps, cache)
                    except Exception as e:  # network / service failure
                        print(f"  {gid}: hips2fits failed for {c} ({e}), skipped")
                        imgs = {}
                        break
                else:
                    if key not in opened:
                        opened[key] = {}
                    if c not in opened[key]:
                        opened[key][c] = MosaicImage(os.path.expanduser(str(src)))
                    imgs[c] = opened[key][c]
            if not imgs:
                continue
        else:
            if key not in opened:
                try:
                    opened[key] = {c: MosaicImage(os.path.expanduser(str(fld[c])))
                                   for c in ("r", "g", "b") if fld.get(c)}
                except (FileNotFoundError, OSError) as e:
                    opened[key] = {}
                    print(f"  [warn] field '{key}' mosaics unavailable ({e}); "
                          "galaxies without stamps will be skipped")
            imgs = opened[key]
            if not imgs:
                continue
        cuts = {}
        ok = True
        for c, mi in imgs.items():
            res = mi.cutout(ra, dec, size)
            if res is None:
                ok = False
                break
            cuts[c] = res
        if not ok or not cuts:
            print(f"  {gid}: outside mosaic '{key}', skipped")
            continue
        order = [c for c in ("r", "g", "b") if c in cuts]
        arrs = [cuts[c][0] for c in order]
        # match shapes (different pixel scales between SW/LW would need
        # reprojection; assume same-scale mosaics as fitsmap uses)
        m = min(a.shape[0] for a in arrs)
        arrs = [a[:m, :m] for a in arrs]
        if len(arrs) == 1:
            arrs = arrs * 3
        elif len(arrs) == 2:
            arrs = [arrs[0], arrs[1], arrs[1]]
        rgb = asinh_rgb(arrs)
        _, cx, cy, ps = cuts[order[0]]

        kron = None
        if all(k in tbl for k in ("A_KRON", "B_KRON", "THETA_KRON")):
            a = float(tbl["A_KRON"][i])
            b = float(tbl["B_KRON"][i])
            th = float(tbl["THETA_KRON"][i])
            if np.isfinite(a) and a > 0:
                if kron_units == "pixel":
                    # catalogue pixels; convert with the catalogue's native
                    # pixel scale (kron_pixel_scale, default: mosaic scale)
                    kps = float(ccfg.get("kron_pixel_scale", ps))
                    a, b = a * kps, b * kps
                a *= kron_scale
                b *= kron_scale
                kron = (a, b, th)

        gal_slits = []
        for s in slits.get(gid, []):
            sx, sy = imgs[order[0]].sky2pix(s["ra"], s["dec"])
            # convert to cutout frame
            x_gal, y_gal = imgs[order[0]].sky2pix(ra, dec)
            gal_slits.append({**s, "_x": cx + (sx - x_gal), "_y": cy + (sy - y_gal)})

        render_cutout(rgb, cx, cy, ps, out, gid, kron=kron,
                      slits=gal_slits, size_arcsec=size)
        n_done += 1
        print(f"  {gid}: cutout written ({tier}, field {key})")

    print(f"done: {n_done} written, {n_skip} already existed -> {out_dir}")


if __name__ == "__main__":
    main()
