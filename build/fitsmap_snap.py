#!/usr/bin/env python3
"""Per-galaxy snapshots of a hosted fitsmap with overlay layers enabled.

fitsmap has no cutout API, so this drives a real browser (Playwright) to the
fitsmap URL centred on each galaxy, switches on the requested overlay layers
(e.g. the NIRCam DR5 photometry markers and the NIRSpec slit outlines), and
saves a cropped screenshot to  site/assets/fitsmap/<id>.webp . The site build
links any existing snapshot on the galaxy page automatically (re-run
build_site.py --no-figures after snapping).

Setup (once):
    pip install playwright pillow
    playwright install chromium

Run:
    python fitsmap_snap.py --config config.yaml [--ids 2150 490] [--force]

Config (config.yaml):
    fitsmap_url: "https://jades.idies.jhu.edu/?ra={ra}&dec={dec}&zoom=12"
    fitsmap_snap:
      layers: ["NIRCam DR5 Photometry", "NIRSpec Slit Overlays"]
      crop_px: 560          # square crop around the map centre
      zoom: 12              # overrides the zoom in fitsmap_url if set
      settle_ms: 3500       # wait for tiles/overlays to load

Layer matching is case-insensitive substring matching against the labels in
fitsmap's leaflet layer control; if a requested layer is not found the
available labels are printed so you can adjust the config.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fits_lite import read_fits_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--headed", action="store_true", help="show the browser")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    scfg = cfg.get("fitsmap_snap") or {}
    url_tpl = cfg.get("fitsmap_url")
    if not url_tpl:
        sys.exit("fitsmap_url missing from config")
    zoom = scfg.get("zoom")
    if zoom is not None:
        url_tpl = re.sub(r"zoom=\d+", f"zoom={int(zoom)}", url_tpl)
    layers = scfg.get("layers", [])
    crop = int(scfg.get("crop_px", 560))
    settle = int(scfg.get("settle_ms", 3500))

    out_root = Path(os.path.expanduser(cfg.get("output", "../site")))
    if not out_root.is_absolute():
        out_root = (Path(args.config).parent / out_root).resolve()
    out_dir = out_root / "assets" / "fitsmap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # galaxies = union of all built runs (fall back to --ids)
    gi_path = out_root / "data" / "galaxy_index.json"
    if args.ids:
        wanted = list(args.ids)
    elif gi_path.exists():
        with open(gi_path) as f:
            wanted = sorted(int(k) for k in json.load(f))
    else:
        sys.exit("no --ids given and no data/galaxy_index.json found - build the site first")

    tbl = read_fits_table(os.path.expanduser(cfg["catalog_fits"]))
    idc = cfg.get("catalog_id_column", "ID")
    import numpy as np
    ids_all = np.asarray(tbl[idc], float).astype(np.int64)
    pos = {int(g): (float(tbl["RA_1"][i]), float(tbl["DEC_1"][i]))
           for i, g in enumerate(ids_all)}

    from playwright.sync_api import sync_playwright
    from PIL import Image

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": max(crop + 500, 1100),
                                          "height": max(crop + 260, 820)})
        warned = False
        n_done = n_skip = 0
        for gid in wanted:
            out = out_dir / f"{gid}.webp"
            if out.exists() and not args.force:
                n_skip += 1
                continue
            if gid not in pos:
                print(f"  {gid}: not in master catalogue, skipped")
                continue
            ra, dec = pos[gid]
            page.goto(url_tpl.format(ra=ra, dec=dec), wait_until="load")
            page.wait_for_timeout(settle)

            # open the leaflet layer control and enable requested overlays
            try:
                ctrl = page.locator(".leaflet-control-layers")
                if ctrl.count():
                    ctrl.first.hover()
                    page.wait_for_timeout(300)
                    labels = page.locator(".leaflet-control-layers label")
                    texts = [labels.nth(i).inner_text().strip()
                             for i in range(labels.count())]
                    for want in layers:
                        hit = next((i for i, t in enumerate(texts)
                                    if want.lower() in t.lower()), None)
                        if hit is None:
                            if not warned:
                                print(f"  [warn] layer '{want}' not found; "
                                      f"available: {texts}")
                                warned = True
                            continue
                        box = labels.nth(hit).locator("input")
                        if not box.is_checked():
                            box.click()
                    # move the mouse off the control so it collapses
                    page.mouse.move(10, 10)
                    page.wait_for_timeout(max(800, settle // 3))
            except Exception as e:
                if not warned:
                    print(f"  [warn] layer control interaction failed: {e}")
                    warned = True

            vp = page.viewport_size
            clip = {"x": (vp["width"] - crop) / 2, "y": (vp["height"] - crop) / 2,
                    "width": crop, "height": crop}
            tmp = str(out.with_suffix(".png"))
            page.screenshot(path=tmp, clip=clip)
            Image.open(tmp).convert("RGB").save(out, quality=86, method=4)
            os.remove(tmp)
            n_done += 1
            print(f"  {gid}: snapshot saved")
        browser.close()
    print(f"done: {n_done} snapshots, {n_skip} already existed -> {out_dir}")
    print("re-run build_site.py --no-figures so galaxy pages pick them up")


if __name__ == "__main__":
    main()
