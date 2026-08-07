# JADES fit explorer

An interactive, static catalogue website for Prospector SED-fitting results of
the JADES NIRCam + NIRSpec sample. Sortable per-run tables, and a detail page
per galaxy per fit with the posterior corner plot, the SED (observed vs
posterior-predicted photometry with the model-spectrum band), the emission-line
predicted/observed ratios, the per-parameter information gain
KL(posterior ‖ prior), catalogue metadata, and a link into fitsmap at the
galaxy's coordinates.

Everything is static files - no server-side code - so the site can be hosted on
GitHub Pages (or any web server) as-is.

```
jades-fit-explorer/
├── build/
│   ├── build_site.py    # main build script (scans results dirs, writes site data)
│   ├── figures.py       # corner / SED / line-ratio / KL figures from posterior npz
│   ├── priors.py        # analytic priors for the KL computation
│   ├── fits_lite.py     # numpy-only FITS table reader (fallback when astropy absent)
│   └── config.yaml      # <-- the one file you edit
├── site/
│   ├── index.html       # the whole web app (vanilla JS, no dependencies)
│   ├── data/            # generated JSON (runs.json, per-run catalog + per-galaxy files)
│   └── assets/          # generated figures (WebP)
└── .github/workflows/deploy.yml   # GitHub Pages deployment
```

## Requirements

Python ≥ 3.10 with `numpy`, `matplotlib`, `pyyaml`, `pillow`. Optional:
`astropy` (faster/robuster FITS reading; without it a built-in numpy reader is
used). No `corner` package needed - the corner plot is self-contained.

## Building / updating the site

```bash
python build/build_site.py --config build/config.yaml --jobs 8
```

The build is **incremental**: a galaxy's figures are only regenerated when its
`posterior_*.npz` is newer than the existing figures (`--force` overrides;
`--no-figures` rebuilds only the JSON tables, which takes seconds).

Preview locally (browsers block `fetch()` on `file://`, so serve it):

```bash
cd site && python -m http.server 8000    # -> http://localhost:8000
```

### Adding a new fit run

1. Append an entry under `runs:` in `build/config.yaml`:

```yaml
  - name: my_new_model_ns
    label: "my new model"
    description: "Nested-sampling fits with ..."
    results_dir: ~/path/to/results_my_new_model
    pattern: 'posterior_jades_(?P<id>\d+)_my_new_model_ns\.npz$'
    summary_csv: ~/path/to/summary.csv        # optional
```

2. Re-run the build (add `--runs my_new_model_ns` to build only that run).
3. Commit and push - the new run appears as a tab, and each galaxy's page gets
   a badge linking between all runs that contain it.

`summary_csv` is optional. When present, chi^2 statistics and derived
quantities (f_esc, M_UV, xi_ion, ...) are read from it; otherwise everything
shown is computed from the npz alone.

### Priors for the KL figure

The per-parameter information gain needs the analytic prior. The
`logsfr_ratios[i]` Student-t prior is read from each npz automatically
(`sfh_prior_df/loc/scale`). For the remaining parameters, fill in the
`priors:` block in `config.yaml` to match your Prospector model. Parameters
without a spec fall back to a uniform prior over the sampled range and are
drawn with lighter bars in the figure (a lower bound on the true information
gain).

## RGB cutouts with aperture + slit overlays

`build/cutouts.py` extracts a cutout per galaxy from the same NIRCam mosaics
your fitsmap is built from, composes an asinh-stretched RGB image, and draws
the Kron aperture (A_KRON/B_KRON/THETA_KRON from the master catalogue, 30 mas
pixels) plus, optionally, MSA shutter rectangles from a CSV
(`id, ra, dec, pa_deg, width_arcsec, height_arcsec`; one row per shutter).

```bash
# 1. fill in cutouts.fields in config.yaml (paths to your mosaic FITS files)
python build/cutouts.py --config build/config.yaml
# 2. then (re)build the site - existing cutouts are linked automatically
python build/build_site.py --config build/config.yaml --no-figures
```

Cutouts land in `site/assets/cutouts/<id>.webp` (~25 kB each, shared across
runs) and appear on the galaxy page next to the fitsmap link. The script was
validated end-to-end against a synthetic TAN-WCS mosaic; point it at the real
GOODS-S/GOODS-N mosaics on the machine where they live.

No local mosaics? Set a channel to `"hips:<HiPS-ID>"` instead of a file path
and the cutout is fetched per galaxy from the CDS hips2fits service (cached
in `build/.hips_cache`); browse available JWST/JADES HiPS surveys at
https://aladin.cds.unistra.fr/hips/list. Local files and `hips:` channels can
be mixed freely; overlays work identically for both.

## Catalogue-wide figures

Each run's table links to an interactive **Catalogue figures** page
(`#/figures/<run>`) with five panels: predicted-vs-observed flux for
[O III] 5007, Hα and Hβ (log-log, 1:1 line, posterior + observational error
bars), stellar mass vs SFR₁₀ with the Simmonds et al. (2025, MNRAS 544, 4551)
star-forming main sequence drawn at z = 3/5/7/9 (solid inside the fitted range
9.0 ≤ log M★ ≤ 10.3, dotted extrapolation), and the mass-metallicity relation
(12+log(O/H) = 8.69 + log Z_gas/Z☉) with an external MZR release drawn as grey
background points (`overlays.mzr_fits` in config.yaml; expects columns
`12+log(O/H)`, `log(Mstar)`, `z_Spec`).

Each flux-recovery panel has a pull subpanel underneath ((pred−obs)/σ_obs vs
observed flux, ±1σ band, ±3σ dashed). Additional panels: SFR₁₀ from the fit vs
SFR from the observed (not dust-corrected) Hα flux via Kennicutt & Evans
(2012) (log SFR = log L(Hα) − 41.27, flat ΛCDM 70/0.3); Tₑ-based metallicity
from the MZR release vs the fitted gas-phase metallicity (galaxies matched on
`Unique_ID`); and two sample-wide pull-distribution panels — one per emission
line, one per filter (box = 16–84 per cent, bar = median, individual points
drawn when ≤60 galaxies pass the filters) — to expose systematic tendencies
across the whole sample.

All points are coloured by fitted redshift and every panel obeys three reader
controls: z_spec-flag toggles (A/B/C), a χ²_ν upper limit, and a fitted-redshift
range. Hovering shows the galaxy id and values; clicking opens its page.

### Data requirements

The pipeline reads **only the `posterior_*.npz` files** - the `.h5` files are
never opened, so they do not need to be transferred from the cluster. The
optional per-run `summary_csv` adds χ² statistics and derived quantities
(f_esc, M_UV, ξ_ion); the master catalogue FITS provides coordinates, z_spec
and flags.

## fitsmap snapshots per galaxy

`build/fitsmap_snap.py` drives a browser (Playwright) to the hosted fitsmap
centred on each galaxy (zoom from `fitsmap_snap.zoom`), switches on the layers
named in `fitsmap_snap.layers` (e.g. the NIRCam DR5 photometry markers and the
NIRSpec slit outlines - matched case-insensitively against the layer-control
labels, with the available labels printed if a name doesn't match), and saves
a square crop to `site/assets/fitsmap/<id>.webp`. Needs
`pip install playwright && playwright install chromium` and network access to
the fitsmap host; afterwards re-run `build_site.py --no-figures` so the galaxy
pages link the snapshots.

## Hosting on GitHub Pages

1. Create a repository and push this directory to it.
2. In the repository settings: **Settings → Pages → Source: GitHub Actions**.
3. Push to `main` - the included workflow (`.github/workflows/deploy.yml`)
   publishes `site/` automatically. The site appears at
   `https://<user>.github.io/<repo>/`.

### Sizes, and scaling to 5,000 galaxies x 4 models

Each galaxy-fit costs roughly 420 kB (corner ≈ 350 kB at `corner_dpi: 110`,
the other three figures ≈ 60 kB, JSON ≈ 10 kB). GitHub Pages' soft limit is
~1 GB per site, i.e. ~2,000-2,500 galaxy-fits in a single repository. A full
5,000 x 4 catalogue (~20,000 galaxy-fits, ~8 GB of figures) therefore needs
the figures split off. The build supports this natively via `config.yaml`:

```yaml
figures:
  corner_dpi: 80              # ~halves the corner-plot size
assets_output: ~/jades-assets # write figures outside site/
asset_shards: 8               # figures split by galaxy id modulo 8
asset_base_url: "https://<user>.github.io/jades-assets-{shard}"
```

Then create 8 repositories `jades-assets-0` ... `jades-assets-7` (each with
Pages enabled, publishing the repository root), push `~/jades-assets/shard<k>/*`
into repository `k`, and push the main repo as usual. The JSON already
contains the absolute figure URLs, so the site needs no changes. The main
repo keeps only JSON + cutouts: ~20k x 10 kB + 5k x 25 kB ≈ 325 MB - well
within limits. At `corner_dpi: 80` each shard holds ~2,500 galaxy-fits
≈ 0.6 GB.

## Notes on the data model

* Photometry is stored/displayed in nJy (converted from maggies).
* The model-spectrum unit (maggies vs erg s^-1 cm^-2 Hz^-1) is auto-detected
  per file by comparison with the observed photometry.
* Galaxies fitted without emission lines (photometry-only) are handled
  automatically: the line figure/table is simply omitted for them.
* The master catalogue (`catalog_fits`) is matched on the `ID` column;
  coordinates, tier, z_spec, and the columns listed in `catalog_columns` are
  merged into each galaxy page.
