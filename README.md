# JADES SED-fitting catalogue explorer

An interactive catalogue of Prospector SED-fitting results for JADES
(JWST Advanced Deep Extragalactic Survey) galaxies with NIRCam photometry and
NIRSpec/MSA spectroscopy, fitted with nested sampling. The site is entirely
static: every number and figure was precomputed from the posterior samples,
so what you see is exactly what the fits contain.

## Using the site

The landing page lists one tab per model run (for example the fesc model and
the extra-young SFH model; some galaxies are fitted by more than one model).
Each tab shows a sortable, searchable table of the fitted galaxies with their
median posterior parameters and 68 per cent credible intervals - click any
column header to sort, and use the search box to jump to an ID, tier, or
coordinate. Clicking a row opens the galaxy's detail page.

Each galaxy page shows, per model: the posterior corner plot for all sampled
parameters; the SED with observed and posterior-predicted photometry and the
model-spectrum band, with a residual (chi) panel; the emission-line
predicted-to-observed ratio figure (line fluxes in erg s^-1 cm^-2); and the
per-parameter information gain KL(posterior || prior), which measures how much
each parameter was constrained by the data rather than the prior. Where
available there is an RGB image cutout (F444W/F200W/F090W) with the Kron
aperture overlaid, made from JWST image stamps. The "open in fitsmap" link
opens the interactive JADES map at the galaxy's position, and "download JSON"
gives you every number on the page (posterior percentiles, line fluxes,
photometry, goodness of fit) in machine-readable form.

Each run also has a Catalogue figures page with sample-wide diagnostics:
predicted vs observed fluxes for [O III] 5007, Halpha, and Hbeta, each with a
pull panel ((pred - obs)/sigma); stellar mass vs SFR_10 against the
star-forming main sequence of Simmonds et al. (2025, MNRAS 544, 4551); SFR_10
vs the dust-corrected Halpha SFR (Balmer decrement, Kennicutt & Evans 2012);
the mass-metallicity relation; a comparison of T_e-based and SED-fit gas-phase
metallicities; and pull distributions per emission line and per filter. All
points are coloured by fitted redshift, and every panel obeys three filters
you control: spectroscopic-redshift quality flag, a chi^2_nu upper limit, and
a redshift range. Hover any point for its ID and values; click it to open the
galaxy's page. Only measurements that actually entered the fit likelihood are
shown anywhere on the site.

## Programmatic access

All data behind the site are plain JSON files you can fetch directly:

```
site/data/runs.json                          # list of model runs and galaxy IDs
site/data/runs/<run>/catalog.json            # one row per galaxy: parameters, line fluxes, pulls
site/data/runs/<run>/galaxies/<id>.json      # everything on a galaxy page
```

Conventions: line fluxes in erg s^-1 cm^-2; SFRs in M_sun yr^-1 averaged over
10/100 Myr; masses as log M*/M_sun; gas-phase metallicity as log Z_gas/Z_sun
(12 + log(O/H) = 8.69 + log Z_gas/Z_sun for solar-scaled abundances); pulls as
(predicted - observed)/sigma_observed. Parameter entries carry the posterior
median and the 16th/84th percentiles.

## Building the site yourself

The build pipeline (Python: numpy, matplotlib, pyyaml, Pillow; astropy
optional) turns Prospector posterior `.npz` files into the static site - see
`build/config.yaml` for the run definitions and `build/build_site.py` to
rebuild. The image cutouts are rendered from per-galaxy multi-band FITS stamps
(courtesy of Francesco D'Eugenio's cutout tooling) or, without local imaging,
from the CDS hips2fits service.

## Acknowledgements

Built on data products of the JADES collaboration and fits made with
Prospector (nested sampling). The interactive map is fitsmap. The SFMS
overlay is from Simmonds et al. (2025), MNRAS 544, 4551 (arXiv:2508.04410);
the Halpha SFR calibration from Kennicutt & Evans (2012); image stamps
courtesy of F. D'Eugenio; T_e-based metallicities from a JADES-internal
release. Please contact the repository owner before using numbers from this
catalogue in a publication - the underlying paper reference will be added
here once available.
