"""Analytic prior distributions used to compute per-parameter KL divergence.

The build config (config.yaml, key ``priors``) can specify a prior per
parameter, e.g.::

    priors:
      logmass:            {dist: uniform, lo: 6.0, hi: 12.0}
      gas_logu:           {dist: uniform, lo: -4.0, hi: -1.0}
      diffuse_dust_index: {dist: normal, mean: 0.0, sigma: 0.5, lo: -1.0, hi: 0.4}

Supported ``dist`` values: uniform, normal (optionally truncated with
lo/hi), studentt (df, loc, scale, optionally truncated).

Parameters named ``logsfr_ratios[i]`` automatically use the Student-t
prior recorded inside each posterior npz (sfh_prior_df/loc/scale) unless
overridden. Anything without a spec falls back to a uniform prior over
the sampled range - the KL figure marks those bars in a lighter colour.
"""
from __future__ import annotations

import math

import numpy as np


class UniformPrior:
    def __init__(self, lo, hi):
        self.lo, self.hi = float(lo), float(hi)

    def support(self):
        return self.lo, self.hi

    def pdf(self, x):
        if self.lo <= x <= self.hi and self.hi > self.lo:
            return 1.0 / (self.hi - self.lo)
        return 0.0


class NormalPrior:
    def __init__(self, mean, sigma, lo=-np.inf, hi=np.inf):
        self.mean, self.sigma = float(mean), float(sigma)
        self.lo, self.hi = float(lo), float(hi)
        # truncation renormalisation
        a = 0.5 * (1 + math.erf((self.hi - self.mean) / (self.sigma * math.sqrt(2)))) \
            if np.isfinite(self.hi) else 1.0
        b = 0.5 * (1 + math.erf((self.lo - self.mean) / (self.sigma * math.sqrt(2)))) \
            if np.isfinite(self.lo) else 0.0
        self._norm = max(a - b, 1e-300)

    def support(self):
        lo = self.lo if np.isfinite(self.lo) else self.mean - 8 * self.sigma
        hi = self.hi if np.isfinite(self.hi) else self.mean + 8 * self.sigma
        return lo, hi

    def pdf(self, x):
        if not (self.lo <= x <= self.hi):
            return 0.0
        z = (x - self.mean) / self.sigma
        return math.exp(-0.5 * z * z) / (self.sigma * math.sqrt(2 * math.pi) * self._norm)


class StudentTPrior:
    def __init__(self, df, loc, scale, lo=-np.inf, hi=np.inf):
        self.df, self.loc, self.scale = float(df), float(loc), float(scale)
        self.lo, self.hi = float(lo), float(hi)
        self._c = (math.gamma((self.df + 1) / 2)
                   / (math.sqrt(self.df * math.pi) * math.gamma(self.df / 2) * self.scale))
        # numeric renormalisation if truncated
        if np.isfinite(self.lo) or np.isfinite(self.hi):
            lo, hi = self.support()
            xs = np.linspace(lo, hi, 4001)
            ys = np.array([self._pdf_raw(x) for x in xs])
            self._norm = max(float(np.trapezoid(ys, xs)), 1e-300)
        else:
            self._norm = 1.0

    def _pdf_raw(self, x):
        z = (x - self.loc) / self.scale
        return self._c * (1 + z * z / self.df) ** (-(self.df + 1) / 2)

    def support(self):
        lo = self.lo if np.isfinite(self.lo) else self.loc - 60 * self.scale
        hi = self.hi if np.isfinite(self.hi) else self.loc + 60 * self.scale
        return lo, hi

    def pdf(self, x):
        if not (self.lo <= x <= self.hi):
            return 0.0
        return self._pdf_raw(x) / self._norm


def student_t_prior_from_extra(extra):
    """Build the SFH log-ratio prior from metadata stored in the npz."""
    try:
        df = float(extra["sfh_prior_df"])
        loc = float(extra["sfh_prior_loc"])
        scale = float(extra["sfh_prior_scale"])
    except (KeyError, TypeError, ValueError):
        return None
    return StudentTPrior(df, loc, scale)


def priors_from_config(spec: dict | None):
    """Build {param_name: prior} from the config.yaml ``priors`` mapping."""
    out = {}
    for name, s in (spec or {}).items():
        dist = str(s.get("dist", "uniform")).lower()
        if dist == "uniform":
            out[name] = UniformPrior(s["lo"], s["hi"])
        elif dist == "normal":
            out[name] = NormalPrior(s["mean"], s["sigma"],
                                    s.get("lo", -np.inf), s.get("hi", np.inf))
        elif dist in ("studentt", "student_t", "t"):
            out[name] = StudentTPrior(s["df"], s.get("loc", 0.0), s.get("scale", 1.0),
                                      s.get("lo", -np.inf), s.get("hi", np.inf))
        else:
            raise ValueError(f"Unknown prior dist '{dist}' for {name}")
    return out
