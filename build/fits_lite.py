"""Minimal FITS binary-table reader (numpy only).

Used as a fallback when astropy is not installed. Supports the common
TFORM codes (L, B, I, J, K, E, D, A, and fixed-length arrays thereof)
found in ordinary catalogue bintables. If astropy is available, prefer
``read_fits_table`` which delegates to it.
"""
from __future__ import annotations

import numpy as np

BLOCK = 2880

_TFORM2DTYPE = {
    "L": "i1",   # logical (stored as 'T'/'F' bytes; converted after read)
    "B": "u1",
    "I": ">i2",
    "J": ">i4",
    "K": ">i8",
    "E": ">f4",
    "D": ">f8",
}


def _read_header(f):
    cards = {}
    raw_end = False
    while not raw_end:
        block = f.read(BLOCK)
        if len(block) < BLOCK:
            raise IOError("Truncated FITS header")
        for i in range(0, BLOCK, 80):
            card = block[i:i + 80].decode("ascii", "replace")
            key = card[:8].strip()
            if key == "END":
                raw_end = True
                break
            if not key or key in ("COMMENT", "HISTORY"):
                continue
            if card[8:10] != "= ":
                continue
            raw = card[10:]
            if raw.lstrip().startswith("'"):
                # quoted string: find the closing quote ('' escapes a quote),
                # so a '/' inside the value is not mistaken for a comment
                s = raw.lstrip()
                out, j = [], 1
                while j < len(s):
                    if s[j] == "'":
                        if j + 1 < len(s) and s[j + 1] == "'":
                            out.append("'")
                            j += 2
                            continue
                        break
                    out.append(s[j])
                    j += 1
                cards[key] = "".join(out).rstrip()
                continue
            val = raw.split("/")[0].strip()
            if val in ("T", "F"):
                val = (val == "T")
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            cards[key] = val
    return cards


def _skip_data(f, cards):
    bitpix = abs(int(cards.get("BITPIX", 8)))
    naxis = int(cards.get("NAXIS", 0))
    n = 1
    for i in range(1, naxis + 1):
        n *= int(cards.get(f"NAXIS{i}", 0))
    nbytes = (bitpix // 8) * n * int(cards.get("GCOUNT", 1)) + int(cards.get("PCOUNT", 0))
    if naxis == 0:
        nbytes = 0
    f.seek((nbytes + BLOCK - 1) // BLOCK * BLOCK, 1)


def read_bintable_numpy(path, hdu=1):
    """Read a FITS binary table into a dict of numpy arrays keyed by column name."""
    with open(path, "rb") as f:
        cards = _read_header(f)          # primary
        _skip_data(f, cards)
        current = 0
        while True:
            cards = _read_header(f)
            current += 1
            if current == hdu:
                break
            _skip_data(f, cards)
        if cards.get("XTENSION", "").strip() != "BINTABLE":
            raise ValueError(f"HDU {hdu} is not a BINTABLE")
        nrows = int(cards["NAXIS2"])
        tfields = int(cards["TFIELDS"])
        names, fmts = [], []
        for i in range(1, tfields + 1):
            name = str(cards.get(f"TTYPE{i}", f"col{i}")).strip()
            tform = str(cards[f"TFORM{i}"]).strip()
            rep = ""
            j = 0
            while j < len(tform) and tform[j].isdigit():
                rep += tform[j]
                j += 1
            code = tform[j]
            rep = int(rep) if rep else 1
            if code == "A":
                dt = f"S{rep}"
                shape = ()
            else:
                dt = _TFORM2DTYPE[code]
                shape = () if rep == 1 else (rep,)
            names.append(name)
            fmts.append((dt, shape))
        dtype = np.dtype([(n, d, s) for n, (d, s) in zip(names, fmts)])
        if dtype.itemsize != int(cards["NAXIS1"]):
            raise ValueError(
                f"Row size mismatch: dtype {dtype.itemsize} vs NAXIS1 {cards['NAXIS1']} "
                "(unsupported TFORM code present?)"
            )
        data = np.frombuffer(f.read(dtype.itemsize * nrows), dtype=dtype, count=nrows)
    out = {}
    for n, (d, s) in zip(names, fmts):
        col = data[n]
        if d == "i1":  # logical
            col = (col == ord("T"))
        elif d.startswith("S"):
            col = np.char.strip(col.astype(str))
        elif d.startswith(">"):
            col = col.astype(d[1:])  # native byte order
        out[n] = col
    return out


_BITPIX2DTYPE = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}


def read_fits_image(path, hdu=0):
    """Read a 2-D FITS image HDU as a numpy memmap plus its header dict.

    Only what a cutout task needs: primary or IMAGE extension, BITPIX in
    {8,16,32,64,-32,-64}, no scaling beyond BSCALE/BZERO.
    """
    with open(path, "rb") as f:
        current = 0
        while True:
            hdr_start = f.tell()
            cards = _read_header(f)
            data_start = f.tell()
            if current == hdu:
                break
            _skip_data(f, cards)
            current += 1
    naxis = int(cards.get("NAXIS", 0))
    if naxis < 2:
        raise ValueError(f"HDU {hdu} of {path} is not a 2-D image (NAXIS={naxis})")
    nx, ny = int(cards["NAXIS1"]), int(cards["NAXIS2"])
    dt = np.dtype(_BITPIX2DTYPE[int(cards["BITPIX"])])
    shape = (ny, nx) if naxis == 2 else tuple(
        int(cards[f"NAXIS{i}"]) for i in range(naxis, 0, -1))
    data = np.memmap(path, mode="r", dtype=dt, offset=data_start, shape=shape)
    if naxis > 2:  # take first plane of a cube
        data = data[(0,) * (naxis - 2)]
    return data, cards


class TanWCS:
    """Minimal gnomonic (TAN) WCS: sky <-> pixel, from a FITS header dict.

    Supports CD matrix, or PC matrix + CDELT, or plain CDELT.
    Pixel convention follows FITS (1-based CRPIX); methods use 0-based pixels.
    """

    def __init__(self, hdr):
        self.crval = np.array([float(hdr["CRVAL1"]), float(hdr["CRVAL2"])])
        self.crpix = np.array([float(hdr["CRPIX1"]), float(hdr["CRPIX2"])])
        if "CD1_1" in hdr:
            self.cd = np.array([[float(hdr.get("CD1_1", 0)), float(hdr.get("CD1_2", 0))],
                                [float(hdr.get("CD2_1", 0)), float(hdr.get("CD2_2", 0))]])
        else:
            cdelt = np.array([float(hdr.get("CDELT1", 1)), float(hdr.get("CDELT2", 1))])
            pc = np.array([[float(hdr.get("PC1_1", 1)), float(hdr.get("PC1_2", 0))],
                           [float(hdr.get("PC2_1", 0)), float(hdr.get("PC2_2", 1))]])
            self.cd = pc * cdelt[:, None]
        self.cdinv = np.linalg.inv(self.cd)

    @property
    def pixel_scale_arcsec(self):
        return float(np.sqrt(abs(np.linalg.det(self.cd))) * 3600.0)

    def sky2pix(self, ra, dec):
        d2r = np.pi / 180.0
        ra0, dec0 = self.crval * d2r
        ra, dec = np.asarray(ra) * d2r, np.asarray(dec) * d2r
        cosc = (np.sin(dec0) * np.sin(dec)
                + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0))
        xi = np.cos(dec) * np.sin(ra - ra0) / cosc / d2r
        eta = (np.cos(dec0) * np.sin(dec)
               - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cosc / d2r
        xy = self.cdinv @ np.vstack([xi, eta])
        return xy[0] + self.crpix[0] - 1, xy[1] + self.crpix[1] - 1

    def pix2sky(self, x, y):
        d2r = np.pi / 180.0
        v = self.cd @ np.vstack([np.asarray(x) - self.crpix[0] + 1,
                                 np.asarray(y) - self.crpix[1] + 1])
        xi, eta = v[0] * d2r, v[1] * d2r
        ra0, dec0 = self.crval * d2r
        den = np.cos(dec0) - eta * np.sin(dec0)
        ra = ra0 + np.arctan2(xi, den)
        dec = np.arctan((np.sin(dec0) + eta * np.cos(dec0))
                        / np.sqrt(xi ** 2 + den ** 2))
        return ra / d2r, dec / d2r


def read_fits_table(path, hdu=1):
    """Read a FITS table, preferring astropy when available."""
    try:
        from astropy.io import fits  # type: ignore
        with fits.open(path, memmap=False) as h:
            rec = h[hdu].data
            out = {}
            for n in rec.dtype.names:
                col = np.asarray(rec[n])
                if col.dtype.kind == "S":      # FITS strings arrive as bytes:
                    col = np.char.decode(col, "ascii", "replace")
                if col.dtype.kind == "U":      # strip FITS blank padding
                    col = np.char.rstrip(col)
                out[n] = col
            return out
    except ImportError:
        return read_bintable_numpy(path, hdu=hdu)
