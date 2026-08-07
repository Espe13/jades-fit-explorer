# coding: utf-8
# author: Francesco D'Eugenio
# email: fdeugenio@gmail.com
# complaints: fdeugenio.robot@gmail.com
import glob
import os
import sys
import tqdm

import numpy as np

from astropy import coordinates, units, wcs
from astropy.io import fits 

"""
>>> python3 create_cutouts.py JGN080660 189.2753941 62.2141291 4 FOLDER [path_to_filter_folders]
"""

def create_cutout_hst(ra, dec, image_size_arc, input_filename, output_filename):

    data = fits.open(input_filename)

    img_wcs = wcs.WCS(data[0].header)
    x, y = img_wcs.all_world2pix(((ra, dec),), 0)[0]
    x = int(x+0.5)
    y = int(y+0.5)
    
    pixel_scale_arc = img_wcs.proj_plane_pixel_scales()[0].to('arcsec')
    image_size = int(image_size_arc/pixel_scale_arc)

    assert x>=0 and y>=0, f'Error, {ra}, {dec} not in footprint'

    for d in data:
        if d.data is not None:
            d.data = d.data[y-image_size:y+image_size+1,
                            x-image_size:x+image_size+1]
            d.header['CRPIX1'] -= (x-image_size)
            d.header['CRPIX2'] -= (y-image_size)

    wht_file = input_filename.replace('sci', 'wht')
    if os.path.isfile(wht_file):
        wht_file = fits.open(wht_file)
        wht_file[0].data = wht_file[0].data[
            y-image_size:y+image_size+1,
            x-image_size:x+image_size+1]
        wht_file[0].header['CRPIX1'] -= (x-image_size)
        wht_file[0].header['CRPIX2'] -= (y-image_size)
        wht_file[0].data = 1/np.sqrt(wht_file[0].data)
        data.append(wht_file[0])
    data[0].header['EXTNAME'] = 'SCI'

    data.writeto(output_filename, overwrite=True)

def create_cutout_jwst(ra, dec, image_size_arc, input_filename, output_filename):

    data = fits.open(input_filename)

    img_wcs = wcs.WCS(data['SCI'].header)
    x, y = img_wcs.all_world2pix(((ra, dec),), 0)[0]
    x = int(x+0.5)
    y = int(y+0.5)
    print(x, y)
    
    pixel_scale_arc = img_wcs.proj_plane_pixel_scales()[0].to('arcsec')
    image_size = int(image_size_arc/pixel_scale_arc)

    print(x, y, image_size)
    for d in data:
        if d.data is not None:
            d.data = d.data[y-image_size:y+image_size+1,
                            x-image_size:x+image_size+1]
            d.header['CRPIX1'] -= (x-image_size)
            d.header['CRPIX2'] -= (y-image_size)
               
    data.writeto(output_filename, overwrite=True)


def get_radec(ra, dec):
    try:
        ra = float(ra)
        dec = float(dec)
    except Exception as e:
        coords = coordinates.SkyCoord(
            ra, dec, frame='fk5')
        ra = coords.ra.value
        dec = coords.dec.value
    return ra, dec

def main():
    args = sys.argv[1:]
    target_name = args[0]
    ra, dec = get_radec(args[1], args[2])
    size = float(args[3]) * units.arcsec
    output_dir = args[4]
    root_dir = args[5] if len(args)==6 else 'GOODS-S/images'
    os.makedirs(output_dir, exist_ok=True)
    # 53.1652755, -27.8141397),), 0)[0]

    jwst_bands = ('F070W', 'F090W', 'F115W', 'F150W', 'F182M', 'F200W',
        'F210M', 'F277W', 'F335M', 'F356W', 'F410M', 'F430M', 'F444W',
        'F460M', 'F480M')
    hst_acs_bands = ('F435W', 'F606W', 'F775W', 'F814W', 'F850LP')
    hst_wfc3_bands = ('F105W', 'F125W', 'F140W', 'F160W',)
    #bands = ('F090W', 'F200W', 'F444W')
    bands = jwst_bands + hst_acs_bands + hst_wfc3_bands

    for band in tqdm.tqdm(bands):
        if band in jwst_bands:
            input_filename  = f'JWST/{band}/mosaic_{band}.fits'
        elif band in hst_acs_bands:
            input_filename  = f'HST/HLF/v2.0.1/{band}/hlsp_hlf_hst_acs-30mas_goodss_{band.lower()}_v2.0.1_sci.fits'
        elif band in hst_wfc3_bands:
            input_filename  = f'HST/HLF/v2.0.1/{band}/hlsp_hlf_hst_wfc3-30mas_goodss_{band.lower()}_v2.0.1_culled_sci.fits'
        else:
            raise ValueError(f'{band} does not seem a JWST or HST band')
        input_filename = os.path.join(root_dir, input_filename)
        output_filename = f'{target_name}_{band}_cutout.fits'
        output_filename = os.path.join(output_dir, output_filename)

        if os.path.isfile(input_filename):
            if band in jwst_bands:
                create_cutout_jwst(ra, dec, size, input_filename, output_filename)
            else:
                create_cutout_hst(ra, dec, size, input_filename, output_filename)
        else:
            print(f'Skipping {band} (file {input_filename} not found)')


if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print(e)
        print('python3 create_cutouts.py GS10578 53.1652755 -27.8141397 4 outdir')
        raise e
