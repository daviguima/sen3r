import logging

import outsourcing as out
# import nc_explorer
import utils
import os
import sys


def tif_stats(image=None):
    print('S3-FRBR:MOD3R-mode')

    # input kml / kml(s)
    vector = '/d_drive_data/A1_JM/manacapuru.shp'

    raster = '/d_drive_data/S3/s3jm/S3A_OL_1_EFR____20200213T133119_20200213T133419_20200214T171023_0179_055_024_3060_LN1_O_NT_002_processed.tif'

    from rasterstats import zonal_stats

    stats = zonal_stats(vector, raster)

    print(len(stats))

    print([f['mean'] for f in stats])
    pass


def mass_atmcor(folder, destination=None):

    work_dir = folder
    field_files = os.listdir(work_dir)

    for image in field_files:
        if destination:
            out.ICORBridge.run_iCOR_on_image(folder + '\\' + image, destination)
        else:
            out.ICORBridge.run_iCOR_on_image(folder + '\\' + image)

    pass


if __name__ == '__main__':
    if len(sys.argv) > 2:
        mass_atmcor(sys.argv[1], sys.argv[2])
    else:
        mass_atmcor(sys.argv[1])
