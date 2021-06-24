import os
import sys
import logging
import argparse
# import outsourcing as outsrc
from rasterstats import zonal_stats
from datetime import datetime
import numpy as np
import gdal
from gdalconst import *
from osgeo import osr


class s3r_tif:
    band_id = {1: 'B1-400',
               2: 'B2-412.5',
               3: 'B3-442.5',
               4: 'B4-490',
               5: 'B5-510',
               6: 'B6-560',
               7: 'B7-620',
               8: 'B8-665',
               9: 'B9-673.75',
               10: 'B10-681.25',
               11: 'B11-708.75',
               12: 'B12-753.75',
               13: 'B16-778.75',
               14: 'B17-865',
               15: 'B18-885',
               16: 'B21-1020'}

    # TODO: fix outsorcing.py dependency
    # def tif_stats(self, vector, raster, keyname='mean'):
    #     logging.info(f'Input vector: {vector}')
    #     logging.info(f'Input raster: {raster}')
    #
    #     gdbr = outsrc.GDALBridge()
    #     bcount = gdbr.get_tiff_band_count(raster)
    #     bdic = {}
    #
    #     input_filename = os.path.basename(raster)
    #
    #     # Removing everything from the raster name and leaving only the acquisition date
    #     str_date_from_file = input_filename[16:31]
    #     datetime_date_from_file = datetime.strptime(str_date_from_file, '%Y%m%dT%H%M%S')
    #
    #     bdic.update({'Datetime': str(datetime_date_from_file),
    #                  'Date-String': str_date_from_file})
    #
    #     for n in range(bcount):
    #         stats = zonal_stats(vector, raster, band=n + 1).pop()
    #
    #         if keyname:
    #             bdic.update({self.band_id[n + 1]: stats[keyname]})
    #         else:
    #             bdic.update({self.band_id[n + 1]: stats})
    #
    #     bdic.update({'filename': input_filename})
    #     logging.info(f'TFXP.tif_stats:\n{bdic}')
    #
    #     return bdic

    # TODO: fix outsorcing.py dependency
    # def mass_atmcor(self, folder, destination=None):
    #
    #     work_dir = folder
    #     field_files = os.listdir(work_dir)
    #
    #     for image in field_files:
    #         if destination:
    #             outsrc.ICORBridge.run_iCOR_on_image(folder + '\\' + image, destination)
    #         else:
    #             outsrc.ICORBridge.run_iCOR_on_image(folder + '\\' + image)
    #
    #     pass

    # https://gis.stackexchange.com/questions/57005/python-gdal-write-new-raster-using-projection-from-old
    @staticmethod
    def get_geoinfo(filename):
        # Function to read the original file's projection:
        SourceDS = gdal.Open(filename, GA_ReadOnly)
        NDV = SourceDS.GetRasterBand(1).GetNoDataValue()
        xsize = SourceDS.RasterXSize
        ysize = SourceDS.RasterYSize
        GeoT = SourceDS.GetGeoTransform()
        Projection = osr.SpatialReference()
        Projection.ImportFromWkt(SourceDS.GetProjectionRef())
        DataType = SourceDS.GetRasterBand(1).DataType
        DataType = gdal.GetDataTypeName(DataType)
        return NDV, xsize, ysize, GeoT, Projection, DataType

    @staticmethod
    # Function to write a new file.
    def CreateGeoTiff(Name, Array, driver, NDV,
                      xsize, ysize, GeoT, Projection, DataType):
        if DataType == 'Float32':
            DataType = gdal.GDT_Float32
        NewFileName = Name + '.tif'
        # Set nans to the original No Data Value
        Array[np.isnan(Array)] = NDV
        # Set up the dataset
        DataSet = driver.Create(NewFileName, xsize, ysize, 1, DataType)
        # the '1' is for band 1.
        DataSet.SetGeoTransform(GeoT)
        DataSet.SetProjection(Projection.ExportToWkt())
        # Write the array
        DataSet.GetRasterBand(1).WriteArray(Array)
        DataSet.GetRasterBand(1).SetNoDataValue(NDV)
        return NewFileName


if __name__ == '__main__':

    run = s3r_tif()

    # LOG HOTFIX FOR PyCharm
    # https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # CREATING HANDLER FOR USER INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log', help='Name and path of the log file to be saved.')
    parser.add_argument('-t', '--tif', help='Input .tiff image.')
    parser.add_argument('-s', '--shp', help='Input Shapefile.shp delimiting area.')
    parser.add_argument('-k', '--kml', help='Input KML delimiting area.')
    # boolean flag argument
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Whether or not to print to stdout.')
    args = parser.parse_args()

    # CREATE AND SAVE LOG FILE
    if args.verbose:
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S',
                            handlers=[logging.FileHandler(args.log),
                                      logging.StreamHandler()],
                            level=logging.INFO)
    else:
        logging.basicConfig(filename=args.log,
                            format='%(asctime)s - %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S',
                            level=logging.INFO)

    # # CALL INTERNAL iCOR MODULE ON EVERY IMAGE OF THE GIVEN FOLDER
    # mass_atmcor(sys.argv[1])

    # # TEST IF THE CORRECTED IMAGE SHOULD BE SAVED IN ANOTHER FOLDER OTHER THAN C:\Temp
    # if len(sys.argv) > 2:
    #     # mass_atmcor(sys.argv[1], sys.argv[2])
    # else:
    #     mass_atmcor(sys.argv[1])

    # # GET ZONAL STATISTICS FROM TIFF FILES
    logging.info(sys.argv)
    run.tif_stats(args.shp, args.tif)
