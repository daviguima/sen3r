import os
import sys
import logging
import netCDF4 as nc

from pathlib import Path
from skimage.transform import resize


class NcEngine:
    """
    Provide methods to manipulate NetCDF4 data from Sentinel-3 OLCI products.
    """

    def __init__(self, input_nc_folder=None, product='WFR', initialize=True, external_use=False):
        self.initiated = initialize
        self.t_lat = None
        self.t_lon = None
        self.g_lat = None
        self.g_lon = None
        self.OAA = None
        self.OZA = None
        self.SAA = None
        self.SZA = None

        if not external_use:
            if input_nc_folder:
                self.nc_folder = Path(input_nc_folder)
            else:
                self.nc_folder = input_nc_folder

            self.product = product.lower()
            self.os = os.name
            self.class_label = 'SEN3R:nc4_agent'
            logging.info(f'Declaring class instance from: {self.class_label}')

            if self.nc_folder is None:
                print(f'Input NetCDF file folder not set. Proceed at your own risk.')
            else:
                print('Reading valid NetCDF files inside image folder...')
                self.netcdf_valid_band_list = self.get_valid_band_files(rad_only=False)

            if input_nc_folder and initialize:
                self.initialize_geometries()
            else:
                logging.info('initialize set to False, ignoring image geometries.'
                             ' This can be later done manually by calling initialize_geometries()'
                             ' after properly setting the nc_folder.')

    def initialize_geometries(self):
        """
        Manually load the metadata of the input NetCDF in case it was not automatically done
        during class instace.
        """
        logging.info(f'Product set to {self.product.upper()}.')
        logging.info('Loading image bands into memory, this may take a while...')
        self.initiated = True
        if self.product.lower() == 'wfr':
            geo_coord = nc.Dataset(self.nc_folder / 'geo_coordinates.nc')
            self.g_lat = geo_coord['latitude'][:]
            self.g_lon = geo_coord['longitude'][:]

            # Load and resize tie LON/LAT Bands using the geo_coordinates.nc file dimensions: (4091, 4865)
            tie_geo = nc.Dataset(self.nc_folder / 'tie_geo_coordinates.nc')
            self.t_lat = tie_geo['latitude'][:]
            self.t_lat = resize(self.t_lat, (self.g_lat.shape[0], self.g_lat.shape[1]), anti_aliasing=False)
            self.t_lon = tie_geo['longitude'][:]
            self.t_lon = resize(self.t_lon, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

            # Load and resize Sun Geometry Angle Bands using the geo_coordinates.nc file dimensions: (4091, 4865)
            t_geometries = nc.Dataset(self.nc_folder / 'tie_geometries.nc')
            self.OAA = t_geometries['OAA'][:]
            self.OAA = resize(self.OAA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

            self.OZA = t_geometries['OZA'][:]
            self.OZA = resize(self.OZA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

            self.SAA = t_geometries['SAA'][:]
            self.SAA = resize(self.SAA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

            self.SZA = t_geometries['SZA'][:]
            self.SZA = resize(self.SZA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

        elif self.product.lower() == 'syn':
            dsgeo = nc.Dataset(self.nc_folder / 'geolocation.nc')
            self.g_lat = dsgeo['lat'][:]
            self.g_lon = dsgeo['lon'][:]

        else:
            print(f'Invalid product: {self.product.upper()}.')
            self.initiated = False

    def get_valid_band_files(self, rad_only=True):
        """
        Search inside the .SEN3 image folder for files ended with .nc; If rad_only is True,
        only reflectance bands are returned, otherwise return everything ended with .nc extension.
        """
        if self.nc_folder is None:
            logging.info('Unable to find files. NetCDF image folder is not defined during NcExplorer class instance.')
            sys.exit(1)

        sentinel_images_path = self.nc_folder

        # retrieve all files in folder
        files = os.listdir(sentinel_images_path)

        # extract only NetCDFs from the file list
        nc_files = [f for f in files if f.endswith('.nc')]

        # extract only the radiometric bands from the NetCDF list
        nc_bands = [b for b in nc_files if b.startswith('Oa')]

        logging.info(f'{self.class_label}.get_valid_band_files()\n'
                     f'Sentinel-3 Image folder:\n'
                     f'{sentinel_images_path}\n'
                     f'Total files in folder: {len(files)}\n'
                     f'Total NetCDF files: {len(nc_files)}\n'
                     f'Total S3 "Oa" bands: {len(nc_bands)}\n')

        if rad_only:
            return nc_bands
        else:
            return nc_files
