import os
import sys
import logging
import netCDF4 as nc
import pandas as pd

from pathlib import Path
from skimage.transform import resize


class NcEngine:
    """
    Provide methods to manipulate NetCDF4 data from Sentinel-3 OLCI products.
    """

    def __init__(self, input_nc_folder=None, log_folder=None, product='WFR', initialize=True, external_use=False):
        self.initiated = initialize
        self.t_lat = None
        self.t_lon = None
        self.g_lat = None
        self.g_lon = None
        self.OAA = None
        self.OZA = None
        self.SAA = None
        self.SZA = None
        self.nc_base_name = None

        # Creating a new instance of logging.basicConfig for each parallel job.
        PID = str(os.getpid()) # Get the python PID
        if log_folder:
            LOG_FILE_NAME = os.path.join(log_folder, f'{PID}.log')

            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S',
                                handlers=[logging.FileHandler(LOG_FILE_NAME, mode='a'), logging.StreamHandler()])

        if not external_use:
            if input_nc_folder:
                self.nc_folder = Path(input_nc_folder)
                self.nc_base_name = os.path.basename(input_nc_folder).split('.')[0]
            else:
                self.nc_folder = input_nc_folder

            self.product = product.lower()
            self.os = os.name

            if self.nc_folder is None:
                logging.info(f'{PID} | Input NetCDF file folder not set. Proceed at your own risk.')
            else:
                # logging.info(f'Reading valid NetCDF files from: {input_nc_folder}')
                self.netcdf_valid_band_list = self.get_valid_band_files(rad_only=False)

            if input_nc_folder and initialize:
                self.initialize_geometries()
            else:
                logging.info(f'{PID} | initialize set to False, ignoring image geometries.')
                # logging.info(f'{PID} | This can be later done manually by calling initialize_geometries()')
                # logging.info(f'{PID} | after properly setting the nc_folder.')

    def initialize_geometries(self):
        """
        Manually load the metadata of the input NetCDF in case it was not automatically done
        during class instace.
        """
        PID = str(os.getpid())
        # logging.info(f'{PID} | Product set to {self.product.upper()}.')
        # logging.info(f'{PID} | Loading image bands into memory, this may take some time...')
        self.initiated = True
        if self.product.lower() == 'wfr':
            logging.info(f'{PID} | Begin: {self.nc_base_name}')
            geo_coord = nc.Dataset(self.nc_folder / 'geo_coordinates.nc')
            self.g_lat = geo_coord['latitude'][:]
            self.g_lon = geo_coord['longitude'][:]

            # Load and resize tie LON/LAT Bands using the geo_coordinates.nc file dimensions: (4091, 4865)
            # logging.info(f'{PID} | Loading tie_geo_coordinates.nc')
            tie_geo = nc.Dataset(self.nc_folder / 'tie_geo_coordinates.nc')
            # logging.info(f'{PID} | Extracting LAT from tie_geo_coordinates.nc')
            self.t_lat = tie_geo['latitude'][:]
            # logging.info(f'{PID} | Transforming LAT size.')
            self.t_lat = resize(self.t_lat, (self.g_lat.shape[0], self.g_lat.shape[1]), anti_aliasing=False)
            # logging.info(f'{PID} | Extracting LON from tie_geo_coordinates.nc')
            self.t_lon = tie_geo['longitude'][:]
            # logging.info(f'{PID} | Transforming LON size.')
            self.t_lon = resize(self.t_lon, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

            # Load and resize Sun Geometry Angle Bands using the geo_coordinates.nc file dimensions: (4091, 4865)
            t_geometries = nc.Dataset(self.nc_folder / 'tie_geometries.nc')
            # logging.info(f'{PID} | Extracting OAA from tie_geometries.nc')
            self.OAA = t_geometries['OAA'][:]
            # logging.info(f'{PID} | Transforming OAA size.')
            self.OAA = resize(self.OAA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)
            # logging.info(f'{PID} | Extracting OZA from tie_geometries.nc')
            self.OZA = t_geometries['OZA'][:]
            # logging.info(f'{PID} | Transforming OZA size.')
            self.OZA = resize(self.OZA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)
            # logging.info(f'{PID} | Extracting SAA from tie_geometries.nc')
            self.SAA = t_geometries['SAA'][:]
            # logging.info(f'{PID} | Transforming SAA size.')
            self.SAA = resize(self.SAA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)
            # logging.info(f'{PID} | Extracting SZA from tie_geometries.nc')
            self.SZA = t_geometries['SZA'][:]
            # logging.info(f'{PID} | Transforming SZA size.')
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

        if rad_only:
            return nc_bands
        else:
            return nc_files

    def get_data_in_poly(self, poly_path, go_parallel=True):
        """
        Given an input polygon and image, return a dataframe containing
        the data of the image that falls inside the polygon.
        """
        # I) Convert the lon/lat polygon into a x/y poly:
        xy_vert, ll_vert = self.get_xy_polygon_from_json(poly_path=poly_path)

        # II) Use the poly to generate an extraction mask:
        mask, cc, rr = self.get_raster_mask(xy_vertices=xy_vert)

        # III) Get the dictionary of available bands based on the product:
        if self.product.lower() == 'wfr':
            bdict = dd.wfr_files
        elif self.product.lower() == 'syn':
            bdict = dd.syn_files
        else:
            print(f'Invalid product: {self.product.upper()}.')
            sys.exit(1)

        if go_parallel:
            pbe = ParallelBandExtract()
            extracted_bands = pbe.parallel_get_bdata_in_nc(rr, cc, self.g_lon, self.g_lat,
                                                           self.nc_folder, dd.wfr_files_p)
            return extracted_bands

        else:
            # IV) Generate the dataframe (NON-PARALLEL):
            custom_subset = {'x': rr, 'y': cc}
            df = pd.DataFrame(custom_subset)
            print('extracting: LON / LAT')
            df['lat'] = [self.g_lat[x, y] for x, y in zip(df['x'], df['y'])]
            df['lon'] = [self.g_lon[x, y] for x, y in zip(df['x'], df['y'])]
            print('extracting: OAA / OZA / SAA / SZA')
            df['OAA'] = [self.OAA[x, y] for x, y in zip(df['x'], df['y'])]
            df['OZA'] = [self.OZA[x, y] for x, y in zip(df['x'], df['y'])]
            df['SAA'] = [self.SAA[x, y] for x, y in zip(df['x'], df['y'])]
            df['SZA'] = [self.SZA[x, y] for x, y in zip(df['x'], df['y'])]

            # V) Populate the DF with data from the other bands:
            for k in bdict:
                ds = nc.Dataset(self.nc_folder/k)
                for layer in bdict[k]:
                    print(f'extracting: {layer}')
                    band = ds[layer][:].data
                    df[layer] = [band[x, y] for x, y in zip(df['x'], df['y'])]

        idx_names = df[df['Oa08_reflectance'] == 65535.0].index
        df.drop(idx_names, inplace=True)

        if self.product.lower() == 'wfr':
            df = df.rename(columns=dd.wfr_vld_names)

        # TODO: check necessity of renaming SYNERGY colnames.
        # if self.product.lower() == 'syn':
        #     df = df.rename(columns=self.syn_vld_names)

        if len(df) == 0:
            print('EMPTY DATAFRAME WARNING! Unable to find valid pixels in file.')
        return df
