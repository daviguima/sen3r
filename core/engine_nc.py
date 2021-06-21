import os
import sys
import core.utils as utils
from core.commons import DefaultDicts as dd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc

from skimage.transform import resize
from pathlib import Path

# TODO: fix parallel run for windows
from parallel_get_xy_poly import ParallelCoord
from parallel_get_band_in_nc import ParallelBandExtract

try:
    from mpl_toolkits.basemap import Basemap
except:
    print('sen3r-NcExplorer: from mpl_toolkits.basemap import Basemap FAILED!\n'
          'You can still proceed without plotting any maps.\n')
    # logging.info('from mpl_toolkits.basemap import Basemap FAILED! '
    #       'You can still proceed without plotting any maps.')


class S3NcEngine:
    """
    Provide methods to manipulate NetCDF4 data from Sentinel-3 OLCI products.
    """
    def __init__(self, input_nc_folder=None, product='WFR', verbose=False, initialize=True, external_use=False):
        self.initiated = initialize
        self.verbose = verbose
        if not external_use:
            if input_nc_folder:
                self.nc_folder = Path(input_nc_folder)
            else:
                self.nc_folder = input_nc_folder
            self.product = product.lower()
            self.os = os.name
            self.class_label = 'SEN3R:nc_explorer'
            print(f'Declaring class instance from: {self.class_label}')
            if self.verbose:
                print(f'Verbose set to True.')
            if self.nc_folder is None:
                print(f'Input NetCDF file folder not set. Proceed at your own risk.')
            else:
                print('Reading valid NetCDF files inside image folder...')
                self.netcdf_valid_band_list = self.get_valid_band_files(rad_only=False)
            if input_nc_folder and initialize:
                self.initialize_geometries()
            else:

                print('initialize set to False, ignoring image geometries.'
                      ' This can be later done manually by calling initialize_geometries()'
                      ' after properly setting the nc_folder.')

    def initialize_geometries(self):
        """
        Manually load the metadata of the input NetCDF in case it was not automatically done
        during class instace.
        """
        print(f'Product set to {self.product.upper()}.')
        print('Loading image bands into memory, this may take a while...')
        self.initiated = True
        if self.product.lower() == 'wfr':
            geo_coord = nc.Dataset(self.nc_folder/'geo_coordinates.nc')
            self.g_lat = geo_coord['latitude'][:]
            self.g_lon = geo_coord['longitude'][:]

            # Load and resize tie LON/LAT Bands using the geo_coordinates.nc file dimensions: (4091, 4865)
            tie_geo = nc.Dataset(self.nc_folder/'tie_geo_coordinates.nc')
            self.t_lat = tie_geo['latitude'][:]
            self.t_lat = resize(self.t_lat, (self.g_lat.shape[0], self.g_lat.shape[1]), anti_aliasing=False)
            self.t_lon = tie_geo['longitude'][:]
            self.t_lon = resize(self.t_lon, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

            # Load and resize Sun Geometry Angle Bands using the geo_coordinates.nc file dimensions: (4091, 4865)
            t_geometries = nc.Dataset(self.nc_folder/'tie_geometries.nc')
            self.OAA = t_geometries['OAA'][:]
            self.OAA = resize(self.OAA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

            self.OZA = t_geometries['OZA'][:]
            self.OZA = resize(self.OZA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

            self.SAA = t_geometries['SAA'][:]
            self.SAA = resize(self.SAA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

            self.SZA = t_geometries['SZA'][:]
            self.SZA = resize(self.SZA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

        elif self.product.lower() == 'syn':
            dsgeo = nc.Dataset(self.nc_folder/'geolocation.nc')
            self.g_lat = dsgeo['lat'][:]
            self.g_lon = dsgeo['lon'][:]

        else:
            print(f'Invalid product: {self.product.upper()}.')
            self.initiated = False

    def _test_initiated(self):
        if not self.initiated:
            print('ERROR: Image class was not initialized. '
                  'This can be done manually by calling initialize_geometries()'
                  ' after properly setting the nc_folder.')
            sys.exit(1)

    @staticmethod  # TODO: maybe move it to utils?
    def ncdump(nc_fid, verb=True):
        # ported from py2.7 to py3 from:
        # http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html#code
        """
        ncdump outputs dimensions, variables and their attribute information.
        The information is similar to that of NCAR's ncdump utility.
        ncdump requires a valid instance of Dataset.

        Parameters
        ----------
        nc_fid : netCDF4.Dataset
            A netCDF4 dateset object
        verb : Boolean
            whether or not nc_attrs, nc_dims, and nc_vars are printed

        Returns
        -------
        nc_attrs : list
            A Python list of the NetCDF file global attributes
        nc_dims : list
            A Python list of the NetCDF file dimensions
        nc_vars : list
            A Python list of the NetCDF file variables
        """
        def print_ncattr(key):
            """
            Prints the NetCDF file attributes for a given key

            Parameters
            ----------
            key : unicode
                a valid netCDF4.Dataset.variables key
            """
            try:
                print("\t\ttype:", repr(nc_fid.variables[key].dtype))
                for ncattr in nc_fid.variables[key].ncattrs():
                    print('\t\t%s:' % ncattr, repr(nc_fid.variables[key].getncattr(ncattr)))
            except KeyError:
                print("\t\tWARNING: %s does not contain variable attributes" % key)

        # NetCDF global attributes
        nc_attrs = nc_fid.ncattrs()
        if verb:
            print("NetCDF Global Attributes:")
            for nc_attr in nc_attrs:
                print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
        nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
        # Dimension shape information.
        if verb:
            print("NetCDF dimension information:")
            for dim in nc_dims:
                print("\tName:", dim)
                print("\t\tsize:", len(nc_fid.dimensions[dim]))
                print_ncattr(dim)
        # Variable information.
        nc_vars = [var for var in nc_fid.variables]  # list of nc variables
        if verb:
            print("NetCDF variable information:")
            for var in nc_vars:
                if var not in nc_dims:
                    print('\tName:', var)
                    print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                    print("\t\tsize:", nc_fid.variables[var].size)
                    print_ncattr(var)
        return nc_attrs, nc_dims, nc_vars

    def get_footprint_xy(self):
        pass

    def footprint2raster(self):
        self._test_initiated()
        vertices = np.vstack(vertices)
        ymin = np.min(vertices[:, 0])
        ymax = np.max(vertices[:, 0])
        xmin = np.min(vertices[:, 1])
        xmax = np.max(vertices[:, 1])
        pass

    def get_valid_band_files(self, rad_only=True):
        """
        Search inside the .SEN3 image folder for files ended with .nc; If rad_only is True,
        only reflectance bands are returned, otherwise return everything ended with .nc extension.
        """
        if self.nc_folder is None:
            print('Unable to find files. NetCDF image folder is not defined during NcExplorer class instance.')
            sys.exit(1)

        sentinel_images_path = self.nc_folder

        # retrieve all files in folder
        files = os.listdir(sentinel_images_path)

        # extract only NetCDFs from the file list
        nc_files = [f for f in files if f.endswith('.nc')]

        # extract only the radiometric bands from the NetCDF list
        nc_bands = [b for b in nc_files if b.startswith('Oa')]

        if self.verbose:
            print(f'{self.class_label}.get_valid_band_files()\n'
                  f'Sentinel-3 Image folder:\n'
                  f'{sentinel_images_path}\n'
                  f'Total files in folder: {len(files)}\n'
                  f'Total NetCDF files: {len(nc_files)}\n'
                  f'Total S3 "Oa" bands: {len(nc_bands)}\n')

        if rad_only:
            return nc_bands
        else:
            return nc_files

    def get_point_data_in_single_band(self, band, lon=None, lat=None, target_lon=None, target_lat=None):
        # TODO: write docstrings
        if self.verbose:
            print(f'{self.class_label}.get_point_data_in_single_band()\n')

        lat = lat[:, :, np.newaxis]
        lon = lon[:, :, np.newaxis]
        grid = np.concatenate([lat, lon], axis=2)
        vector = np.array([target_lat, target_lon]).reshape(1, 1, -1)
        subtraction = vector - grid
        dist = np.linalg.norm(subtraction, axis=2)
        result = np.where(dist == dist.min())
        target_x_y = result[0][0], result[1][0]

        rad = band[target_x_y]

        return target_x_y, rad

    def get_point_data_in_bands(self, bands_dictionary, lon=None, lat=None, target_lon=None, target_lat=None):
        # TODO: write docstrings
        if self.verbose:
            print(f'{self.class_label}.get_point_data_in_bands()\n')

        # Mauricio's multidimensional wichcraft
        lat = lat[:, :, np.newaxis]
        lon = lon[:, :, np.newaxis]
        grid = np.concatenate([lat, lon], axis=2)
        vector = np.array([target_lat, target_lon]).reshape(1, 1, -1)
        subtraction = vector - grid  # TODO: study multidimensional vector subtraction
        dist = np.linalg.norm(subtraction, axis=2)
        result = np.where(dist == dist.min())
        target_x_y = result[0][0], result[1][0]

        rad_in_bands = []
        for b in bands_dictionary:
            rad = bands_dictionary[b][target_x_y]
            rad_in_bands.append(rad)

        return target_x_y, rad_in_bands

    def get_xy_polygon_from_json(self, poly_path, geojson=True):
        """
        Takes in a geojson polygon file and do black magic to return a dataframe
        containing the data in each band that falls inside the polygon.
        """
        self._test_initiated()
        if geojson:
            vertices = utils.geojson_to_polygon(poly_path)

        gpc = ParallelCoord()

        xy_vertices = [gpc.parallel_get_xy_poly(self.g_lat, self.g_lon, vert) for vert in vertices]

        return xy_vertices, vertices

    def get_raster_mask(self, xy_vertices):
        """
        Creates a boolean mask of 0 and 1 with the polygons using the nc resolution.
        """
        self._test_initiated()
        # Generate extraction mask
        from skimage.draw import polygon

        img = np.zeros(self.g_lon.shape)
        cc = np.ndarray(shape=(0,), dtype='int64')
        rr = np.ndarray(shape=(0,), dtype='int64')

        for vert in xy_vertices:
            t_rr, t_cc = polygon(vert[:, 0], vert[:, 1], self.g_lon.shape)
            img[t_rr, t_cc] = 1
            cc = np.append(cc, t_cc)
            rr = np.append(rr, t_rr)

        return img, cc, rr

    def get_rgb_from_poly(self, poly_path):

        # I) Convert the lon/lat polygon into a x/y poly:
        xy_vert, ll_vert = self.get_xy_polygon_from_json(poly_path=poly_path)

        # II) Get the bounding box:
        xmin, xmax, ymin, ymax = utils.bbox(xy_vert)

        # III) Get only the RGB bands:
        if self.product.lower() == 'wfr':
            ds = nc.Dataset(self.nc_folder / 'Oa08_reflectance.nc')
            red = ds['Oa08_reflectance'][:]
            ds = nc.Dataset(self.nc_folder / 'Oa06_reflectance.nc')
            green = ds['Oa06_reflectance'][:]
            ds = nc.Dataset(self.nc_folder / 'Oa03_reflectance.nc')
            blue = ds['Oa03_reflectance'][:]

        elif self.product.lower() == 'syn':
            ds = nc.Dataset(self.nc_folder / 'Syn_Oa08_reflectance.nc')
            red = ds['SDR_Oa08'][:]
            ds = nc.Dataset(self.nc_folder / 'Syn_Oa06_reflectance.nc')
            green = ds['SDR_Oa06'][:]
            ds = nc.Dataset(self.nc_folder / 'Syn_Oa03_reflectance.nc')
            blue = ds['SDR_Oa03'][:]
        else:
            print(f'Invalid product: {self.product.upper()}.')
            sys.exit(1)

        # IV) Subset the bands using the bbox:
        red = red[ymin:ymax, xmin:xmax]
        green = green[ymin:ymax, xmin:xmax]
        blue = blue[ymin:ymax, xmin:xmax]

        # V) Stack the bands vertically:
        # self.rgb = np.vstack((red, green, blue))
        return red, green, blue

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

    def _extract_band_data(self, full_nc_path, unmask=False):
        '''
        Assumes the NetCDF file to be a valid Sentinel-3 band file, like the one below:
        D:\S3\S3A_OL_1_EFR____20190830T140112_20190830T140412_20190831T183009_0179_048_338_3060_LN1_O_NT_002.SEN3\Oa01_radiance.nc
        '''
        nc_file = nc.Dataset(full_nc_path, 'r')
        if self.os == "nt":  # TODO: fix OS compatibility
            bname = full_nc_path.split('\\')
        else:
            bname = full_nc_path.split('/')

        bname = bname[-1].split('.')[0]
        extrated_band = nc_file.variables[bname][:]
        if unmask:
            extrated_band = extrated_band.data
        nc_file.close()  # it is good practice to close the netcdf after using it (:
        return bname, extrated_band

    def get_lon_lat_from_nc(self):
        # TODO: write docstrings
        if self.nc_folder is None:
            print('Unable to get Longitude and Latitude data if NetCDF image folder is not defined during NcExplorer class instance.')
            sys.exit(1)

        utils.tic()
        netcdf_files_folder = self.nc_folder

        if self.verbose:
            print(f'{self.class_label}.get_lon_lat_from_nc()\n')
            print(f'Extracting Lon/Lat dataframes from: \n{netcdf_files_folder}\n')
        # extract LAT LON from NetCDF
        if self.os == 'nt':
            coords_file = '\\geo_coordinates.nc'
        else:
            coords_file = '/geo_coordinates.nc'

        nc_coord = nc.Dataset(netcdf_files_folder + coords_file, 'r')

        lat = nc_coord.variables['latitude'][:]
        lat = lat.data
        lon = nc_coord.variables['longitude'][:]
        lon = lon.data
        t_hour, t_min, t_sec = utils.tac()
        if self.verbose:
            print(f'Longitude shape: {lon.shape}, size: {lon.size}')
            print(f'Latitude shape: {lat.shape}, size: {lat.size}\n')
            print(f'Done in {t_hour}h:{t_min}m:{t_sec}s')

        return lon, lat

    def extract_data_from_netcdf_bands(self, netcdf_valid_band_list, unmask=False):
        """ Returns a list of pandas DataFrames for each valid band in the input list.
            This function assumes the presence of a valid folder path containing NetCDF files inside.

        Parameters:
            netcdf_valid_band_list (list): A list of strings containing the name of
            each band as described inside de NetCDF file.

            unmask (bool): Weather or not the internal function _extract_band_data() should
            return a numpy masked array (Default = False).

        Returns:
            bands (list): A list of pandas DataFrames containing a 2D matrix for each
            band name passed in the netcdf_valid_band_list.

        """
        if self.nc_folder is None:
            print('Unable to extract band data if NetCDF image folder is not defined during NcExplorer class instance.')
            sys.exit(1)

        utils.tic()
        if self.verbose:
            print(f'{self.class_label}.extract_data_from_netcdf_bands()\n')

        nc_bands = netcdf_valid_band_list
        bands = {}
        total = len(nc_bands)
        for x, i in enumerate(nc_bands):
            if self.verbose:
                print(f'extracting band: {nc_bands[x]} -- {x + 1} of {total}')
            if unmask:
                band_name, df = self._extract_band_data(os.path.join(self.nc_folder, nc_bands[x]), unmask=unmask)
            else:
                band_name, df = self._extract_band_data(os.path.join(self.nc_folder, nc_bands[x]))
            bands[band_name] = df

        t_hour, t_min, t_sec = utils.tac()
        if self.verbose:
            print(f'\nDone in {t_hour}h:{t_min}m:{t_sec}s\n')
        return bands

    def extract_data_from_single_band(self, netcdf_valid_band_name, unmask=False):
        """ Returns a pandas DataFrames for a valid band in the input list.
            This function assumes the presence of a valid folder path containing NetCDF files inside.

        Parameters:
            netcdf_valid_band_name (str): A string containing the name of
            the band of interest as described inside de NetCDF file.

            unmask (bool): Weather or not the internal function _extract_band_data() should
            return a numpy masked array (Default = False).

        Returns:
            df (pd.DataFrame): pandas DataFrames containing a 2D matrix for the band name
            passed in the netcdf_valid_band_name.

        """
        if self.nc_folder is None:
            print('Unable to extract band data if NetCDF image folder is not defined during NcExplorer class instance.'
                  '\nTry pointing your class NcExplorer.nc_folder() to a valid NetCDF Sentinel-3 image folder.')
            sys.exit(1)

        utils.tic()
        if self.verbose:
            print(f'{self.class_label}.extract_data_from_single_band()\n')

        if self.verbose:
            print(f'extracting band: {netcdf_valid_band_name}')
        if unmask:
            band_name, df = self._extract_band_data(os.path.join(self.nc_folder, netcdf_valid_band_name), unmask=unmask)
        else:
            band_name, df = self._extract_band_data(os.path.join(self.nc_folder, netcdf_valid_band_name))

        t_hour, t_min, t_sec = utils.tac()
        if self.verbose:
            print(f'\nDone in {t_hour}h:{t_min}m:{t_sec}s\n')
        return df
