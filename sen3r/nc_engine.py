import os
import sys
import logging
import netCDF4 as nc
import numpy as np
import concurrent.futures
import pandas as pd
from skimage.draw import polygon
from pathlib import Path
from skimage.transform import resize
from sen3r import commons

dd = commons.DefaultDicts()
utils = commons.Utils()


class NcEngine:
    """
    Provide methods to manipulate NetCDF4 data from Sentinel-3 OLCI products.
    :input_nc_folder: This is the first param.
    :parent_log: This is a second param.
    """

    def __init__(self, input_nc_folder=None, parent_log=None, product='wfr'):
        self.log = parent_log
        self.nc_folder = Path(input_nc_folder)
        self.nc_base_name = os.path.basename(input_nc_folder).split('.')[0]
        self.product = product.lower()
        self.netcdf_valid_band_list = self.get_valid_band_files(rad_only=False)

        if self.product.lower() == 'wfr':
            self.log.info(f'{os.getpid()} - Initializing geometries for: {self.nc_base_name}')
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
            self.log.info(f'Invalid product: {self.product.upper()}.')
            sys.exit(1)

    def __repr__(self):
        return f'{type(self.t_lat)}, ' \
               f'{type(self.t_lon)}, ' \
               f'{type(self.g_lat)}, ' \
               f'{type(self.g_lon)}, ' \
               f'{type(self.OAA)}, ' \
               f'{type(self.OZA)}, ' \
               f'{type(self.SAA)},' \
               f'{type(self.SZA)},' \
               f'nc_base_name:{self.nc_base_name}'

    def get_valid_band_files(self, rad_only=True):
        """
        Search inside the .SEN3 image folder for files ended with .nc; If rad_only is True,
        only reflectance bands are returned, otherwise return everything ended with .nc extension.
        """
        if self.nc_folder is None:
            self.log.info('Unable to find files. NetCDF image folder is not defined during NcExplorer class instance.')
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

    def latlon_2_xy_poly(self, poly_path, go_parallel=True):
        """
        Given an input polygon and image, return a dataframe containing
        the data of the image that falls inside the polygon.
        """
        self.log.info(f'Converting the polygon coordinates into a matrix x,y poly...')
        # I) Convert the lon/lat polygon into a x/y poly:
        xy_vert, ll_vert = self._lat_lon_2_xy(poly_path=poly_path, parallel=go_parallel)

        return xy_vert, ll_vert

    def _lat_lon_2_xy(self, poly_path, geojson=True, parallel=True):
        """
        Takes in a polygon file and return a dataframe containing
        the data in each band that falls inside the polygon.
        """
        # self._test_initialized()

        if parallel:
            gpc = ParallelCoord()

            xy_vertices = [gpc.parallel_get_xy_poly(self.g_lat, self.g_lon, vert) for vert in poly_path]
        else:
            xy_vertices = [utils.get_x_y_poly(self.g_lat, self.g_lon, vert) for vert in poly_path]

        return xy_vertices, poly_path

    def get_raster_mask(self, xy_vertices):
        """
        Creates a boolean mask of 0 and 1 with the polygons using the nc resolution.
        """
        # self._test_initialized()
        # Generate extraction mask

        img = np.zeros(self.g_lon.shape)
        cc = np.ndarray(shape=(0,), dtype='int64')
        rr = np.ndarray(shape=(0,), dtype='int64')

        for vert in xy_vertices:
            t_rr, t_cc = polygon(vert[:, 0], vert[:, 1], self.g_lon.shape)
            img[t_rr, t_cc] = 1
            cc = np.append(cc, t_cc)
            rr = np.append(rr, t_rr)

        return img, cc, rr

    def get_rgb_from_poly(self, xy_vertices):

        # II) Get the bounding box:
        xmin, xmax, ymin, ymax = utils.bbox(xy_vertices)

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
            self.log.info(f'Invalid product: {self.product.upper()}.')
            sys.exit(1)

        # IV) Subset the bands using the bbox:
        red = red[ymin:ymax, xmin:xmax]
        green = green[ymin:ymax, xmin:xmax]
        blue = blue[ymin:ymax, xmin:xmax]

        # V) Stack the bands vertically:
        # https://stackoverflow.com/questions/10443295/combine-3-separate-numpy-arrays-to-an-rgb-image-in-python
        rgb_uint8 = (np.dstack((red, green, blue)) * 255.999).astype(np.uint8)

        return red, green, blue, rgb_uint8


class ParallelCoord:

    @staticmethod
    def vect_dist_subtraction(coord_pair, grid):
        subtraction = coord_pair - grid
        dist = np.linalg.norm(subtraction, axis=2)
        result = np.where(dist == dist.min())
        target_x_y = [result[0][0], result[1][0]]
        return target_x_y

    def parallel_get_xy_poly(self, lat_arr, lon_arr, polyline):
        # Stack LAT and LON in the Z axis
        grid = np.concatenate([lat_arr[..., None], lon_arr[..., None]], axis=2)

        # Polyline is a GeoJSON coordinate array
        polyline = polyline.squeeze()  # squeeze removes one of the dimensions of the array
        # https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html

        # Generate a list containing the lat, lon coordinates for each point of the input poly
        coord_vect_pairs = []
        for i in range(polyline.shape[0]):
            coord_vect_pairs.append(np.array([polyline[i, 1], polyline[i, 0]]).reshape(1, 1, -1))

        # for future reference
        # https://stackoverflow.com/questions/6832554/multiprocessing-how-do-i-share-a-dict-among-multiple-processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            try:
                result = list(executor.map(self.vect_dist_subtraction, coord_vect_pairs, [grid]*len(coord_vect_pairs)))

            except concurrent.futures.process.BrokenProcessPool as ex:
                self.log.info(f"{ex} This might be caused by limited system resources. "
                              f"Try increasing system memory or disable concurrent processing. ")

        return np.array(result)


class ParallelBandExtract:

    def __init__(self, parent_log=None):
        if parent_log:
            self.log = parent_log

    def _get_band_in_nc(self, file_n_band, rr, cc):

        print(f'{os.getpid()} | Extracting band: {file_n_band[1]} from file: {file_n_band[0]}.\n')
        # logging.info(f'{os.getpid()} | Extracting band: {file_n_band[1]} from file: {file_n_band[0]}.\n')
        # self.log.info(f'{os.getpid()} | Extracting band: {file_n_band[1]} from file: {file_n_band[0]}.\n')
        result = {}
        # load NetCDF folder + nc_file_name
        ds = nc.Dataset(file_n_band[0])
        # load the nc_band_name as a matrix and unmask its values
        band = ds[file_n_band[1]][:].data
        # extract the values of the matrix and return as a dict entry
        result[file_n_band[1]] = [band[x, y] for x, y in zip(rr, cc)]
        return result

    def nc_2_df(self, rr, cc, oaa, oza, saa, sza, lon, lat, nc_folder, wfr_files_p, parent_log=None):
        """
        Given an input polygon and image, return a dataframe containing
        the data of the image that falls inside the polygon.
        """
        if parent_log:
            self.log = logging.getLogger(name=parent_log)

        wfr_files_p = [(os.path.join(nc_folder, nc_file), nc_band) for nc_file, nc_band in wfr_files_p]

        # Generate initial df
        custom_subset = {'x': rr, 'y': cc}
        df = pd.DataFrame(custom_subset)
        df['lat'] = [lat[x, y] for x, y in zip(df['x'], df['y'])]
        df['lon'] = [lon[x, y] for x, y in zip(df['x'], df['y'])]
        df['OAA'] = [oaa[x, y] for x, y in zip(df['x'], df['y'])]
        df['OZA'] = [oza[x, y] for x, y in zip(df['x'], df['y'])]
        df['SAA'] = [saa[x, y] for x, y in zip(df['x'], df['y'])]
        df['SZA'] = [sza[x, y] for x, y in zip(df['x'], df['y'])]

        # Test the number of cores available, minimum requires 2.
        if (os.cpu_count() - 1) <= 0:
            self.log.info(f'Invalid number of CPU cores available: {os.cpu_count()}.')
            sys.exit(1)

        # Populate the initial DF with the output from the other bands
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            try:
                list_of_bands = list(executor.map(
                    self._get_band_in_nc, wfr_files_p,
                    [rr] * len(wfr_files_p),
                    [cc] * len(wfr_files_p)
                ))
            except concurrent.futures.process.BrokenProcessPool as ex:
                self.log.info(f"{ex} This might be caused by limited system resources. "
                              f"Try increasing system memory or disable concurrent processing. ")

        # For every returned dict inside the list, grab only the Key and append it at the final DF
        for b in list_of_bands:
            for key, val in b.items():
                df[key] = val

        # DROP NODATA
        idx_names = df[df['Oa08_reflectance'] == 65535.0].index
        df.drop(idx_names, inplace=True)
        return df
