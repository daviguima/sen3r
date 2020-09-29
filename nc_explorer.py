import os
import sys
# import logging
import utils
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/

# logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.DEBUG)

try:
    from mpl_toolkits.basemap import Basemap
except:
    logging.info('from mpl_toolkits.basemap import Basemap FAILED! '
          'You can still proceed without plotting any maps.')


class NcExplorer:
    """
    This class is intended to provide methods to manipulate NetCDF data from Sentinel-3
    """
    def __init__(self, input_nc_folder=None, verbose=False):
        self.nc_folder = input_nc_folder
        self.verbose = verbose
        self.os = os.name
        self.class_label = 'S3-FRBR:Nc_Explorer'
        print(f'Declaring class instance from: {self.class_label}')
        if self.verbose:
            print(f'Verbose set to True.')
        if self.nc_folder is None:
            print(f'Input NetCDF file folde not set. Proceed at your own risk.')

    s3_bands_l1 = {'Oa01': 400,
                   'Oa02': 412.5,
                   'Oa03': 442.5,
                   'Oa04': 490,
                   'Oa05': 510,
                   'Oa06': 560,
                   'Oa07': 620,
                   'Oa08': 665,
                   'Oa09': 673.75,
                   'Oa10': 681.25,
                   'Oa11': 708.75,
                   'Oa12': 753.75,
                   'Oa13': 761.25,
                   'Oa14': 764.375,
                   'Oa15': 767.5,
                   'Oa16': 778.75,
                   'Oa17': 865,
                   'Oa18': 885,
                   'Oa19': 900,
                   'Oa20': 940,
                   'Oa21': 1020}

    s3_bands_l2 = {'Oa01': 400,
                   'Oa02': 412.5,
                   'Oa03': 442.5,
                   'Oa04': 490,
                   'Oa05': 510,
                   'Oa06': 560,
                   'Oa07': 620,
                   'Oa08': 665,
                   'Oa09': 673.75,
                   'Oa10': 681.25,
                   'Oa11': 708.75,
                   'Oa12': 753.75,
                   'Oa16': 778.75,
                   'Oa17': 865,
                   'Oa18': 885,
                   'Oa21': 1020}

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

    @staticmethod  # TODO: as the name states, temp stuff either needs to be fixed, moved or removed.
    def _temp_plot(lon, lat, plot_var, roi_lon=None, roi_lat=None):
        # Miller projection:
        m = Basemap(projection='mill',
                    lat_ts=10,
                    llcrnrlon=lon.min(),
                    urcrnrlon=lon.max(),
                    llcrnrlat=lat.min(),
                    urcrnrlat=lat.max(),
                    resolution='c')

        # BR bbox
        # m = Basemap(projection='mill',
        #             llcrnrlat=-60,
        #             llcrnrlon=-90,
        #             urcrnrlat=20,
        #             urcrnrlon=-25)

        x, y = m(lon, lat)
        # x, y = m(lon, lat, inverse=True)

        m.pcolormesh(x, y, plot_var, shading='flat', cmap=plt.cm.jet)
        m.colorbar(location='right')  # ('top','bottom','left','right')

        # dd_lon, dd_lat = -60.014493, -3.158980  # Manaus
        # if roi_lon is not None and roi_lat is not None:
        xpt, ypt = m(roi_lon, roi_lat)
        m.plot(xpt, ypt, 'rD')  # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
        # m.drawcoastlines()
        plt.show()
        # plt.figure()

    def get_valid_band_files(self):
        # TODO: write docstrings
        if self.nc_folder is None:
            print('Unable to find files if NetCDF image folder is not defined during NcExplorer class instance.')
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

        return nc_bands

    def get_point_data_in_bands(self, bands_dictionary, lon=None, lat=None, target_lon=None, target_lat=None):
        # TODO: write docstrings
        if self.verbose:
            print(f'{self.class_label}.get_point_data_in_bands()\n')

        # TODO: this looks weird...
        # # [pos for pos, x in np.ndenumerate(np.array(lat)) if x == -0.21162]
        # x_lon, y_lon = np.unravel_index((np.abs(lon - target_lon)).argmin(), lon.shape)
        # x_lat, y_lat = np.unravel_index((np.abs(lat - target_lat)).argmin(), lat.shape)
        # print(f'x_lon:{x_lon} y_lon:{y_lon} \n'
        #       f'x_lat:{x_lat} y_lat:{y_lat}')
        #
        # relative_lat = y_lon + int((y_lat - y_lon) / 2)
        # relative_lon = x_lat + int((x_lon - x_lat) / 2)

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

    # TODO: this is very specific, make it more generic.
    def plot_s3_lv2_reflectances(self, radiance_list, icor, band_radiances, figure_title):
        # TODO: write docstrings
        ### L2 WFR
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Wavelenght (nm)')
        ax1.set_ylabel('Reflectance')
        ax1.set_title(figure_title, y=1, fontsize=16)
        ax1.plot(list(self.s3_bands_l2.values()), icor, label='L1 iCOR', marker='o')
        ax1.plot(list(self.s3_bands_l2.values()), band_radiances, label='L2 WFR', marker='o')
        ax1.axhline(y=0, xmin=0, xmax=1, linewidth=0.5, color='black', linestyle='--')
        ax1.set_xticks(list(self.s3_bands_l2.values()))
        ax1.set_xticklabels(list(self.s3_bands_l2.values()))
        ax1.tick_params(labelrotation=90, labelsize='small')
        # ax1.set_yticklabels(labels=np.linspace(
        #     ax1.get_yticks().min(), ax1.get_yticks().max(), len(ax1.get_yticks()) * 2),
        #     rotation=0)
        ax1.legend()
        ax2 = ax1.twiny()
        ax2.plot(np.linspace(min(list(self.s3_bands_l2.values())),
                             max(list(self.s3_bands_l2.values())),
                             num=len(self.s3_bands_l2)), band_radiances, alpha=0.0)
        ax2.set_xticks(list(self.s3_bands_l2.values()))
        ax2.set_xticklabels(list(self.s3_bands_l2.keys()))
        ax2.tick_params(labelrotation=90, labelsize='xx-small')
        ax2.set_title('Sentinel-3 Oa Bands', y=0.93, x=0.12, fontsize='xx-small')
        # ax2.grid()
        plt.show()

    def _extract_band_data(self, full_nc_path, unmask=False):
        '''
        Assumes the NetCDF file to be a valid Sentinel-3 band file, like the one below:
        D:\S3\S3A_OL_1_EFR____20190830T140112_20190830T140412_20190831T183009_0179_048_338_3060_LN1_O_NT_002.SEN3\Oa01_radiance.nc
        '''
        nc_file = Dataset(full_nc_path, 'r')
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

        nc_coord = Dataset(netcdf_files_folder + coords_file, 'r')

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
            df (pd.DataFrame): A pandas DataFrames containing a 2D matrix for the band name passed in the netcdf_valid_band_name.

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

if __name__ == "__main__":

    print('nc_explorer.py : hello from __main__ !')
    # s3_netcdf_folder = 'D:\processing\S3A_OL_1_EFR____20190830T140112_20190830T140412_20190831T183009_0179_048_338_3060_LN1_O_NT_002.SEN3'

    # exp = NcExplorer(input_nc_folder=s3_netcdf_folder,
    #                  verbose=True)

    # valid_nc_band_names = exp.get_valid_band_files()

    # lon, lat = exp.get_lon_lat_from_nc()

    # bands = exp.extract_data_from_netcdf_bands(valid_nc_band_names)

    # Where is Manaus in the lat lon netcdf matrix?
    # query_lon, query_lat = -60.014493, -3.158980

    # exp._temp_plot(lon, lat, df, query_lon, query_lat)

    # mat_x_y, band_radiances = exp.get_data_in_bands(bands, lon, lat, query_lon, query_lat)

    # file = 'C:\Temp\S3A_OL_1_EFR____20190830T140112_20190830T140412_20190831T183009_0179_048_338_3060_LN1_O_NT_002_iCOR.tif'
    # gdal_query_result = exp.get_gdal_value_by_lon_lat(file, query_lon, query_lat)
    #
    # icor = [float(x) for x in gdal_query_result.split()]
    #
    # exp.plot_s3_lv2_reflectances(band_radiances=band_radiances,
    #                              icor=icor,
    #                              figure_title=f'Sentinel-3 reflectance in pixel lon:{query_lon} lat:{query_lat}')

    # shapefile_roi_mask = exp.gdal_kml_to_shp('D:\processing\\rio_solimoes.kml')

    ### L1B
    # fig, ax1 = plt.subplots()
    # ax1.set_xlabel('Wavelenght (nm)')
    # ax1.set_ylabel('Radiance')
    # ax1.plot(list(s3_bands_l1.values()), band_radiances)
    # ax1.set_xticks(list(s3_bands_l1.values()))
    # ax1.set_xticklabels(list(s3_bands_l1.values()))
    # ax1.tick_params(labelrotation=90, labelsize='small')
    #
    # ax2 = ax1.twiny()
    # ax2.plot(np.linspace(min(list(s3_bands_l1.values())),
    #                      max(list(s3_bands_l1.values())),
    #                      num=len(s3_bands_l1)), band_radiances, alpha=0.0)
    # ax2.set_xticks(list(s3_bands_l1.values()))
    # ax2.set_xticklabels(list(s3_bands_l1.keys()))
    #
    # ax2.tick_params(labelrotation=90, labelsize='xx-small')
    # ax2.grid()
    # # ax2.set_visible(False)
    # plt.show()


