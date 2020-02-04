import os
import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid


class NcExplorer:
    """
    This class is intended to provide methods to manipulate NetCDF data from Sentinel-3
    """
    def __init__(self, input_nc_data=None):
        self.inputnc = input_nc_data

    @staticmethod
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

    @staticmethod
    def _temp():
        # folder = 'D:\S3\LV2_LFR\S3A_OL_2_LFR____20190830T140112_20190830T140412_20190831T185237_0179_048_338_3060_LN1_O_NT_002.SEN3'
        folder = 'D:\processing\S3A_OL_1_EFR____20190830T140112_20190830T140412_20190831T183009_0179_048_338_3060_LN1_O_NT_002.SEN3'
        band = '\Oa21_radiance.nc'

        coords = '\\geo_coordinates.nc'

        nc_fid = Dataset(folder + band, 'r')
        band_n = nc_fid.variables['Oa21_radiance'][:]

        nc_coord = Dataset(folder + coords, 'r')
        lat = nc_coord.variables['latitude'][:]
        lon = nc_coord.variables['longitude'][:]

        print(folder + '\n' + band + '\n')
        nc_attrs, nc_dims, nc_vars = explorer.ncdump(nc_fid)
        print('+++++++++++++++++++++++++++++++++++')

        print(folder + '\n' + coords + '\n')
        nc_attrs, nc_dims, nc_vars = explorer.ncdump(nc_coord)
        print('+++++++++++++++++++++++++++++++++++')

        nc_fid.close()
        nc_coord.close()

    @staticmethod
    def _plot(lat, lon, plot_var):
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

        m.pcolormesh(x, y, plot_var, shading='flat', cmap=plt.cm.jet)
        m.colorbar(location='right')

        lon, lat = -60.014493, -3.158980  # Manaus
        xpt, ypt = m(lon, lat)
        m.plot(xpt, ypt, 'rD')  # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
        m.drawcoastlines()
        plt.show()
        plt.figure()


if __name__ == "__main__":
    print("hello s3-frbr:nc_explorer!")
    explorer = NcExplorer()
    work_dir = 'D:\processing\\'
    file_name = work_dir + 'S3A_OL_1_EFR____20190830T140112_20190830T140412_20190831T183009_0179_048_338_3060_LN1_O_NT_002.SEN3'
    # retrieve all files in folder
    files = os.listdir(file_name)
    print('All files:\n')
    print(files)

    # extract LAT LON from NetCDF
    # coords = '\\geo_coordinates.nc'
    # nc_coord = Dataset(file_name + coords, 'r')
    # lat = nc_coord.variables['latitude'][:]
    # lon = nc_coord.variables['longitude'][:]

    def _extract_band(full_nc_path):
        print('extracting values from file:\n'+full_nc_path)
        # nc_file = Dataset(full_nc_path, 'r')
        bname = full_nc_path.split('\\')
        # extrated_band = nc_file.variables[bname[-1].split('.')[0]][:]
        print(bname)
        # return extrated_band

    # test for files ended in ".nc"
    nc_files = [f for f in files if f.endswith('.nc')]
    nc_bands = [b for b in nc_files if b.startswith('Oa')]
    print('All NetCDF files:\n')
    print(nc_files, end='\n\n')

    print('All bands files:\n')
    print(nc_bands, end='\n\n')

    for i in nc_bands:
        _extract_band(file_name + '\\' + nc_bands[0])


