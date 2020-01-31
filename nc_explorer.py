import datetime as dt  # Python standard library datetime  module
import numpy as np
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    print("hello s3-frbr:nc_explorer!")
    explorer = NcExplorer()
    folder = 'D:\processing\S3A_OL_1_EFR____20190830T140112_20190830T140412_20190831T183009_0179_048_338_3060_LN1_O_NT_002.SEN3'
    band = '\Oa08_radiance.nc'
    coords = '\\tie_geo_coordinates.nc'

    nc_fid = Dataset(folder+band, 'r')
    radiance_b8 = nc_fid.variables['Oa08_radiance'][:]

    nc_coord = Dataset(folder + coords, 'r')
    lat = nc_coord.variables['latitude'][:]
    lon = nc_coord.variables['longitude'][:]

    print(folder + '\n' + band + '\n')
    nc_attrs, nc_dims, nc_vars = explorer.ncdump(nc_fid)
    print('+++++++++++++++++++++++++++++++++++')

    print(folder + '\n' + coords + '\n')
    nc_attrs, nc_dims, nc_vars = explorer.ncdump(nc_coord)
    print('+++++++++++++++++++++++++++++++++++')

    m = Basemap(projection='merc',
                llcrnrlat=-80, urcrnrlat=-60,
                llcrnrlon=10, urcrnrlon=-10,
                lat_ts=20, resolution='c')
    m.drawcoastlines()
    m.fillcontinents(color='coral', lake_color='aqua')
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90., 91., 30.))
    m.drawmeridians(np.arange(-180., 181., 60.))
    m.drawmapboundary(fill_color='aqua')
    plt.title("Mercator Projection")
    plt.show()