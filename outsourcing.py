import os
import sys
import numpy as np


def snappy_bridge(product_input_path, product_output_path):
    """
    product_input_path = '/path/to/S3B_OL_1_EFR____FILE.dim'
    product_output_path = '/path/to/output/'
    """
    from snappy import ProductIO
    # http://step.esa.int/docs/v7.0/apidoc/engine/org/esa/snap/core/dataio/ProductIO.html
    # https://forum.step.esa.int/t/writing-out-a-product-as-geotiff-bigtiff/9263/2
    # https://senbox.atlassian.net/wiki/spaces/SNAP/pages/42041346/What+to+consider+when+writing+an+Operator+in+Python
    p = ProductIO.readProduct(product_input_path)  # read product
    ProductIO.writeProduct(p, product_output_path, 'GeoTIFF-BigTIFF')  # write product


class gpt_bridge:
    """
    TODO: write docstrings
    """
    def __init__(self, gpt_sys_path, output_path, output_format='CSV'):
        self.gpt_path = gpt_sys_path  # something like: /user_home/snap/bin/gpt
        self.output_path = output_path  # where to save the output.
        self.output_format = output_format  # which format should the output be: NetCDF4-CF, CSV, etc.
        self.graph_xml_path = self.output_path  # this will just assume the same path as the output.

    def get_pixels_by_kml(self, kml_path):
        gdtk = gdal_bridge()  # create a class instance to exploit its tools and call it gdtk as short for gdal-toolkit.
        shp_path = gdtk.gdal_kml_to_shp(kml_path, self.output_path)  # converts the input kml file into .shp and return its path.

        shp_name = os.path.basename(shp_path).split('.')[0]
        source_string = '${source}'
        xml_string = f"""
        <graph id="someGraphId">
          <version>1.0</version>
          <node id="regionalSubset">
            <operator>Subset</operator>
            <sources>
              <source>{source_string}</source>
            </sources>
            <parameters>
              <geoRegion>{wkt_polygon}</geoRegion>
              <copyMetadata>true</copyMetadata>
            </parameters>
          </node>
          <node id="importShapefile">
            <operator>Import-Vector</operator>
            <sources>
              <source>regionalSubset</source>
            </sources>
            <parameters>
              <vectorFile>{shp_path}</vectorFile>
              <separateShapes>false</separateShapes>
            </parameters>
          </node>
          <node id="maskArea">
            <operator>Land-Sea-Mask</operator>
            <sources>
              <source>importShapefile</source>
            </sources>
            <parameters>
              <landMask>false</landMask>
              <useSRTM>false</useSRTM>
              <geometry>{shp_name}</geometry>
              <invertGeometry>false</invertGeometry>
              <shorelineExtension>0</shorelineExtension>
            </parameters>
          </node>
        </graph>
        """

        # https://forum.step.esa.int/t/pixel-extraction-from-many-sentinel-3-products-in-snap/13464/2?u=daviguima
        # <!-- gpt shapefileExtraction.xml -f NetCDF4-CF -t <target_product_path> -Ssource=<source_product_path>
        # Instead of NetCDF4-CF the format CSV can be used, if ASCII output is desired. -->
        os.popen(f'{self.gpt_path} {self.graph_xml_path} -f {self.output_format} -t %s -Ssource=/d_drive_data/L2_WFR/S3A_OL_2_WFR____20190309T141223_20190309T141523_20190310T211622_0179_042_167_3060_MAR_O_NT_002.SEN3')


class gdal_bridge:
    """
    TODO: write docstrings
    """
    @staticmethod
    def gdal_kml_to_shp(input_kml_path, output_shape_path=None):
        if output_shape_path:
            filename = os.path.basename(input_kml_path).split('.')[0]+'.shp'
            output_shp = os.path.join(output_shape_path, filename)
        else:
            output_shp = input_kml_path.split('.')[0]+'.shp'
        os.popen('ogr2ogr -f "ESRI Shapefile" %s %s' % (output_shp, input_kml_path))
        return output_shp

    @staticmethod
    def get_gdal_value_by_lon_lat(tif_file, lon, lat):

        result = os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' %
                          (tif_file, lon, lat)).read()
        # https://gis.stackexchange.com/questions/118397/storing-result-from-gdallocationinfo-as-variable-in-python
        return result

    @staticmethod
    def read_tiff_bands(file):
        from osgeo import gdal

        src_ds = gdal.Open(file)
        if src_ds is None:
            print('Unable to open input .tif')
            sys.exit(1)

        print("[ RASTER BAND COUNT ]: ", src_ds.RasterCount)
        for band in range(src_ds.RasterCount):
            band += 1
            print("[ GETTING BAND ]: ", band)
            srcband = src_ds.GetRasterBand(band)
            if srcband is None:
                continue

            stats = srcband.GetStatistics(True, True)
            if stats is None:
                continue

            print("[ STATS ] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" %
                  (stats[0], stats[1], stats[2], stats[3]))


def keep_df_interval(keepfrom: 0.0, keepto: 1.0, dataframe, target_col: str):
    """
    Drop data outside the given interval
    :param keepfrom: minimun range of rain rate in millimeters (float)
    :param keepto: maximum range of rain rate in millimeters (float)
    :param dataframe:
    :param target_col:
    :return:
    """
    keepinterval = np.where((dataframe[target_col] >= keepfrom) &
                            (dataframe[target_col] <= keepto))
    result = dataframe.iloc[keepinterval]
    return result
