import os
import sys
import subprocess
import re
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


class GPTBridge:
    """
    So what this class actually does is to dynamically create an <xml> file that later will be embedded in
    a SNAP-GPT function call made from the python os.popen library.

    Arguments:
    gpt_sys_path (str): the absolute path in the system to the snap-gpt tool.
                    Something like: /user_home/snap/bin/gpt (required).

    output_path (str): where in the file system this function should save the results (required).

    output_format (str): which format should the output be: NetCDF4-CF, CSV, etc.

    graph_xml_path (str): the snap-gpt requires a <xml> file to work, this function will create one and will save it
                      in the directory described in here (defaults to the same path as output_path).

    Returns:

    """
    def __init__(self, gpt_sys_path, output_path, output_format='CSV'):
        self.gpt_path = gpt_sys_path
        self.output_path = output_path
        self.output_format = output_format
        self.graph_xml_path = self.output_path  # this will just assume the same path as the output.

    def __repr__(self):
        return f'gpt_bridge class instance using gpt: {self.gpt_path} and output: {self.output_path} as {self.output_format}'



    def get_pixels_by_kml(self, kml_path):
        """
        TODO: write docstring
        """
        gdtk = GDALBridge()  # create a class instance to exploit its tools and call it gdtk as short for gdal-toolkit.

        shp_path = gdtk.gdal_kml_to_shp(kml_path, self.output_path)  # converts the input kml file into .shp and return its path.

        wkt_str = gdtk.gdal_kml_to_wkt(kml_path, self.output_path, destroy_files=True, output_as_string=True)  # converts the input kml file into .wkt and return its path.

        wkt_bbox = gdtk.get_envelope_from_wkt(wkt_str)

        shp_name = os.path.basename(shp_path).split('.')[0]
        source_string = '${source}'
        xml_string = (f'<graph id="someGraphId">'
                      f'<version>1.0</version>'
                      f'<node id="regionalSubset">'
                      f'<operator>Subset</operator>'
                      f'<sources>'
                      f'<source>{source_string}</source>'
                      f'</sources>'
                      f'<parameters>'
                      f'<geoRegion>{wkt_bbox}</geoRegion>'
                      f'<copyMetadata>true</copyMetadata>'
                      f'</parameters>'
                      f'</node>'
                      f'<node id="importShapefile">'
                      f'<operator>Import-Vector</operator>'
                      f'<sources>'
                      f'<source>regionalSubset</source>'
                      f'</sources>'
                      f'<parameters>'
                      f'<vectorFile>{shp_path}</vectorFile>'
                      f'<separateShapes>false</separateShapes>'
                      f'</parameters>'
                      f'</node>'
                      f'<node id="maskArea">'
                      f'<operator>Land-Sea-Mask</operator>'
                      f'<sources>'
                      f'<source>importShapefile</source>'
                      f'</sources>'
                      f'<parameters>'
                      f'<landMask>false</landMask>'
                      f'<useSRTM>false</useSRTM>'
                      f'<geometry>{shp_name}</geometry>'
                      f'<invertGeometry>false</invertGeometry>'
                      f'<shorelineExtension>0</shorelineExtension>'
                      f'</parameters>'
                      f'</node>'
                      f'</graph>')

        filename = f'gpt_{shp_name}.xml'
        output_xml = os.path.join(self.output_path, filename)

        with open(output_xml, 'w') as f:
            f.write(xml_string)

        return print(output_xml)

        # https://forum.step.esa.int/t/pixel-extraction-from-many-sentinel-3-products-in-snap/13464/2?u=daviguima
        # <!-- gpt shapefileExtraction.xml -f NetCDF4-CF -t <target_product_path> -Ssource=<source_product_path>
        # Instead of NetCDF4-CF the format CSV can be used, if ASCII output is desired. -->
        # os.popen(f'{self.gpt_path} {self.graph_xml_path} -f {self.output_format} -t %s -Ssource=/d_drive_data/L2_WFR/S3A_OL_2_WFR____20190309T141223_20190309T141523_20190310T211622_0179_042_167_3060_MAR_O_NT_002.SEN3')


class GDALBridge:
    """
    TODO: write docstrings
    """
    @staticmethod
    def get_envelope_from_wkt(raw_wkt):
        """
        Calculates the bounding box rectangle from a given wkt file.
        Implementation advices from: https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html

        Arguments: raw_wkt (str): A string containing a WKT geometry like:
        'POLYGON ((-61.3059852737445 -3.3072135055284,-60.5018493671852 -3.3072135055284,
        -60.5018493671852 -3.62205969741951, -61.3059852737445 -3.62205969741951,
        -61.3059852737445 -3.3072135055284))'

        Returns: wkt (str): A string containing a WKT geometry with the bounding box rectangle
                            calculated from the input raw_wkt.
        """
        from osgeo import ogr
        geom_poly = ogr.CreateGeometryFromWkt(raw_wkt)
        geom_poly_envelope = geom_poly.GetEnvelope()

        minX = geom_poly_envelope[0]
        minY = geom_poly_envelope[2]
        maxX = geom_poly_envelope[1]
        maxY = geom_poly_envelope[3]
        '''
        coord0----coord1
        |           |
        coord3----coord2
        '''
        coord0 = minX, maxY
        coord1 = maxX, maxY
        coord3 = minX, minY
        coord2 = maxX, minY

        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint_2D(*coord0)
        ring.AddPoint_2D(*coord1)
        ring.AddPoint_2D(*coord2)
        ring.AddPoint_2D(*coord3)
        ring.AddPoint_2D(*coord0)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        # Export geometry to WKT
        wkt = poly.ExportToWkt()
        return wkt

    @staticmethod
    def gdal_kml_to_shp(input_kml_path, output_shape_path=None):
        """
        Converts a given google earth engine KML file to SHP.
        """
        if output_shape_path:
            filename = os.path.basename(input_kml_path).split('.')[0]+'.shp'
            output_shp = os.path.join(output_shape_path, filename)
        else:
            output_shp = input_kml_path.split('.')[0]+'.shp'

        subprocess.call(f'ogr2ogr -f "ESRI Shapefile" {output_shp} {input_kml_path}', shell=True)

        return output_shp

    @staticmethod
    def gdal_kml_to_wkt(input_kml_path, output_wkt_path=None, destroy_files=False, output_as_string=False):
        """
        Converts a given Google earth KML file to text format following the WKT standard.
        """
        if output_wkt_path:
            filename = os.path.basename(input_kml_path).split('.')[0] + '.csv'
            output_csv = os.path.join(output_wkt_path, filename)
            filename = os.path.basename(input_kml_path).split('.')[0] + '.wkt'
            output_wkt = os.path.join(output_wkt_path, filename)
        else:
            output_csv = input_kml_path.split('.')[0] + '.csv'
            output_wkt = input_kml_path.split('.')[0] + '.wkt'

        ogr2ogr_proccess = subprocess.Popen(['ogr2ogr', '-overwrite', '-f', 'CSV', '-dsco', 'GEOMETRY=AS_WKT', output_csv, input_kml_path])
        ogr2ogr_proccess.wait()

        wkt_file = open(output_csv, 'r')
        wkt_txt = wkt_file.read()

        pattern = re.search(r'\(\(.*\)\)', wkt_txt, re.MULTILINE)

        if pattern:
            with open(output_wkt, 'w') as f:
                final_wkt_string = f'POLYGON {pattern.group()}\n'
                f.write(final_wkt_string)
        else:
            sys.exit(f'Unable to find WKT pattern inside file {wkt_file}')

        if destroy_files and output_as_string:
            os.remove(output_csv)
            os.remove(output_wkt)
        elif destroy_files:
            os.remove(output_csv)

        if output_as_string:
            return final_wkt_string
        else:
            return output_wkt

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
