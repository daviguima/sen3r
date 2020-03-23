import os

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
    def __init__(self, gpt_path):
        self.gpt_path = gpt_path

    def subset_by_shp_and_get_pixel_values(self, shp_path):
        # os.popen('ogr2ogr -f "ESRI Shapefile" %s %s' % (output_shp, input_kml_path))
        os.popen('%s /d_drive_data/processing/shapefileExtraction_custom.xml -f CSV -t /d_drive_data/processing/output.txt -Ssource=/d_drive_data/L2_WFR/S3A_OL_2_WFR____20190309T141223_20190309T141523_20190310T211622_0179_042_167_3060_MAR_O_NT_002.SEN3' %
                 (self.gpt_path, shp_path))


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


def gdal_kml_to_shp(input_kml_path):
    output_shp = input_kml_path.split('.')[0]+'.shp'
    os.popen('ogr2ogr -f "ESRI Shapefile" %s %s' % (output_shp, input_kml_path))
    return output_shp


def get_gdal_value_by_lon_lat(tif_file, lon, lat):

    result = os.popen('gdallocationinfo -valonly -wgs84 %s %s %s' %
                      (tif_file, lon, lat)).read()
    # https://gis.stackexchange.com/questions/118397/storing-result-from-gdallocationinfo-as-variable-in-python
    return result


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