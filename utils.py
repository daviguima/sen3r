import time
import numpy as np


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    return t_hour, t_min, t_sec


def repeat_to_length(s, wanted):
    return (s * (wanted//len(s) + 1))[:wanted]


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