import os
import sys
import subprocess
import logging
import re
import numpy as np
import utils
import shutil


class ICORBridge:
    # https://forum.step.esa.int/t/icor-module-using-gpt-operator-icor-s2/7205/8
    """
    WINDOWS ONLY!
    """
    @staticmethod
    def run_iCOR_on_image(s3_image_path, simec=True, keep_water=False, destination=None):
        """
        s3_image_path is expected to be something like:
        D:\L1_EFR\S3A_OL_1_EFR____20181106T140113_20181106T140413_20181107T182233_0179_037_338_3060_LN1_O_NT_002.SEN3
        """
        # WARNING, this may change according to your iCOR installation folder.
        # TODO: write a conditional check for the iCOR and its python path.
        icor_python = r'"C:\Program Files\VITO\iCOR\bin\Python27\python.exe"'
        icor_path = r'"C:\Program Files\VITO\iCOR\src\icor.py"'

        output_tif = str(os.path.basename(s3_image_path).split('.')[0]) + '_processed.tif'
        output_hdr = str(os.path.basename(s3_image_path).split('.')[0]) + '_processed.hdr'

        icor_output_tif = f'"C:\\Temp\\{output_tif}"'
        icor_output_hdr = f'"C:\\Temp\\{output_hdr}"'

        # this is splitting the name of the sensor and adding its last letter A/B to the end of the sensor variable.
        sensor_check = os.path.basename(s3_image_path).split('_')[0][-1]
        sensor = f'3{sensor_check}'

        netcdf_xml_manifest = f'"{s3_image_path}\\xfdumanifest.xml"'

        # Ternary operator
        str_simec = "true" if simec else "false"
        str_keep_water = "true" if keep_water else "false"

        icor_call = f'{icor_python} {icor_path} ' \
                    f'--keep_intermediate false ' \
                    f'--cloud_average_threshold 0.23 ' \
                    f'--cloud_low_band B02 ' \
                    f'--cloud_low_threshold 0.2 ' \
                    f'--aot true ' \
                    f'--aot_window_size 100 ' \
                    f'--simec {str_simec} ' \
                    f'--bg_window 1 ' \
                    f'--aot_override 0.1 ' \
                    f'--ozone true ' \
                    f'--aot_override 0.1 ' \
                    f'--ozone_override 0.33 ' \
                    f'--watervapor true ' \
                    f'--wv_override 2.0 ' \
                    f'--water_band B18 ' \
                    f'--water_threshold 0.06 ' \
                    f'--data_type S3 ' \
                    f'--output_file {icor_output_tif} ' \
                    f'--sensor {sensor} ' \
                    f'--apply_svc_gains true ' \
                    f'--inlandwater true ' \
                    f'--productwater true ' \
                    f'--keep_land false ' \
                    f'--keep_water {str_keep_water} ' \
                    f'--project true {netcdf_xml_manifest}'
        utils.tic()
        print(f'Applying iCOR to image:\n{s3_image_path}\nSensor: {sensor}\n')
        print(f'******* S3FRBR iCOR call ********')
        print(f'{icor_call}')
        print(f'*********************************')

        icor_proccess = subprocess.Popen(icor_call)
        icor_proccess.wait()
        # # stdout, sterr = icor_proccess.communicate()
        # # return_code = icor_proccess.returncode
        #
        # completed_process = subprocess.run(icor_call, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # return_code = completed_process.returncode
        # stdout = completed_process.stdout
        # sterr = completed_process.stderr
        #
        # print(f"subprocess.run complete:\nreturn_code:{return_code} \nstdout:{stdout} \nstderr:{sterr}")

        t_hour, t_min, t_sec = utils.tac()
        print(f'************** DONE *************')
        print(f'TIME: {t_hour}h:{t_min}m:{t_sec}s')
        print(f'*********************************')
        # return icor_call

        if destination:
            try:
                print(utils.repeat_to_length('*', len(icor_output_tif)))
                print(f'Moving iCOR output .tif from:\n'
                      f'{icor_output_tif}\n'
                      f'to:\n'
                      f'{destination}\\{output_tif}')
                print(utils.repeat_to_length('*', len(icor_output_tif)))

                destination_output = f'{destination}\\{output_tif}'
                shutil.move(icor_output_tif, destination_output)

                print(f'Moving iCOR output .hdr from:\n'
                      f'{icor_output_hdr}\n'
                      f'to:\n'
                      f'{destination}\\{output_hdr}')

                shutil.move(icor_output_hdr, destination + '\\' + output_hdr)
            except:
                print("Unexpected error:", sys.exc_info()[0])
        pass


class SNAPPYBridge:

    def write_geotiff(product_input_path, product_output_path):
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
    def __init__(self, gpt_sys_path, output_path, output_format='CSV', verbose=False, kml_path=None):
        self.gpt_path = gpt_sys_path
        self.output_path = output_path
        self.output_format = output_format
        self.graph_xml_path = self.output_path  # this will just assume the same path as the output.
        self.verbose = verbose
        self.kml_path = kml_path

    def __repr__(self):
        return f'gpt_bridge class instance using gpt: {self.gpt_path} and output: {self.output_path} as {self.output_format}'

    def get_pixels_by_kml(self, s3imgfolder):
        """
        TODO: write docstring
        """
        kml_path = self.kml_path
        # create empty folder to store intermediate files
        output_folder = 'AUX_' + str(os.path.basename(s3imgfolder).split('.')[0])
        output_folder_path = os.path.join(self.output_path, output_folder)

        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)
            print("Directory created @", output_folder_path)
        else:
            print("Directory already exists @", output_folder_path)

        # create a class instance to exploit its tools and call it gdtk as short for gdal-toolkit.
        gdtk = GDALBridge()

        # converts the input kml file into .shp and return its path.
        shp_path = gdtk.gdal_kml_to_shp(kml_path, output_folder_path)

        # converts the input kml file into .wkt and return it as a string.
        wkt_str = gdtk.gdal_kml_to_wkt(kml_path, output_folder_path, destroy_files=True, output_as_string=True)

        # calculate the bounding box geometry based in the given wkt string
        wkt_bbox = gdtk.get_envelope_from_wkt(wkt_str)

        shp_name = os.path.basename(shp_path).split('.')[0]
        graph_id = 'S3FRBRGraphId'
        version = '1.0'
        source_string = '${source}'
        filename = f'gpt_{shp_name}.xml'

        output_xml = os.path.join(output_folder_path, filename)
        output_pixel_txt = os.path.basename(s3imgfolder).split('.')[0] + '_subset.txt'
        output_pixel_txt_path = os.path.join(self.output_path, output_pixel_txt)

        xml_string = (f'<graph id="{graph_id}">'
                      f'<version>{version}</version>'
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

        with open(output_xml, 'w') as f:
            f.write(xml_string)
        if os.name == 'nt':
            gpt_command_str = f'"{self.gpt_path}" "{output_xml}" -f CSV -t "{output_pixel_txt_path}" -Ssource={s3imgfolder}'
        else:
            gpt_command_str = f'{self.gpt_path} {output_xml} -f CSV -t {output_pixel_txt_path} -Ssource={s3imgfolder}'

        logging.info(f'Calling SNAP-GPT using command string:\n\n{gpt_command_str}\n\n')
        # gpt_proccess = subprocess.Popen(gpt_command_str.split())
        gpt_proccess = subprocess.Popen(gpt_command_str)
        gpt_proccess.wait()
        logging.info(f'GPT processing complete, output file should be generated at:\n{output_pixel_txt_path}\n\n')

        return output_pixel_txt_path

class GDALBridge:
    from osgeo import ogr
    from osgeo import gdal
    """
    TODO: write docstrings
    """
    def get_envelope_from_wkt(self, raw_wkt):
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
        ogr = self.ogr
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
        logging.info(f'Starting GDALBridge: gdal_kml_to_wkt...\n')
        if output_wkt_path:
            filename = os.path.basename(input_kml_path).split('.')[0] + '.csv'
            output_csv = os.path.join(output_wkt_path, filename)
            filename = os.path.basename(input_kml_path).split('.')[0] + '.wkt'
            output_wkt = os.path.join(output_wkt_path, filename)
        else:
            output_csv = input_kml_path.split('.')[0] + '.csv'
            output_wkt = input_kml_path.split('.')[0] + '.wkt'

        # ogr2ogr_proccess = subprocess.Popen(['ogr2ogr', '-overwrite', '-f', 'CSV', '-dsco', 'GEOMETRY=AS_WKT', output_csv, input_kml_path])
        # ogr2ogr_proccess.wait()

        if os.name == 'nt':
            ogr2ogr_command_str = f'ogr2ogr -overwrite -f CSV -dsco GEOMETRY=AS_WKT "{output_csv}" "{input_kml_path}"'
        else:
            ogr2ogr_command_str = f'ogr2ogr -overwrite -f CSV -dsco GEOMETRY=AS_WKT {output_csv} {input_kml_path}'

        logging.info(f'Making internal call to GDAL using command string:\n{ogr2ogr_command_str}\n')

        ogr2ogr_proccess = subprocess.Popen(ogr2ogr_command_str)
        ogr2ogr_proccess.wait()

        with open(output_csv, 'r') as wkt_file:
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

    def get_tiff_band_count(self, file):
        gdal = self.gdal
        src_ds = gdal.Open(file)
        if src_ds is None:
            logging.error('Unable to open input .tif')
            sys.exit(1)
        logging.info(f"[ INPUT RASTER ]: {file}")
        logging.info(f"[ RASTER BAND COUNT ]: {src_ds.RasterCount}")
        return src_ds.RasterCount

    # https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    def read_tiff_bands(self, file):
        gdal = self.gdal
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
