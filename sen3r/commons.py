import os
import sys
import time
import json
import logging
import zipfile
import subprocess

import numpy as np

try:
    from osgeo import gdal
except:
    print("Unable to import osgeo.gdal! SEN3R can still operate but some critical functions may fail.")


from PIL import Image


class DefaultDicts:

    clustering_methods = {'M0': ['Oa17_reflectance:float', 'Oa21_reflectance:float'],
                          'M1': ['Oa17_reflectance:float', 'T865:float', 'A865:float'],
                          'M2': ['Oa21_reflectance:float', 'T865:float', 'A865:float'],
                          'M3': ['Oa17_reflectance:float', 'Oa21_reflectance:float', 'T865:float'],
                          'M4': ['Oa08_reflectance:float', 'Oa17_reflectance:float', 'Oa21_reflectance:float'],
                          'M5': ['Oa04_reflectance:float', 'Oa08_reflectance:float', 'Oa21_reflectance:float']}

    wfr_norm_s3_bands = ['Oa01_reflectance:float',
                         'Oa02_reflectance:float',
                         'Oa03_reflectance:float',
                         'Oa04_reflectance:float',
                         'Oa05_reflectance:float',
                         'Oa06_reflectance:float',
                         'Oa07_reflectance:float',
                         'Oa08_reflectance:float',
                         'Oa09_reflectance:float',
                         'Oa10_reflectance:float',
                         'Oa11_reflectance:float',
                         'Oa12_reflectance:float',
                         'Oa16_reflectance:float',
                         'Oa17_reflectance:float',
                         'Oa18_reflectance:float',
                         'Oa21_reflectance:float']

    wfr_l2_bnames = {'B1-400': 'Oa01: 400 nm',
                     'B2-412.5': 'Oa02: 412.5 nm',
                     'B3-442.5': 'Oa03: 442.5 nm',
                     'B4-490': 'Oa04: 490 nm',
                     'B5-510': 'Oa05: 510 nm',
                     'B6-560': 'Oa06: 560 nm',
                     'B7-620': 'Oa07: 620 nm',
                     'B8-665': 'Oa08: 665 nm',
                     'B9-673.75': 'Oa09: 673.75 nm',
                     'B10-681.25': 'Oa10: 681.25 nm',
                     'B11-708.75': 'Oa11: 708.75 nm',
                     'B12-753.75': 'Oa12: 753.75 nm',
                     'B16-778.75': 'Oa16: 778.75 nm',
                     'B17-865': 'Oa17: 865 nm',
                     'B18-885': 'Oa18: 885 nm',
                     'B21-1020': 'Oa21: 1020 nm'}

    wfr_int_flags = [1, 2, 4, 8, 8388608, 16777216, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                     32768, 65536, 131072, 262144, 524288, 2097152, 33554432, 67108864, 134217728, 268435456,
                     4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944,
                     549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416,
                     35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312,
                     1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992,
                     18014398509481984, 36028797018963968]

    wfr_str_flags = ['INVALID', 'WATER', 'LAND', 'CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'SNOW_ICE',
                     'INLAND_WATER', 'TIDAL', 'COSMETIC', 'SUSPECT', 'HISOLZEN', 'SATURATED', 'MEGLINT', 'HIGHGLINT',
                     'WHITECAPS', 'ADJAC', 'WV_FAIL', 'PAR_FAIL', 'AC_FAIL', 'OC4ME_FAIL', 'OCNN_FAIL', 'KDM_FAIL',
                     'BPAC_ON', 'WHITE_SCATT', 'LOWRW', 'HIGHRW', 'ANNOT_ANGSTROM', 'ANNOT_AERO_B', 'ANNOT_ABSO_D',
                     'ANNOT_ACLIM', 'ANNOT_ABSOA', 'ANNOT_MIXR1', 'ANNOT_DROUT', 'ANNOT_TAU06', 'RWNEG_O1', 'RWNEG_O2',
                     'RWNEG_O3', 'RWNEG_O4', 'RWNEG_O5', 'RWNEG_O6', 'RWNEG_O7', 'RWNEG_O8', 'RWNEG_O9', 'RWNEG_O10',
                     'RWNEG_O11', 'RWNEG_O12', 'RWNEG_O16', 'RWNEG_O17', 'RWNEG_O18', 'RWNEG_O21']

    # https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-3-olci/level-2/quality-and-science-flags-op
    wfr_bin2flags = {0: 'INVALID',
                     1: 'WATER',
                     2: 'LAND',
                     3: 'CLOUD',
                     23: 'CLOUD_AMBIGUOUS',
                     24: 'CLOUD_MARGIN',
                     4: 'SNOW_ICE',
                     5: 'INLAND_WATER',
                     6: 'TIDAL',
                     7: 'COSMETIC',
                     8: 'SUSPECT',
                     9: 'HISOLZEN',
                     10: 'SATURATED',
                     11: 'MEGLINT',
                     12: 'HIGHGLINT',
                     13: 'WHITECAPS',
                     14: 'ADJAC',
                     15: 'WV_FAIL',
                     16: 'PAR_FAIL',
                     17: 'AC_FAIL',
                     18: 'OC4ME_FAIL',
                     19: 'OCNN_FAIL',
                     21: 'KDM_FAIL',
                     25: 'BPAC_ON',
                     26: 'WHITE_SCATT',
                     27: 'LOWRW',
                     28: 'HIGHRW',
                     32: 'ANNOT_ANGSTROM',
                     33: 'ANNOT_AERO_B',
                     34: 'ANNOT_ABSO_D',
                     35: 'ANNOT_ACLIM',
                     36: 'ANNOT_ABSOA',
                     37: 'ANNOT_MIXR1',
                     38: 'ANNOT_DROUT',
                     39: 'ANNOT_TAU06',
                     40: 'RWNEG_O1',
                     41: 'RWNEG_O2',
                     42: 'RWNEG_O3',
                     43: 'RWNEG_O4',
                     44: 'RWNEG_O5',
                     45: 'RWNEG_O6',
                     46: 'RWNEG_O7',
                     47: 'RWNEG_O8',
                     48: 'RWNEG_O9',
                     49: 'RWNEG_O10',
                     50: 'RWNEG_O11',
                     51: 'RWNEG_O12',
                     52: 'RWNEG_O16',
                     53: 'RWNEG_O17',
                     54: 'RWNEG_O18',
                     55: 'RWNEG_O21'}

    # WFR pixels with these flags will be removed:
    wfr_remove = ['INVALID', 'CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'SNOW_ICE',
                  'SUSPECT', 'SATURATED', 'AC_FAIL', 'MEGLINT', 'HIGHGLINT', 'LOWRW']

    # Pixels must include these flags:
    wfr_keep = ['INLAND_WATER']

    # TODO: used in parallel only (verify).
    wfr_files_p = (('w_aer.nc', 'A865'),
                   ('w_aer.nc', 'T865'),
                   ('Oa01_reflectance.nc', 'Oa01_reflectance'),
                   ('Oa02_reflectance.nc', 'Oa02_reflectance'),
                   ('Oa03_reflectance.nc', 'Oa03_reflectance'),
                   ('Oa04_reflectance.nc', 'Oa04_reflectance'),
                   ('Oa05_reflectance.nc', 'Oa05_reflectance'),
                   ('Oa06_reflectance.nc', 'Oa06_reflectance'),
                   ('Oa07_reflectance.nc', 'Oa07_reflectance'),
                   ('Oa08_reflectance.nc', 'Oa08_reflectance'),
                   ('Oa09_reflectance.nc', 'Oa09_reflectance'),
                   ('Oa10_reflectance.nc', 'Oa10_reflectance'),
                   ('Oa11_reflectance.nc', 'Oa11_reflectance'),
                   ('Oa12_reflectance.nc', 'Oa12_reflectance'),
                   ('Oa16_reflectance.nc', 'Oa16_reflectance'),
                   ('Oa17_reflectance.nc', 'Oa17_reflectance'),
                   ('Oa18_reflectance.nc', 'Oa18_reflectance'),
                   ('Oa21_reflectance.nc', 'Oa21_reflectance'),
                   ('wqsf.nc', 'WQSF'))

    syn_files = {
        'Syn_AOT550.nc': ['T550'],
        'Syn_Angstrom_exp550.nc': ['A550']
    }

    syn_vld_names = {}  # TODO: populate with the valid Synergy equivalent netcdf files.

    wfr_files = {
        'w_aer.nc': ['A865', 'T865'],
        'Oa01_reflectance.nc': ['Oa01_reflectance'],
        'Oa02_reflectance.nc': ['Oa02_reflectance'],
        'Oa03_reflectance.nc': ['Oa03_reflectance'],
        'Oa04_reflectance.nc': ['Oa04_reflectance'],
        'Oa05_reflectance.nc': ['Oa05_reflectance'],
        'Oa06_reflectance.nc': ['Oa06_reflectance'],
        'Oa07_reflectance.nc': ['Oa07_reflectance'],
        'Oa08_reflectance.nc': ['Oa08_reflectance'],
        'Oa09_reflectance.nc': ['Oa09_reflectance'],
        'Oa10_reflectance.nc': ['Oa10_reflectance'],
        'Oa11_reflectance.nc': ['Oa11_reflectance'],
        'Oa12_reflectance.nc': ['Oa12_reflectance'],
        'Oa16_reflectance.nc': ['Oa16_reflectance'],
        'Oa17_reflectance.nc': ['Oa17_reflectance'],
        'Oa18_reflectance.nc': ['Oa18_reflectance'],
        'Oa21_reflectance.nc': ['Oa21_reflectance'],
        'wqsf.nc': ['WQSF']
    }

    wfr_vld_names = {
        'lon': 'longitude:double',
        'lat': 'latitude:double',
        'Oa01_reflectance': 'Oa01_reflectance:float',
        'Oa02_reflectance': 'Oa02_reflectance:float',
        'Oa03_reflectance': 'Oa03_reflectance:float',
        'Oa04_reflectance': 'Oa04_reflectance:float',
        'Oa05_reflectance': 'Oa05_reflectance:float',
        'Oa06_reflectance': 'Oa06_reflectance:float',
        'Oa07_reflectance': 'Oa07_reflectance:float',
        'Oa08_reflectance': 'Oa08_reflectance:float',
        'Oa09_reflectance': 'Oa09_reflectance:float',
        'Oa10_reflectance': 'Oa10_reflectance:float',
        'Oa11_reflectance': 'Oa11_reflectance:float',
        'Oa12_reflectance': 'Oa12_reflectance:float',
        'Oa16_reflectance': 'Oa16_reflectance:float',
        'Oa17_reflectance': 'Oa17_reflectance:float',
        'Oa18_reflectance': 'Oa18_reflectance:float',
        'Oa21_reflectance': 'Oa21_reflectance:float',
        'OAA': 'OAA:float',
        'OZA': 'OZA:float',
        'SAA': 'SAA:float',
        'SZA': 'SZA:float',
        'WQSF': 'WQSF_lsb:double',
        'A865': 'A865:float',
        'T865': 'T865:float',
    }

    l1_wave_bands = {
        'Oa01': 400,
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
        'Oa21': 1020
    }

    s3_bands_l2 = {
        'Oa01': 400,
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
        'Oa21': 1020
    }

    dbs_colors = {-1: 'C0', 0: 'C1', 1: 'C2', 2: 'C3', 3: 'C4', 4: 'C5', 5: 'C6', 6: 'C7', 7: 'C8', 8: 'C9', 9: 'k',
                  10: 'k', 11: 'k', 12: 'k', 13: 'k', 14: 'k', 15: 'k', 16: 'k', 17: 'k', 18: 'k', 19: 'k', 20: 'k'}

    dbs_s3b_l2 = {
        'Oa01_reflectance:float': 400,
        'Oa02_reflectance:float': 412.5,
        'Oa03_reflectance:float': 442.5,
        'Oa04_reflectance:float': 490,
        'Oa05_reflectance:float': 510,
        'Oa06_reflectance:float': 560,
        'Oa07_reflectance:float': 620,
        'Oa08_reflectance:float': 665,
        'Oa09_reflectance:float': 673.75,
        'Oa10_reflectance:float': 681.25,
        'Oa11_reflectance:float': 708.75,
        'Oa12_reflectance:float': 753.75,
        'Oa16_reflectance:float': 778.75,
        'Oa17_reflectance:float': 865,
        'Oa18_reflectance:float': 885,
        'Oa21_reflectance:float': 1020
    }


class Utils:

    def __init__(self, parent_log=None):
        if parent_log:
            self.log = parent_log

    @staticmethod
    def create_log_handler(fname):
        # based in this answer:
        # https://stackoverflow.com/questions/62835466/create-a-separate-logger-for-each-process-when-using-concurrent-futures-processp
        logger = logging.getLogger(name=fname)
        logger.setLevel(logging.INFO)

        fileHandler = logging.FileHandler(fname)
        fileHandler.setLevel(logging.INFO)

        logger.addHandler(fileHandler)
        # logger.addHandler(logging.StreamHandler())

        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

        fileHandler.setFormatter(formatter)

        return logger

    @staticmethod
    def tic():
        global _start_time
        _start_time = time.time()

    @staticmethod
    def tac():
        t_sec = round(time.time() - _start_time)
        (t_min, t_sec) = divmod(t_sec, 60)
        (t_hour, t_min) = divmod(t_min, 60)
        return t_hour, t_min, t_sec

    @staticmethod
    def repeat_to_length(s, wanted):
        return (s * (wanted // len(s) + 1))[:wanted]

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def get_available_cores(self):
        cpu_count = os.cpu_count() - 1
        if cpu_count <= 0:
            self.log.info(f'Invalid number of CPU cores available: {os.cpu_count()}.')
            sys.exit(1)
        elif cpu_count > 61:
            cpu_count = 61

        return cpu_count

    @staticmethod
    def pil_grid(images, max_horiz=np.iinfo(int).max):
        """
        Combine several images
        https://stackoverflow.com/a/46877433/2238624
        """
        n_images = len(images)
        n_horiz = min(n_images, max_horiz)
        #     h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
        h_sizes, v_sizes = [0] * n_horiz, [0] * ((n_images // n_horiz) + (1 if n_images % n_horiz > 0 else 0))
        for i, im in enumerate(images):
            h, v = i % n_horiz, i // n_horiz
            h_sizes[h] = max(h_sizes[h], im.size[0])
            v_sizes[v] = max(v_sizes[v], im.size[1])
        h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
        im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
        for i, im in enumerate(images):
            im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
        return im_grid

    @staticmethod
    def geojson_to_polygon(poly_or_path, read_file=True):
        """
        Transform a given input .geojson file into a list of coordinates
        poly_path: string (Path to .geojson file)
        return: dict (Containing one or several polygons depending on the complexity of the input .json)
        """
        if read_file:
            with open(poly_or_path) as f:
                data = json.load(f)
        else:
            data = poly_or_path

        poly_lst = []
        for feature in data['features']:
            poly = feature['geometry']['coordinates']
            poly = np.array(poly[0])
            poly_lst.append(poly)

        vertices = [poly[:, :2] for poly in poly_lst]

        return vertices

    @staticmethod
    def shp2geojson_geopandas(shp_file_path, json_out_path):
        import geopandas
        shpfile = geopandas.read_file(shp_file_path)
        f_name = os.path.basename(shp_file_path).split('.')[0]
        shpfile.to_file(os.path.join(json_out_path, f_name + '.geojson'), driver='GeoJSON')

    @staticmethod
    def shp2json_pyshp(shp_file_path):
        import shapefile
        with shapefile.Reader(shp_file_path) as shp:
            geojson_data = shp.__geo_interface__
        return geojson_data

    @staticmethod
    # KML -> GEOJson : https://github.com/mrcagney/kml2geojson
    def kml2json_gdal(input_kml_path, output_json_path=None):
        """
        Converts a given google earth engine KML file to SHP.
        """
        if output_json_path:
            filename = os.path.basename(input_kml_path).split('.')[0] + '.geojson'
            output_json = os.path.join(output_json_path, filename)
        else:
            output_json = input_kml_path.split('.')[0] + '.geojson'

        logging.info('Opening subprocess call to GDAL:ogr2ogr...')
        subprocess.call(f'ogr2ogr -f "GeoJSON" {output_json} {input_kml_path}', shell=True)
        logging.info(f'{output_json} created.')

        return output_json

    @staticmethod
    # KML -> GEOJson : https://github.com/mrcagney/kml2geojson
    def kmz2kml_unzip(input_kmz_path, output_kml_path=None):
        """
        Converts a given google earth engine KMZ file to KML by extracting its content.
        """
        fname = os.path.basename(input_kmz_path).split('.')[0]
        if output_kml_path:
            filename = fname + '.geojson'
            output_kml = os.path.join(output_kml_path, filename)
        else:
            output_json = input_kmz_path.split('.')[0] + '.kml'

        logging.info('Extracting KMZ content...')
        with zipfile.ZipFile(input_kmz_path, 'r') as zip_ref:
            zip_ref.extractall(output_kml_path)
        # Search for everything ended in .kml inside the extracted files folder (expects 1) and remove it from the list.
        kml_file = [os.path.join(output_kml_path, f) for f in os.listdir(output_kml_path) if f.endswith('.kml')].pop()
        # Create a new name for the output file.
        kml_result = os.path.join(output_kml_path, fname + '.kml')
        # Rename whatever came out of the KMZ.zip to fname.kml
        os.rename(kml_file, kml_result)
        logging.info(f'{kml_result} created.')
        return kml_result

    @staticmethod
    def roi2vertex(roi, aux_folder_out=os.getcwd()):
        """
        Test the format of the input vector file and return a list of vertices.

        :param roi: (str) path to input vector file to be tested and processed.
        :param aux_folder_out: (str) path to save ancillary data.
        :return python list of arrays: (list) containing one array for each vector feature.
        """

        def parse_kml(path2vector):
            logging.info('KML file detected. Attempting to parse...')
            vtx_path = Utils.kml2json_gdal(input_kml_path=path2vector, output_json_path=aux_folder_out)
            return Utils.geojson_to_polygon(vtx_path)

        def parse_kmz(path2vector):
            logging.info('KMZ file detected. Attempting to parse...')
            # KMZ -> KML
            vtx_path = Utils.kmz2kml_unzip(input_kmz_path=path2vector, output_kml_path=aux_folder_out)
            # KML -> JSON
            vtx = Utils.kml2json_gdal(input_kml_path=vtx_path, output_json_path=aux_folder_out)
            # JSON -> VTX
            return Utils.geojson_to_polygon(vtx)

        def parse_json(path2vector):
            logging.info('JSON file detected. Attempting to parse...')
            return Utils.geojson_to_polygon(path2vector)

        def parse_shp(path2vector):
            logging.info('SHP file detected. Attempting to parse...')
            # Convert SHP -> JSON
            vtx = Utils.shp2json_pyshp(path2vector)
            # Convert JSON -> vertex
            # return Utils.geojson_to_polygon(vtx, read_file=False)
            return Utils.shp2geojson_geopandas(vtx)

        roi_typename = os.path.basename(roi).split('.')[1]

        roi_options = {'kml': parse_kml,
                       'kmz': parse_kmz,
                       'shp': parse_shp,
                       'json': parse_json,
                       'geojson': parse_json}

        if roi_typename in roi_options:
            vertex = roi_options[roi_typename](roi)
        else:
            logging.info(f'Input ROI {os.path.basename(roi)} not recognized as a valid vector file. '
                         f'Make sure the input file is of type .shp .kml .kmz .json or .geojson')
            sys.exit(1)

        return vertex

    @staticmethod
    def get_x_y_poly(lat_arr, lon_arr, polyline):
        grid = np.concatenate([lat_arr[..., None], lon_arr[..., None]], axis=2)

        # Polyline is a GeoJSON coordinate array
        polyline = polyline.squeeze()

        # loop through each vertice
        vertices = []
        for i in range(polyline.shape[0]):
            vector = np.array([polyline[i, 1], polyline[i, 0]]).reshape(1, 1, -1)
            subtraction = vector - grid
            dist = np.linalg.norm(subtraction, axis=2)
            result = np.where(dist == dist.min())
            target_x_y = [result[0][0], result[1][0]]

            vertices.append(target_x_y)
        return np.array(vertices)

    @staticmethod
    def bbox(vertices):
        """
        Get the bounding box of the vertices. Just for visualization purposes
        """
        vertices = np.vstack(vertices)
        ymin = np.min(vertices[:, 0])
        ymax = np.max(vertices[:, 0])
        xmin = np.min(vertices[:, 1])
        xmax = np.max(vertices[:, 1])
        return xmin, xmax, ymin, ymax
