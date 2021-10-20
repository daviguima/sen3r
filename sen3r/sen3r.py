import os
import sys
import pandas as pd
import netCDF4 as nc
from pathlib import Path
from datetime import datetime
from sen3r.commons import Utils, DefaultDicts
from sen3r.nc_engine import NcEngine, ParallelBandExtract

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

dd = DefaultDicts()


class Core:
    """
    Core methods to build the intermediate files from input Sentinel-3 NetCDF4 images.
    """

    def __init__(self, input_args: dict):
        self.arguments = input_args
        self.INPUT_DIR = self.arguments['input']
        self.OUTPUT_DIR = self.arguments['out']
        self.ROI = self.arguments['roi']
        self.product = self.arguments['product']
        self.AUX_DIR = os.path.join(self.OUTPUT_DIR, 'AUX_DATA')  # WARNING: just "AUX" is a reserved name in Windows!
        self.AUX_LOG_DIR = os.path.join(self.AUX_DIR)
        self.INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')
        self.arguments['logfile'] = os.path.join(self.arguments['out'], 'sen3r_' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.log')
        self.log = Utils.create_log_handler(self.arguments['logfile'])
        self.IMG_DIR = os.path.join(self.OUTPUT_DIR, 'images')
        # Section 5 for single source of truth for the version number:
        # https://packaging.python.org/guides/single-sourcing-package-version/#single-sourcing-the-version
        self.VERSION = metadata.version('sen3r')  # TODO: May be outdated depending on the environment installed version
        self.vertices = None  # Further declaration may happen inside build_intermediary_files
        self.sorted_file_list = None  # Declaration may happen inside build_intermediary_files

    @staticmethod
    def build_list_from_subset(input_directory_path):
        """
        Creates a python list containing the Posixpath from all the files inside the directory and sort them by date.
        """
        # convert input string to Posix
        in_path_obj = Path(input_directory_path)
        # get only the '20160425T134227' from the file name and use it to sort the list by date
        sorted_output_files = sorted(os.listdir(in_path_obj), key=lambda s: s.split('____')[1].split('_')[0])
        sorted_output_files_fullpath = [os.path.join(in_path_obj, img) for img in sorted_output_files]

        return sorted_output_files_fullpath

    def wfr2csv(self, wfr_img_folder, vertices=None, rgb=True):
        """
        Given a vector and a S3_OL2_WFR image, extract the NC data inside the vector.
        """

        img = wfr_img_folder

        # Class instance of NcEngine containing information about all the bands.
        nce = NcEngine(input_nc_folder=img, parent_log=self.log)

        # Convert the input ROI LAT/LON vertices to X,Y coordinates based on the geo_coordinates.nc file
        xy_vert, ll_vert = nce.latlon_2_xy_poly(poly_path=vertices)

        # II) Use the poly to generate an extraction mask:
        mask, cc, rr = nce.get_raster_mask(xy_vertices=xy_vert)  # cc = cols, rr = rows

        # III) Get the dictionary of available bands based on the product:
        if self.product and self.product.lower() == 'wfr':
            bdict = dd.wfr_files
        elif self.product and self.product.lower() == 'syn':
            bdict = dd.syn_files
        else:
            self.log.info(f'Invalid product: {self.product.upper()}.')
            sys.exit(1)

        # IV) Extract the data from the NetCDF using the mask
        pbe = ParallelBandExtract()
        df = pbe.parallel_get_bdata_in_nc(rr=rr, cc=cc,
                                          lon=nce.g_lon,
                                          lat=nce.g_lat,
                                          oaa=nce.OAA,
                                          oza=nce.OZA,
                                          saa=nce.SAA,
                                          sza=nce.SZA,
                                          nc_folder=nce.nc_folder,
                                          wfr_files_p=dd.wfr_files_p,
                                          parent_log=self.arguments['logfile'])

        if self.product.lower() == 'wfr':
            df = df.rename(columns=dd.wfr_vld_names)

        # TODO: check necessity of renaming SYNERGY colnames.
        # if self.product.lower() == 'syn':
        #     df = df.rename(columns=self.syn_vld_names)

        if len(df) == 0:
            self.log.info('EMPTY DATAFRAME WARNING! Unable to find valid pixels in file.')

        if rgb:
            colors = {}
            r_red, r_green, r_blue, roi_img_rgb = nce.get_rgb_from_poly(xy_vertices=xy_vert)
            colors['red'] = r_red
            colors['green'] = r_green
            colors['blue'] = r_blue
            return df, roi_img_rgb, colors

        return df

    def build_intermediary_files(self):
        """
        Parse the input arguments and return a path containing the output intermediary files.
        :return: l1_output_path Posixpath
        """
        self.log.info(f'Searching for WFR files inside: {self.INPUT_DIR}')
        self.log.info('Sorting input files by date.')
        self.sorted_file_list = self.build_list_from_subset(input_directory_path=self.INPUT_DIR)
        self.log.info(f'Input files found: {len(self.sorted_file_list)}')
        self.log.info('------')
        self.log.info(f'Generating ancillary data folder: {self.AUX_DIR}')
        Path(self.AUX_DIR).mkdir(parents=True, exist_ok=True)
        self.log.info(f'Attempting to extract geometries from: {self.ROI}')
        self.vertices = Utils.roi2vertex(roi=self.ROI, aux_folder_out=self.AUX_DIR)

        total = len(self.sorted_file_list)

        for n, img in enumerate(self.sorted_file_list):
            percent = int((n*100)/total)
            figdate = os.path.basename(img).split('____')[1].split('_')[0]
            self.log.info(f'({percent}%) {n+1} of {total} - {figdate}')
            self.wfr2csv(wfr_img_folder=img, vertices=self.vertices)
        pass

    def build_single_file(self):
        """
        Parse the input arguments and return a path containing the output intermediary file.
        :return: l1_output_path Posixpath
        """
        self.log.info(f'RUNNING IN SINGLE MODE.')
        self.log.info(f'Searching for WFR file inside: {self.INPUT_DIR}')
        self.log.info(f'Generating ancillary data folder: {self.AUX_DIR}')
        Path(self.AUX_DIR).mkdir(parents=True, exist_ok=True)
        self.log.info(f'Attempting to extract geometries from: {self.ROI}')
        self.vertices = Utils.roi2vertex(roi=self.ROI, aux_folder_out=self.AUX_DIR)

        band_data, image, color = self.wfr2csv(wfr_img_folder=self.INPUT_DIR, vertices=self.vertices)

        return band_data, image, color
