import os
import sys
import logging
import concurrent.futures

from pathlib import Path
from datetime import datetime
from sen3r.commons import Utils
from sen3r.nc4_agent import NcEngine

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata


class Core:
    """
    Core methods to build the intermediate files from input Sentinel-3 NetCDF4 images.
    """

    def __init__(self, input_args: dict):
        self.arguments = input_args
        self.INPUT_DIR = self.arguments['input']
        self.OUTPUT_DIR = self.arguments['out']
        self.ROI = self.arguments['roi']
        self.AUX_DIR = os.path.join(self.OUTPUT_DIR, 'extras')  # AUX is a reserved name in Windows, therefore extras...
        self.AUX_LOG_DIR = os.path.join(self.AUX_DIR, 'PARALLEL_LOG')
        self.IMG_DIR = os.path.join(self.OUTPUT_DIR, 'images')
        self.INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')
        self.LOGFILE = os.path.join(self.OUTPUT_DIR, f'SEN3R_{self.INSTANCE_TIME_TAG}.log')
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

    def wfr2csv(self, wfr_img_folder, vertices):

        img = wfr_img_folder

        # Class instance of NcEngine containing information about all the bands.
        nce = NcEngine(input_nc_folder=img, log_folder=self.AUX_LOG_DIR, product='wfr')

        # Get the values inside the input ROI vertices
        df = nce.get_data_inside_polygon(vertices=vertices)

        # if df is not None:
        # logging.info(f'Saving DF: {f_b_name}')
        # df.to_csv(os.path.join(out_dir, f_b_name + '.csv'), index=False)
        pass

    def build_intermediary_files(self):
        """
        Parse the input arguments and return a path containing the output intermediary files.
        :return: l1_output_path Posixpath
        """
        logging.info(f'Searching for WFR files inside: {self.INPUT_DIR}')
        logging.info('Sorting input files by date.')
        self.sorted_file_list = self.build_list_from_subset(input_directory_path=self.INPUT_DIR)
        logging.info(f'Input files found: {len(self.sorted_file_list)}')
        logging.info('------')
        logging.info(f'Generating ancillary data folder: {self.AUX_DIR}')
        Path(self.AUX_DIR).mkdir(parents=True, exist_ok=True)
        logging.info(f'Attempting to extract geometries from: {self.ROI}')
        self.vertices = Utils.get_roi_format2vertex(roi=self.ROI, aux_folder_out=self.AUX_DIR)
        logging.info(f'Generating ancillary parallel jobs log folder: {self.AUX_LOG_DIR}')
        Path(self.AUX_LOG_DIR).mkdir(parents=True, exist_ok=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            try:
                result = list(executor.map(self.wfr2csv, self.sorted_file_list, self.vertices))
            except concurrent.futures.process.BrokenProcessPool as ex:
                print(f"{ex} This might be caused by limited system resources. "
                      f"Try increasing system memory or disable concurrent processing. ")
        pass
