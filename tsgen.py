import os
import sys
import logging
import time
import pandas as pd
import numpy as np
import outsourcing as out
import matplotlib.pyplot as plt
import concurrent.futures

from datetime import datetime
from scipy.signal import argrelextrema
from scipy import stats

import matplotlib
import matplotlib.cm as cm
from nc_explorer import NcExplorer

class TsGenerator():

    def __init__(self):
        # Setting up information logs
        logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

    exp = NcExplorer()
    imgdpi = 100

    bname_dict = {'B1-400': 'Oa01: 400 nm',
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

    int_flags = [1, 2, 4, 8, 8388608, 16777216, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                 131072, 262144, 524288, 2097152, 33554432, 67108864, 134217728, 268435456, 4294967296, 8589934592,
                 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776,
                 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664,
                 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248,
                 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968]

    str_flags = ['INVALID', 'WATER', 'LAND', 'CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'SNOW_ICE', 'INLAND_WATER',
                 'TIDAL', 'COSMETIC', 'SUSPECT', 'HISOLZEN', 'SATURATED', 'MEGLINT', 'HIGHGLINT', 'WHITECAPS', 'ADJAC',
                 'WV_FAIL', 'PAR_FAIL', 'AC_FAIL', 'OC4ME_FAIL', 'OCNN_FAIL', 'KDM_FAIL', 'BPAC_ON', 'WHITE_SCATT',
                 'LOWRW', 'HIGHRW', 'ANNOT_ANGSTROM', 'ANNOT_AERO_B', 'ANNOT_ABSO_D', 'ANNOT_ACLIM', 'ANNOT_ABSOA',
                 'ANNOT_MIXR1', 'ANNOT_DROUT', 'ANNOT_TAU06', 'RWNEG_O1', 'RWNEG_O2', 'RWNEG_O3', 'RWNEG_O4',
                 'RWNEG_O5', 'RWNEG_O6', 'RWNEG_O7', 'RWNEG_O8', 'RWNEG_O9', 'RWNEG_O10', 'RWNEG_O11', 'RWNEG_O12',
                 'RWNEG_O16', 'RWNEG_O17', 'RWNEG_O18', 'RWNEG_O21']

    # bin_to_flag = {}
    # for n, i in enumerate(int_flags):
    #     bin = "{0:b}".format(i)
    #     print(f'{n} | {str_flags[n]} - {i} - {bin} - {(len(bin))}')
    #     bin_to_flag[len(bin) - 1] = str_flags[n]
    #
    # # https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-3-olci/level-2/quality-and-science-flags-op

    bin2flag = {0: 'INVALID',
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

    def get_flags(self, val):

        if isinstance(val, float):
            binexval = "{0:b}".format(int(val))
        elif isinstance(val, int):
            binexval = "{0:b}".format(val)
        else:
            print('Input must be of type int or float.')
            return False
        if binexval != '11111111111111111111111111111110':
            flags = [self.bin2flag[n] for n, e in enumerate(binexval[::-1]) if e == '1']
        else:
            return False

        return flags

    # -----------------------------------------------------------------------------------
    # DEFINITION OF FLAGS TO KEEP AND REMOVE
    remove = ['INVALID',
              'CLOUD',
              'CLOUD_AMBIGUOUS',
              'CLOUD_MARGIN',
              'SNOW_ICE',
              'SUSPECT',
              'SATURATED',
              'AC_FAIL',
              'MEGLINT',  # experimental
              'HIGHGLINT',  # experimental
              'ADJAC',
              'LOWRW']  # experimental

    # LOWRW # experimental

    # MUST HAVE
    keep = ['INLAND_WATER']

    def get_quality(self, checklist):

        if checklist:
            if all(i in checklist for i in self.keep):
                if any(i in checklist for i in self.remove):
                    return 0
                else:
                    return 1
            else:
                return 0
        else:
            return 0

    def get_glint(self, df):
        """
        Calculates glint angle based on paper:
        An Enhanced Contextual Fire Detection Algorithm for MODIS
        https://www.sciencedirect.com/science/article/pii/S0034425703001846
        https://doi.org/10.1016/S0034-4257(03)00184-6
        """

        df['GLINT'] = np.degrees(np.arccos(np.cos(np.deg2rad(df['OZA:float'])) *
                                           np.cos(np.deg2rad(df['SZA:float'])) -
                                           np.sin(np.deg2rad(df['OZA:float'])) *
                                           np.sin(np.deg2rad(df['SZA:float'])) *
                                           np.cos(np.deg2rad(abs(df['SAA:float'] - df['OAA:float'])))))

        # excel version
        # =GRAUS(ACOS(COS(RADIANOS(OZA))*COS(RADIANOS(SZA))-SEN(RADIANOS(OZA))*SEN(RADIANOS(SZA))*COS(RADIANOS(ABS(SAA-OAA)))))

        return df

    @staticmethod
    def get_pct_valid(df, total_ini):
        total_end = len(df)
        df['PCTVLDPX'] = (total_end * 100) / total_ini
        return df

    def add_flags_to_df(self, df):
        df['FLAGS'] = df['WQSF_lsb:double'].apply(self.get_flags)
        df['QUALITY'] = df['FLAGS'].apply(self.get_quality)
        return df

    def update_csvs(self, csv_path, savepath=False, threshold=False):
        """
        Given an CSV of pixels extracted using GPT(SNAP), filter the dataset and add some new columns.

        Input:
            csv_path (string): complete path to the CSV to be updated.
            ex: "D:\\sentinel3\\inputs\\S3B_OL_2_WFR____20191002T140633_subset_masked.txt"

            savepath (string): system folder where to save the modified csv.
            ex: "D:\\sentinel3\\outputs"

            When savepath is not given, the new DF will no be saved, but it will still be returned.

        Output:
            df (pandas dataframe): in-memory version of the input data that was read and modified from csv_path.
        """

        # read text file and convert to pandas dataframe
        df = pd.read_csv(csv_path, sep='\t', skiprows=1)

        # Get + Delete: indexes for which column LON has value 0
        indexNames = df[df['longitude:double'] == 0].index
        df.drop(indexNames, inplace=True)

        # This should represent 100% of the pixels inside the SHP area.
        df['PCTVLDPX'] = len(df)

        # Assuming the reflectance of water pixels should not be above 0.2 (Oa17:865nm), we will drop using this threshold
        if threshold:
            indexNames = df[df['Oa17_reflectance:float'] > threshold].index
            # Delete these row indexes from dataFrame
            df.drop(indexNames, inplace=True)

        # Add new FLAGS and QUALITY cols
        df = self.add_flags_to_df(df)

        # Delete indexes for which QUALITY = 0
        indexNames = df[df['QUALITY'] == 0].index
        df.drop(indexNames, inplace=True)

        # Delete indexes for which FLAGS = False
        indexNames = df[df['FLAGS'] == False].index
        df.drop(indexNames, inplace=True)

        # Calculate GLINT for DF
        print('Calculating GLINT column...')
        df = self.get_glint(df)

        # Delete rows where GLINT < 25
        # indexNames = df[df['GLINT'] < 25].index
        # df.drop(indexNames, inplace=True)

        # Get names of indexes for which T865 (Aerosol optical depth) is thicker than 0.6
        indexNames = df[df['T865:float'] >= 0.6].index
        # Delete these row indexes from dataFrame
        df.drop(indexNames, inplace=True)

        ################################
        # FILTER NEGATIVE REFLECTANCES #
        ################################
        df.loc[df['Oa01_reflectance:float'] < 0, 'Oa01_reflectance:float'] = np.nan
        df.loc[df['Oa02_reflectance:float'] < 0, 'Oa02_reflectance:float'] = np.nan
        df.loc[df['Oa03_reflectance:float'] < 0, 'Oa03_reflectance:float'] = np.nan
        df.loc[df['Oa04_reflectance:float'] < 0, 'Oa04_reflectance:float'] = np.nan
        df.loc[df['Oa05_reflectance:float'] < 0, 'Oa05_reflectance:float'] = np.nan
        df.loc[df['Oa06_reflectance:float'] < 0, 'Oa06_reflectance:float'] = np.nan
        df.loc[df['Oa07_reflectance:float'] < 0, 'Oa07_reflectance:float'] = np.nan
        df.loc[df['Oa08_reflectance:float'] < 0, 'Oa08_reflectance:float'] = np.nan
        df.loc[df['Oa09_reflectance:float'] < 0, 'Oa09_reflectance:float'] = np.nan
        df.loc[df['Oa10_reflectance:float'] < 0, 'Oa10_reflectance:float'] = np.nan
        df.loc[df['Oa11_reflectance:float'] < 0, 'Oa11_reflectance:float'] = np.nan
        df.loc[df['Oa12_reflectance:float'] < 0, 'Oa12_reflectance:float'] = np.nan
        df.loc[df['Oa16_reflectance:float'] < 0, 'Oa16_reflectance:float'] = np.nan
        df.loc[df['Oa17_reflectance:float'] < 0, 'Oa17_reflectance:float'] = np.nan
        df.loc[df['Oa18_reflectance:float'] < 0, 'Oa18_reflectance:float'] = np.nan
        df.loc[df['Oa21_reflectance:float'] < 0, 'Oa21_reflectance:float'] = np.nan

        ###############################
        # DROP EVERY NAN REFLECTANCES #
        ###############################
        df = df.dropna()

        # Oa16 must always be above Oa12, for Oa12 is an atmospheric attenuation window
        df = df[df['Oa16_reflectance:float'] > df['Oa12_reflectance:float']]

        # Oa11 must always be higher than Oa12
        df = df[df['Oa11_reflectance:float'] > df['Oa12_reflectance:float']]

        ############################
        # CLASS TEST FOR T865/A865 #
        ############################

        # lowelim = 0.4
        # upperlim = 0.6
        #
        # indexNames = df[df['T865:float'] < lowelim].index
        # df.drop(indexNames, inplace=True)
        # indexNames = df[df['T865:float'] > upperlim].index
        # df.drop(indexNames, inplace=True)
        
        ################################################
        # DROP OUTSIDE 25% OF THE MEDIAN FOR T865/A865 #
        ################################################

        # Get values 25% below or above the median for T865
        # T865_median = np.nanmedian(df['T865:float'], axis=0)
        # T865_upper_lim = T865_median + (0.25 * T865_median)
        # T865_lower_lim = T865_median - (0.25 * T865_median)

        # # Get values 25% below or above the median for A865
        # A865_median = np.nanmedian(df['A865:float'], axis=0)
        # A865_upper_lim = A865_median + (0.25 * A865_median)
        # A865_lower_lim = A865_median - (0.25 * A865_median)

        # Drop data outside bounds for T865:
        # indexNames = df[df['T865:float'] > T865_upper_lim].index
        # df.drop(indexNames, inplace=True)
        # indexNames = df[df['T865:float'] < T865_lower_lim].index
        # df.drop(indexNames, inplace=True)

        # # Drop data outside bounds for A865:
        # indexNames = df[df['A865:float'] > A865_upper_lim].index
        # df.drop(indexNames, inplace=True)
        # indexNames = df[df['A865:float'] < A865_lower_lim].index
        # df.drop(indexNames, inplace=True)

        # Fix the indexing of the dataframe
        df.reset_index(drop=True, inplace=True)

        # Save V2
        if savepath:
            full_saving_path = os.path.join(savepath, os.path.basename(csv_path))
            if len(df) > 0:
                print(f'Saving dataset: {full_saving_path}')
                df.to_csv(full_saving_path)
                return full_saving_path, df
            else:
                print(f'Skipping empty dataset: {os.path.basename(csv_path)}')
                return full_saving_path, df

        else:
            return 'unsaved', df

    @staticmethod
    def get_mean_and_clean(image_path, threshold=None):
        # read text file and convert to pandas dataframe
        df = pd.read_csv(image_path, sep='\t', skiprows=1)
        # # Columns to keep
        keep = ['Oa01_reflectance:float',
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
                'Oa21_reflectance:float',
                'latitude:double',
                'longitude:double']
        # Drop columns not present in the list
        df = df.filter(keep)
        # Get names of indexes for which column LON has value 0
        indexNames = df[df['longitude:double'] == 0].index
        # Delete these row indexes from dataFrame
        df.drop(indexNames, inplace=True)
        # Assuming the reflectance of water pixels should not be above 0.16 (Oa08:665nm), we will drop using this threshold
        if threshold:
            indexNames = df[df['Oa08_reflectance:float'] > threshold].index
            # Delete these row indexes from dataFrame
            df.drop(indexNames, inplace=True)
        # drop lon/lat columns
        df = df.drop(['latitude:double', 'longitude:double'], axis=1)
        return df.mean(skipna=True)

    @staticmethod
    def kde_local_maxima(x):

        kernel = stats.gaussian_kde(dataset=x, bw_method='silverman')

        kde_res = kernel(x)

        xs, ys = zip(*sorted(zip(x, kde_res)))

        xray = np.array(xs)
        yray = np.array(ys)

        ma = argrelextrema(yray, np.greater)[0]
        peak_position = list(ma)
        return peak_position, xray, yray, kde_res

    @staticmethod
    def get_mean_and_clean_v2(image_path):
        # read text file and convert to pandas dataframe
        df = pd.read_csv(image_path)

        # Columns to keep
        keep = ['Oa01_reflectance:float',
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
                'Oa21_reflectance:float',
                'latitude:double',
                'longitude:double',
                'OAA:float',
                'OZA:float',
                'SAA:float',
                'SZA:float',
                'A865:float',
                'T865:float',
                'GLINT',
                'PCTVLDPX']

        # Drop columns not present in the list
        df = df.filter(keep)

        # get the std deviation of the specific column
        glintstd = df.loc[:, 'GLINT'].std(skipna=True)

        # get the % of valid pixels in DF
        validpx = df['PCTVLDPX'][0]
        pctvalidpx = (len(df) * 100) / validpx

        # drop lon/lat columns
        df = df.drop(['latitude:double', 'longitude:double', 'PCTVLDPX'], axis=1)

        result_dict = {}
        for colname in df:
            result_dict[colname] = df[colname].mean(skipna=True)

        result_dict['AbsVldPx'] = validpx
        result_dict['VldPx.pct'] = pctvalidpx
        result_dict['GLINT.std'] = glintstd

        # https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-3-olci/level-2/aerosol-optical-thickness
        t865_desc = df.loc[:, 'T865:float'].describe()

        result_dict['T865.count'], \
        result_dict['T865.mean'], \
        result_dict['T865.std'], \
        result_dict['T865.min'], \
        result_dict['T865.25%ile'], \
        result_dict['T865.50%ile'], \
        result_dict['T865.75%ile'], \
        result_dict['T865.max'] = list(t865_desc)

        # https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-3-olci/level-2/aerosol-angstrom-exponent
        a865_desc = df.loc[:, 'A865:float'].describe()

        result_dict['A865.count'], \
        result_dict['A865.mean'], \
        result_dict['A865.std'], \
        result_dict['A865.min'], \
        result_dict['A865.25%ile'], \
        result_dict['A865.50%ile'], \
        result_dict['A865.75%ile'], \
        result_dict['A865.max'] = list(a865_desc)

        return result_dict

    @staticmethod
    def netcdf_kml_slicer(gpt_path, kml_path, input_imgs_folder, output_folder):
        """
        This function takes in a folder containing several Sentienl-3 L2 WFR images (input_imgs_folder) and makes subsets
        for each one of them by using a input KML file (kml_path) that can be generated at google earth engine.
        To make it work the user also needs to input where in the system it is installed
        the GPT tool that comes along with ESA-SNAP (gpt_path).
        (output_folder) is where do you want to save the extracted images.
        By the end of the execution, the function returns a list with each CSV extracted this way.

        gpt_path: where in your system is the SNAP-gpt tool.
            unix: '/d_drive_data/snap/bin/gpt'
            wind: 'C:\Program Files\snap\bin\gpt.exe'

        kml_path: where is the Google Earth KML delimited region to be used.

        input_imgs_folder: where is your sentinel 3 images. expects something like:
            unix: '/d_drive_data/L2_WFR'
            wind: 'E:\S3\L2_WFR'

        output_folder: where to save the processed files. Expects something like:
            unix: '/d_drive_data/processing/MANACAPURU/'
            wind: 'D:\\processing\\win\\COARI'

        """
        t1 = time.perf_counter()
        work_dir = input_imgs_folder

        logging.info(f'Importing GPTBridge...\n')
        gpt = out.GPTBridge(gpt_sys_path=gpt_path, output_path=output_folder, kml_path=kml_path)

        # adding every image inside the working directory to a list
        field_files = os.listdir(work_dir)

        # adding the complete path to each image folder listed
        field_files = [os.path.join(work_dir, image) for image in field_files]

        total = len(field_files)
        logging.info(f'Total files identified in folder: {total}\n')

        # Asynchronous-processing: (DISCLAIMER: this can be a REALLY INTENSIVE computation, be careful.)
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            try:
                result = list(executor.map(gpt.get_pixels_by_kml, field_files))
            except concurrent.futures.process.BrokenProcessPool as ex:
                logging.error(f"{ex} This might be caused by limited system resources. "
                              f"Try increasing system memory or disable concurrent processing. ")

        # # Synchronous-processing: (DEPRECATED)
        # for n, image in enumerate(field_files):
        #
        #     file_name = os.path.join(work_dir, image)
        #     print(f'Processing image {n} of {total}:\n{image}\n')
        #     gpt.get_pixels_by_kml(file_name)

        # s3frbr_output_files = os.listdir(work_dir)
        # sorted_s3frbr_output_files = sorted(os.listdir(work_dir), key=lambda s: s[16:31])

        t2 = time.perf_counter()
        logging.info(f'>>> Finished in {round(t2 - t1, 2)} second(s). <<<')

        final_result = [os.path.join(output_folder, subset) for subset in result]
        return final_result

    @staticmethod
    def build_list_from_subset(work_dir):
        """
        Creates a python list containing the accumulated data from all the extracted areas by the kml file.
        """
        sorted_s3frbr_output_files = sorted(os.listdir(work_dir), key=lambda s: s[16:31])

        return sorted_s3frbr_output_files

    def generate_time_series_data(self, work_dir, sorted_list):

        Oa01_reflectance_tms = []
        Oa02_reflectance_tms = []
        Oa03_reflectance_tms = []
        Oa04_reflectance_tms = []
        Oa05_reflectance_tms = []
        Oa06_reflectance_tms = []
        Oa07_reflectance_tms = []
        Oa08_reflectance_tms = []
        Oa09_reflectance_tms = []
        Oa10_reflectance_tms = []
        Oa11_reflectance_tms = []
        Oa12_reflectance_tms = []
        Oa16_reflectance_tms = []
        Oa17_reflectance_tms = []
        Oa18_reflectance_tms = []
        Oa21_reflectance_tms = []
        OAA_tms = []
        OZA_tms = []
        SAA_tms = []
        SZA_tms = []
        A865_tms = []
        T865_tms = []
        T865std_tms = []
        datetime_labels = []
        string_labels = []

        total = len(sorted_list)

        for n, image in enumerate(sorted_list):
            logging.info(f'Extracting image {n + 1}/{total} - {image[:31]}...')
            file_name = os.path.join(work_dir, image)
            dtlbl = datetime.strptime(image[16:31], '%Y%m%dT%H%M%S')
            strlbl = image[16:31]
            Oa01, Oa02, Oa03, Oa04, Oa05, Oa06, Oa07, Oa08, Oa09, Oa10, Oa11, Oa12, Oa16, Oa17, Oa18, Oa21, OAA, OZA, SAA, SZA, A865, T865, T865std = list(
                self.get_mean_and_clean_v2(file_name))

            Oa01_reflectance_tms.append(Oa01)
            Oa02_reflectance_tms.append(Oa02)
            Oa03_reflectance_tms.append(Oa03)
            Oa04_reflectance_tms.append(Oa04)
            Oa05_reflectance_tms.append(Oa05)
            Oa06_reflectance_tms.append(Oa06)
            Oa07_reflectance_tms.append(Oa07)
            Oa08_reflectance_tms.append(Oa08)
            Oa09_reflectance_tms.append(Oa09)
            Oa10_reflectance_tms.append(Oa10)
            Oa11_reflectance_tms.append(Oa11)
            Oa12_reflectance_tms.append(Oa12)
            Oa16_reflectance_tms.append(Oa16)
            Oa17_reflectance_tms.append(Oa17)
            Oa18_reflectance_tms.append(Oa18)
            Oa21_reflectance_tms.append(Oa21)

            OAA_tms.append(OAA)
            OZA_tms.append(OZA)
            SAA_tms.append(SAA)
            SZA_tms.append(SZA)
            A865_tms.append(A865)
            T865_tms.append(T865)
            T865std_tms.append(T865std)

            datetime_labels.append(dtlbl)
            string_labels.append(strlbl)

            d = {'Datetime': datetime_labels,
                 'Date-String': string_labels,
                 'B1-400': Oa01_reflectance_tms,
                 'B2-412.5': Oa02_reflectance_tms,
                 'B3-442.5': Oa03_reflectance_tms,
                 'B4-490': Oa04_reflectance_tms,
                 'B5-510': Oa05_reflectance_tms,
                 'B6-560': Oa06_reflectance_tms,
                 'B7-620': Oa07_reflectance_tms,
                 'B8-665': Oa08_reflectance_tms,
                 'B9-673.75': Oa09_reflectance_tms,
                 'B10-681.25': Oa10_reflectance_tms,
                 'B11-708.75': Oa11_reflectance_tms,
                 'B12-753.75': Oa12_reflectance_tms,
                 'B16-778.75': Oa16_reflectance_tms,
                 'B17-865': Oa17_reflectance_tms,
                 'B18-885': Oa18_reflectance_tms,
                 'B21-1020': Oa21_reflectance_tms,
                 'OAA': OAA_tms,
                 'OZA': OZA_tms,
                 'SAA': SAA_tms,
                 'SZA': SZA_tms,
                 'A865': A865_tms,
                 'T865': T865_tms,
                 'T865std': T865std_tms,
                 'filename': sorted_list}

        return d

    def generate_time_series_datav2(self, work_dir, sorted_list):

        Oa01_reflectance_tms = []
        Oa02_reflectance_tms = []
        Oa03_reflectance_tms = []
        Oa04_reflectance_tms = []
        Oa05_reflectance_tms = []
        Oa06_reflectance_tms = []
        Oa07_reflectance_tms = []
        Oa08_reflectance_tms = []
        Oa09_reflectance_tms = []
        Oa10_reflectance_tms = []
        Oa11_reflectance_tms = []
        Oa12_reflectance_tms = []
        Oa16_reflectance_tms = []
        Oa17_reflectance_tms = []
        Oa18_reflectance_tms = []
        Oa21_reflectance_tms = []

        OAA_tms = []
        OZA_tms = []
        SAA_tms = []
        SZA_tms = []

        A865_tms = []
        A865std_tms = []
        A865min_tms = []
        A865max_tms = []
        A865tile25_tms = []
        A865tile50_tms = []
        A865tile75_tms = []

        T865_tms = []
        T865std_tms = []
        T865min_tms = []
        T865max_tms = []
        T865tile25_tms = []
        T865tile50_tms = []
        T865tile75_tms = []

        glint_tms = []
        glintstd_tms = []
        absvldpx_tms = []
        pctvlddpx_tms = []
        datetime_labels = []
        string_labels = []

        total = len(sorted_list)

        for n, image in enumerate(sorted_list):
            print(f'Extracting image {n + 1}/{total} - {image[:31]}...')
            file_name = os.path.join(work_dir, image)
            dtlbl = datetime.strptime(image[16:31], '%Y%m%dT%H%M%S')
            strlbl = image[16:31]

            means_dict = self.get_mean_and_clean_v2(file_name)

            Oa01_reflectance_tms.append(means_dict['Oa01_reflectance:float'])
            Oa02_reflectance_tms.append(means_dict['Oa02_reflectance:float'])
            Oa03_reflectance_tms.append(means_dict['Oa03_reflectance:float'])
            Oa04_reflectance_tms.append(means_dict['Oa04_reflectance:float'])
            Oa05_reflectance_tms.append(means_dict['Oa05_reflectance:float'])
            Oa06_reflectance_tms.append(means_dict['Oa06_reflectance:float'])
            Oa07_reflectance_tms.append(means_dict['Oa07_reflectance:float'])
            Oa08_reflectance_tms.append(means_dict['Oa08_reflectance:float'])
            Oa09_reflectance_tms.append(means_dict['Oa09_reflectance:float'])
            Oa10_reflectance_tms.append(means_dict['Oa10_reflectance:float'])
            Oa11_reflectance_tms.append(means_dict['Oa11_reflectance:float'])
            Oa12_reflectance_tms.append(means_dict['Oa12_reflectance:float'])
            Oa16_reflectance_tms.append(means_dict['Oa16_reflectance:float'])
            Oa17_reflectance_tms.append(means_dict['Oa17_reflectance:float'])
            Oa18_reflectance_tms.append(means_dict['Oa18_reflectance:float'])
            Oa21_reflectance_tms.append(means_dict['Oa21_reflectance:float'])

            OAA_tms.append(means_dict['OAA:float'])
            OZA_tms.append(means_dict['OZA:float'])
            SAA_tms.append(means_dict['SAA:float'])
            SZA_tms.append(means_dict['SZA:float'])

            A865_tms.append(means_dict['A865.mean'])
            A865std_tms.append(means_dict['A865.std'])
            A865min_tms.append(means_dict['A865.min'])
            A865max_tms.append(means_dict['A865.max'])
            A865tile25_tms.append(means_dict['A865.25%ile'])
            A865tile50_tms.append(means_dict['A865.50%ile'])
            A865tile75_tms.append(means_dict['A865.75%ile'])

            T865_tms.append(means_dict['T865.mean'])
            T865std_tms.append(means_dict['T865.std'])
            T865min_tms.append(means_dict['T865.min'])
            T865max_tms.append(means_dict['T865.max'])
            T865tile25_tms.append(means_dict['T865.25%ile'])
            T865tile50_tms.append(means_dict['T865.50%ile'])
            T865tile75_tms.append(means_dict['T865.75%ile'])

            glint_tms.append(means_dict['GLINT'])
            glintstd_tms.append(means_dict['GLINT.std'])
            absvldpx_tms.append(means_dict['AbsVldPx'])
            pctvlddpx_tms.append(means_dict['VldPx.pct'])
            datetime_labels.append(dtlbl)
            string_labels.append(strlbl)

            d = {'Datetime': datetime_labels,
                 'Date-String': string_labels,
                 'B1-400': Oa01_reflectance_tms,
                 'B2-412.5': Oa02_reflectance_tms,
                 'B3-442.5': Oa03_reflectance_tms,
                 'B4-490': Oa04_reflectance_tms,
                 'B5-510': Oa05_reflectance_tms,
                 'B6-560': Oa06_reflectance_tms,
                 'B7-620': Oa07_reflectance_tms,
                 'B8-665': Oa08_reflectance_tms,
                 'B9-673.75': Oa09_reflectance_tms,
                 'B10-681.25': Oa10_reflectance_tms,
                 'B11-708.75': Oa11_reflectance_tms,
                 'B12-753.75': Oa12_reflectance_tms,
                 'B16-778.75': Oa16_reflectance_tms,
                 'B17-865': Oa17_reflectance_tms,
                 'B18-885': Oa18_reflectance_tms,
                 'B21-1020': Oa21_reflectance_tms,
                 'OAA': OAA_tms,
                 'OZA': OZA_tms,
                 'SAA': SAA_tms,
                 'SZA': SZA_tms,

                 'A865': A865_tms,
                 'A865.std': A865std_tms,
                 'A865.min': A865min_tms,
                 'A865.max': A865max_tms,
                 'A865.25%tile': A865tile25_tms,
                 'A865.50%tile': A865tile50_tms,
                 'A865.75%tile': A865tile75_tms,

                 'T865': T865_tms,
                 'T865.std': T865std_tms,
                 'T865.min': T865min_tms,
                 'T865.max': T865max_tms,
                 'T865.25%tile': T865tile25_tms,
                 'T865.50%tile': T865tile50_tms,
                 'T865.75%tile': T865tile75_tms,

                 'meanGlint': glint_tms,
                 'Glintstd': glintstd_tms,
                 'abs-vld-px': absvldpx_tms,
                 'pct-vld-px': pctvlddpx_tms,
                 'filename': sorted_list}

        return d

    def s3l2_custom_reflectance_plot(self, df, figure_title, save_title=None):

        colnms = ['T865:float',
                  'Oa01_reflectance:float',
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

        # create a list with the value in (nm) of the 16 Sentinel-3 bands for L2 products.
        s3_bands_tick = list(self.exp.s3_bands_l2.values())

        # create a list with the name of the 16 Sentinel-3 bands for L2 products.
        s3_bands_tick_label = list(self.exp.s3_bands_l2.keys())

        plt.rcParams['figure.figsize'] = [12, 6]

        fig = plt.figure()
        fig.show()
        ax1 = fig.add_subplot(111)

        ax1.set_xlabel('Wavelenght (nm)')
        ax1.set_ylabel('Reflectance')
        ax1.set_title(figure_title, y=1, fontsize=16)

        # creating color scale based on T865
        lst = df['T865:float']
        minima = min(lst)
        maxima = max(lst)
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

        for _, row in df[colnms].iterrows():
            # ax1.plot(s3_bands_tick, list(row))
            t865c = mapper.to_rgba(row[0])
            ax1.plot(s3_bands_tick, list(row[1:]), alpha=0.4, c=t865c)

        ax1.axhline(y=0, xmin=0, xmax=1, linewidth=0.5, color='black', linestyle='--')
        ax1.set_xticks(s3_bands_tick)
        ax1.set_xticklabels(s3_bands_tick)
        ax1.tick_params(labelrotation=90, labelsize='small')

        ax2 = ax1.twiny()
        ax2.plot(s3_bands_tick, [0] * (len(s3_bands_tick)), alpha=0.0)
        ax2.set_xticks(s3_bands_tick)
        ax2.set_xticklabels(s3_bands_tick_label)
        ax2.tick_params(labelrotation=90, labelsize='xx-small')
        ax2.set_title('Sentinel-3 Oa Bands', y=0.93, x=0.12, fontsize='xx-small')

        if save_title:
            plt.savefig(save_title, dpi=self.imgdpi)
        else:
            plt.show()

    def plot_kde_hist(self, title, xray, yray, x, kde_res, pk, svpath_n_title=None):
        plt.rcParams['figure.figsize'] = [16, 6]
        fig = plt.figure()
        ax = plt.axes()
        ax.set_title(title, fontsize=16)

        ax.plot(xray, yray, color='k', label='Fitted KDE', zorder=11)
        ax.plot(xray[pk], yray[pk], 'or', zorder=11, label='KDE Local Maxima')
        ax.hist(x, 100, color='lightblue', label='Histogram')
        ax.scatter(x, kde_res, zorder=10, marker='x', label='Observations')

        ax.set_xlabel('Reflectance - Oa08:665nm', fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)

        # Get the mean
        ax.axvline(x.mean(), color='g', label='Mean')
        # Get the std. dev.
        ax.axvline(x=np.mean(x) - np.std(x), ls="--", color='g', alpha=0.7, label='Std.Deviation')
        ax.axvline(x=np.mean(x) + np.std(x), ls="--", color='g', alpha=0.7)


        ax.legend()

        for m in xray[pk]:
            ax.axvline(m, color='r')
        if svpath_n_title:
            plt.savefig(svpath_n_title, dpi=self.imgdpi)
            plt.close(fig)

        if not svpath_n_title:
            plt.show()

    def plot_single_sktr(self, xdata, ydata, xlabel, ylabel, color, clabel, title, savepathname):
        plt.rcParams['figure.figsize'] = [9.4, 8]
        fig = plt.figure()
        ax = plt.axes()
        ax.set_title(title)

        img = ax.scatter(xdata, ydata, s=3, c=color)
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label(clabel)

        ax.plot([-1, 1], [-1, 1], 'k-', linewidth=1)
        ax.plot([0, 0], [-1, 1], c='grey', linestyle='dashed', linewidth=1)
        ax.plot([-1, 1], [0, 0], c='grey', linestyle='dashed', linewidth=1)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # TODO: add label in the colorbar

        ax.set_xlim(-0.02, 0.2)
        ax.set_ylim(-0.02, 0.2)
        plt.text(0.160, 0.003, '% Reflectance')

        plt.savefig(savepathname, dpi=self.imgdpi)

        plt.close(fig)

    # GENERATES COMPARATIVE SCATTERPLOTS
    def plot_overlap_sktr(self, x1_data, y1_data, x2_data, y2_data, x_lbl, y_lbl, c1_data, c1_lbl, c2_data, c2_lbl, title,
                          savepathname):
        plt.rcParams['figure.figsize'] = [12, 8]
        fig = plt.figure()
        ax = plt.axes()
        ax.set_title(title)

        img = ax.scatter(x2_data, y2_data, s=5, c=c2_data, cmap='winter_r')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label(c2_lbl)

        img = ax.scatter(x1_data, y1_data, s=5, c=c1_data, cmap='autumn_r')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label(c1_lbl)

        ax.plot([-1, 1], [-1, 1], 'k-', linewidth=1)
        ax.plot([0, 0], [-1, 1], c='grey', linestyle='dashed', linewidth=1)
        ax.plot([-1, 1], [0, 0], c='grey', linestyle='dashed', linewidth=1)

        ax.set_xlabel(x_lbl)  # RED: Oa08 (865nm)
        ax.set_ylabel(y_lbl)  # NIR: Oa17 (665nm)

        ax.set_xlim(-0.02, 0.2)
        ax.set_ylim(-0.02, 0.2)
        plt.text(0.160, 0.003, '% Reflectance')

        plt.savefig(savepathname, dpi=self.imgdpi)

        plt.close(fig)

    # GENERATES COMPARATIVE SCATTERPLOTS
    def plot_overlap_sktr(self, x1_data, y1_data, x2_data, y2_data, x_lbl, y_lbl, c1_data, c1_lbl, c2_data, c2_lbl, title,
                          savepathname):
        plt.rcParams['figure.figsize'] = [12, 8]
        fig = plt.figure()
        ax = plt.axes()
        ax.set_title(title)

        img = ax.scatter(x2_data, y2_data, s=5, c=c2_data, cmap='winter_r')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label(c2_lbl)

        img = ax.scatter(x1_data, y1_data, s=5, c=c1_data, cmap='autumn_r')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label(c1_lbl)

        ax.plot([-1, 1], [-1, 1], 'k-', linewidth=1)
        ax.plot([0, 0], [-1, 1], c='grey', linestyle='dashed', linewidth=1)
        ax.plot([-1, 1], [0, 0], c='grey', linestyle='dashed', linewidth=1)

        ax.set_xlabel(x_lbl)  # RED: Oa08 (865nm)
        ax.set_ylabel(y_lbl)  # NIR: Oa17 (665nm)

        ax.set_xlim(-0.02, 0.2)
        ax.set_ylim(-0.02, 0.2)
        plt.text(0.160, 0.003, '% Reflectance')

        plt.savefig(savepathname, dpi=self.imgdpi)

        plt.close(fig)

    # GENERATES COMPARATIVE SCATTERPLOTS
    def plot_sidebyside_sktr(self, x1_data, y1_data, x2_data, y2_data, x_lbl, y_lbl, c1_data, c1_lbl, c2_data, c2_lbl, title,
                             savepathname=None):

        plt.rcParams['figure.figsize'] = [14, 5.2]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(title)

        skt1 = ax1.scatter(x1_data, y1_data, s=3, c=c1_data, cmap='viridis')
        cbar = fig.colorbar(skt1, ax=ax1)
        cbar.set_label(c1_lbl)

        skt2 = ax2.scatter(x2_data, y2_data, s=3, c=c2_data, cmap='viridis')
        cbar = fig.colorbar(skt2, ax=ax2)
        cbar.set_label(c2_lbl)

        ax1.plot([-0.02, 0.2], [-0.02, 0.2], 'k-', linewidth=1)
        ax1.plot([0.01, 0.01], [-0.02, 0.2], c='red', linestyle='dashed', linewidth=1)
        ax1.plot([-0.02, 0.2], [0.01, 0.01], c='red', linestyle='dashed', linewidth=1)
        ax1.plot([0, 0], [-0.02, 0.2], c='grey', linestyle='dashed', linewidth=1)
        ax1.plot([-0.02, 0.2], [0, 0], c='grey', linestyle='dashed', linewidth=1)

        ax2.plot([-0.02, 0.2], [-0.02, 0.2], 'k-', linewidth=1)
        ax2.plot([0.01, 0.01], [-0.02, 0.2], c='red', linestyle='dashed', linewidth=1)
        ax2.plot([-0.02, 0.2], [0.01, 0.01], c='red', linestyle='dashed', linewidth=1)
        ax2.plot([0, 0], [-0.02, 0.2], c='grey', linestyle='dashed', linewidth=1)
        ax2.plot([-0.02, 0.2], [0, 0], c='grey', linestyle='dashed', linewidth=1)

        ax1.set_xlabel(x_lbl)  # RED: Oa08 (865nm)
        ax1.set_ylabel(y_lbl)  # NIR: Oa17 (665nm)
        ax2.set_xlabel(x_lbl)

        ax1.set_xlim(-0.02, 0.2)
        ax1.set_ylim(-0.02, 0.2)

        ax2.set_xlim(-0.02, 0.2)
        ax2.set_ylim(-0.02, 0.2)

        if savepathname:
            plt.savefig(savepathname, dpi=self.imgdpi)
            plt.close(fig)

        if not savepathname:
            plt.show()

    def plot_time_series(self, tms_dict, tms_key, fig_title, save_title=None):
        plt.rcParams['figure.figsize'] = [16, 6]
        # fig = plt.figure()
        ax = plt.axes()
        ax.set_title(fig_title, fontsize=16)
        ax.plot(tms_dict['Datetime'], tms_dict[tms_key], marker='o', markersize=5, label=self.bname_dict[tms_key])
        ax.set_xlabel('Date', fontsize=16)
        ax.set_ylabel('Reflectance', fontsize=16)
        ax.legend()
        if save_title:
            plt.savefig(save_title, dpi=self.imgdpi)
        plt.show()

    def plot_multiple_time_series(self, tms_dict, tms_keys, fig_title, save_title=None):

        plt.rcParams['figure.figsize'] = [16, 6]
        fig = plt.figure()
        ax = plt.axes()
        ax.set_title(fig_title, fontsize=16)
        for element in tms_keys:
            ax.plot(tms_dict['Datetime'], tms_dict[element], marker='o', markersize=5, label=self.bname_dict[element])
        ax.set_xlabel('Date', fontsize=16)
        ax.set_ylabel('Reflectance', fontsize=16)
        ax.legend()
        if save_title:
            plt.savefig(save_title, dpi=self.imgdpi)
        plt.show()

    def plot_ts_from_csv(self, csv_path, tms_key, fig_title, save_title=None):
        tms_dict = pd.read_csv(csv_path, parse_dates=['Datetime'])
        self.plot_time_series(tms_dict, tms_key, fig_title, save_title)

    def plot_multi_ts_from_csv(self, csv_path, tms_keys, fig_title, save_title=None):
        tms_dict = pd.read_csv(csv_path, parse_dates=['Datetime'])
        self.plot_multiple_time_series(tms_dict, tms_keys, fig_title, save_title)

    @staticmethod
    def save_tms_to_csv(tms_dicst, csv_file_name):
        logging.info(f'Saving time-series DataFrame @ {csv_file_name}')
        df = pd.DataFrame(data=tms_dicst)
        df.to_csv(csv_file_name)
        logging.info(f'Done.')


if __name__ == '__main__':
    logging.info(f'Arguments received:\n{sys.argv}\n')

    tsgen = TsGenerator()

    # DEPRECATED ?
    # if len(sys.argv) > 2:
    #     logging.info(f'Running Time-series generator.\n'
    #                  f'SNAP-GPT folder: {sys.argv[1]}\n'
    #                  f'using KML file: {sys.argv[2]}\n'
    #                  f'Input folder: {sys.argv[3]}\n'
    #                  f'Output folder: {sys.argv[4]}\n')
    #
    #     extracted_list = tsgen.netcdf_kml_slicer(gpt_path=sys.argv[1],
    #                                              kml_path=sys.argv[2],
    #                                              input_imgs_folder=sys.argv[3],
    #                                              output_folder=sys.argv[4])
    # else:
    #     logging.info('Execution is over: Insufficient number of arguments.')

    sorted_files = tsgen.build_list_from_subset(sys.argv[1])

    ts_dict = tsgen.generate_time_series_data(sys.argv[1], sorted_files)

    output_file = sys.argv[2]

    tsgen.save_tms_to_csv(ts_dict, output_file)

    # =========================================================================
    # PLOTTING STUFF
    area = sys.argv[1].split('\\')[-1]  # "D:\processing\win\COARI" -> 'COARI'
    save_file = sys.argv[3]

    # tsgen.plot_ts_from_csv(csv_path=output_file,
    #                        tms_key='B8-665',
    #                        fig_title='{area}: S3-WFR Oa08 Reflectance (665nm) time-series from 2019-03-09 to 2020-03-31')

    tsgen.plot_multi_ts_from_csv(csv_path=output_file,
                                 tms_keys=['B8-665', 'B17-865'],
                                 fig_title=f'{area}: S3-WFR Reflectance time-series from 2019-03-09 to 2020-03-31',
                                 save_title=save_file)

    # python tsgen.py "D:\processing\win\COARI" "D:\processing\win\coari-ts.csv" "D:\processing\win\coari-ts-plot.png"
