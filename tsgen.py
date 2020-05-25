import os
import sys
import logging
import time
import pandas as pd
import outsourcing as out
import matplotlib.pyplot as plt
import concurrent.futures
from datetime import datetime

class TsGenerator():

    def __init__(self):
        # Setting up information logs
        logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

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

    @staticmethod
    def get_mean_and_clean(image_path):
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
        # Assuming the reflectance of water pixels should not be above 0.16, we will drop using this threshold
        indexNames = df[df['Oa08_reflectance:float'] > 0.16].index
        # Delete these row indexes from dataFrame
        df.drop(indexNames, inplace=True)
        # drop lon/lat columns
        df = df.drop(['latitude:double', 'longitude:double'], axis=1)
        return df.mean(skipna=True)

    @staticmethod
    def netcdf_kml_slicer(gpt_path, kml_path, input_imgs_folder, output_folder):
        """
        This function takes in a folder containing several Sentienl-3 L2 WFR images (input_imgs_folder) and makes subsets
        for each one of them by using a input KML file (kml_path) that can be generated at google earth angine.
        To make it work the user also needs to input where in the system it is installed
        the GPT tool that comes along with ESA-SNAP (gpt_path).
        (output_folder) is where do you want to save the extrated images.
        By the end of the executing, the function returns a list with each CSV extracted this way.

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
        Creates a python dictionary containing the accumulated data from all the extracted areas by the kml file.
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
        datetime_labels = []
        string_labels = []

        total = len(sorted_list)

        for n, image in enumerate(sorted_list):
            logging.info(f'Extracting image {n + 1}/{total} - {image[:31]}...')
            file_name = os.path.join(work_dir, image)
            dtlbl = datetime.strptime(image[16:31], '%Y%m%dT%H%M%S')
            strlbl = image[16:31]
            Oa01, Oa02, Oa03, Oa04, Oa05, Oa06, Oa07, Oa08, Oa09, Oa10, Oa11, Oa12, Oa16, Oa17, Oa18, Oa21 = list(
                self.get_mean_and_clean(file_name))

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
                 'filename': sorted_list}

        return d

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
            plt.savefig(save_title, dpi=300)
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
            plt.savefig(save_title, dpi=300)
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
