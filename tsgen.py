import os
import sys
import logging
import time

import subprocess
import pandas as pd
import outsourcing as out
from nc_explorer import NcExplorer
import concurrent.futures

# Setting up information logs
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.DEBUG)


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


def run(gpt_path, kml_path, input_imgs_folder, output_folder):
    """
    gpt_path: where in your system is the SNAP-gpt tool.
        unix: '/d_drive_data/snap/bin/gpt'
        wind: 'C:\Program Files\snap\bin\gpt.exe'

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
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        try:
            results = list(executor.map(gpt.get_pixels_by_kml, field_files))
            return results
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

    logging.info(f'>>> Finished in {round(t2-t1,2)} second(s). <<<')


if __name__ == '__main__':
    logging.info(f'Arguments received:\n{sys.argv}\n')
    if len(sys.argv) > 2:
        logging.info(f'Running Time-series generator.\n'
                     f'SNAP-GPT folder: {sys.argv[1]}\n'
                     f'using KML file: {sys.argv[2]}\n'
                     f'Input folder: {sys.argv[3]}\n'
                     f'Output folder: {sys.argv[4]}\n')

        results = run(gpt_path=sys.argv[1],
                      kml_path=sys.argv[2],
                      input_imgs_folder=sys.argv[3],
                      output_folder=sys.argv[4])
    else:
        logging.info('Execution is over: Insufficient number of arguments.')