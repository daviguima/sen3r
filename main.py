import os
import time
import sen3r
import logging
import argparse

from sen3r.sen3r import Core
from sen3r.commons import Utils
from datetime import datetime


def main():
    """
    Entry point for the SEN3R package. Call sen3r -h or --help to see further options.
    """
    parser = argparse.ArgumentParser(
        description='SEN3R (Sentinel-3 Reflectance Retrieval over Rivers) '
                    'enables extraction of reflectance time series from Sentinel-3 L2 WFR images over water bodies.')
    parser.add_argument("-i", "--input", help="The products input folder. Required.", type=str)
    parser.add_argument("-o", "--out", help="Output directory. Required.", type=str)
    parser.add_argument("-r", "--roi", help="Region of interest (SHP, KML or GeoJSON). Required", type=str)
    parser.add_argument("-p", "--product", help='Currently only WFR is available.', default='WFR', type=str)
    parser.add_argument("-c", "--cams", help="Path to search for auxiliary CAMS file. Optional.", type=str)
    parser.add_argument('-ng', '--no-graphics', help='Do not generate graphical reports.', action='store_true')
    parser.add_argument('-np', '--no-pdf', help='Do not generate PDF report.', action='store_true')
    parser.add_argument("-s", "--single",
                        help="Single mode: run SEN3R over only one image instead of a whole directory."
                             " Optional.", action='store_true')
    parser.add_argument('-v', '--version', help='Displays current package version.', action='store_true')

    # ,--------------------------------------,
    # | STORE INPUT VARS INSIDE SEN3R OBJECT |--------------------------------------------------------------------------
    # '--------------------------------------'
    args = parser.parse_args().__dict__  # Converts the input arguments from Namespace() to dict

    if args['version']:
        print(f'SEN3R version: {sen3r.__version__}')

    elif (args['input'] is None) or (args['out'] is None) or (args['roi'] is None):
        print('Please specify required INPUT/OUTPUT folders and REGION of interest (-i, -o, -r)')

    else:
        # ,------------,
        # | LOG SETUP  |------------------------------------------------------------------------------------------------
        # '------------'
        # args['logfile'] = os.path.join(args['out'], 'sen3r_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'.log')
        # args['logger'] = Utils.create_log_handler(args['logfile'])
        s3r = Core(args)  # Declare a SEN3R Core Object
        print(f'Starting SEN3R - LOG operations saved at:{s3r.arguments["logfile"]}')
        s3r.log.info(f'Starting SEN3R {s3r.VERSION} ({sen3r.__version__})')
        s3r.log.info('------')
        s3r.log.info('Input arguments:')
        for key in args:
            s3r.log.info(f'{key}: {args[key]}')
        s3r.log.info('------')

        if args['single']:  # Single mode
            band_data, img_data, doneList = s3r.build_single_file()

        else:  # Default mode: several images
            doneList = s3r.build_intermediary_files()
            print('cams_args:', s3r.arguments['cams'])
            if s3r.arguments["cams"]:
                s3r.process_csv_list(raw_csv_list=doneList, use_cams=True)
            else:
                s3r.process_csv_list(raw_csv_list=doneList)

    pass


if __name__ == '__main__':
    # ,--------------,
    # | Start timers |--------------------------------------------------------------------------------------------------
    # '--------------'
    Utils.tic()
    t1 = time.perf_counter()
    # ,-----,
    # | RUN |-----------------------------------------------------------------------------------------------------------
    # '-----'
    main()
    # ,------------------------------,
    # | End timers and report to log |----------------------------------------------------------------------------------
    # '------------------------------'
    t_hour, t_min, t_sec = Utils.tac()
    t2 = time.perf_counter()
    outputstr = f'Finished in {round(t2 - t1, 2)} second(s).'
    final_message = f'Elapsed execution time: {t_hour}h : {t_min}m : {t_sec}s'
    print(outputstr)
    print(final_message)
