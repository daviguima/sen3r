import os
import time
import sen3r
import argparse
from pathlib import Path


def main():
    """
    Entry point for the SEN3R package. Call sen3r -h or --help to see further options.
    """
    parser = argparse.ArgumentParser(
        description='SEN3R (Sentinel-3 Reflectance Retrieval over Rivers) '
                    'enables extraction of reflectance time series from Sentinel-3 L2 WFR images over water bodies.')

    parser.add_argument("-i", "--input", help="The products input folder. Required.", required=True, type=str)
    parser.add_argument("-o", "--out", help="Output directory. Required.", required=True, type=str)
    parser.add_argument("-r", "--roi", help="Region of interest (SHP, KML or GeoJSON). Required", required=True,
                        type=str)
    parser.add_argument("-p", "--product", help='Currently only WFR is available.', default='WFR', type=str)

    parser.add_argument("-c", "--cams", help="Auxiliary CAMS file. Optional.", type=str)
    parser.add_argument("-s", "--single",
                        help="Single mode: run SEN3R over only one image instead of a whole directory."
                             "Optional.", action='store_true')

    parser.add_argument('-v', '--version', help='Displays current package version', action='store_true')
    parser.add_argument('-b', '--verbose', help='Debug basic parameters (paths/files)', action='store_true')

    args = parser.parse_args()

    if args.version:
        print(f'SEN3R version: {sen3r.__version__}')

    type(args)

    pass


if __name__ == '__main__':
    main()
