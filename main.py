import os
import sys
import logging
import argparse
import outsourcing as out


# import nc_explorer


class Engine:
    band_id = {1: 'Oa01: 400 nm',
               2: 'Oa02: 412.5 nm',
               3: 'Oa03: 442.5 nm',
               4: 'Oa04: 490 nm',
               5: 'Oa05: 510 nm',
               6: 'Oa06: 560 nm',
               7: 'Oa07: 620 nm',
               8: 'Oa08: 665 nm',
               9: 'Oa09: 673.75 nm',
               10: 'Oa10: 681.25 nm',
               11: 'Oa11: 708.75 nm',
               12: 'Oa12: 753.75 nm',
               13: 'Oa16: 778.75 nm',
               14: 'Oa17: 865 nm',
               15: 'Oa18: 885 nm',
               16: 'Oa21: 1020 nm'}

    def tif_stats(self, vector, raster):

        logging.info(f'Input vector: {vector}')
        logging.info(f'Input raster: {raster}')

        from rasterstats import zonal_stats

        gdb = out.GDALBridge()
        bcount = gdb.get_tiff_band_count(raster)
        for n in range(bcount):
            print(self.band_id[n + 1])
            stats = zonal_stats(vector, raster, band=n + 1)
            print(stats)
        pass

    @staticmethod
    def mass_atmcor(folder, destination=None):

        work_dir = folder
        field_files = os.listdir(work_dir)

        for image in field_files:
            if destination:
                out.ICORBridge.run_iCOR_on_image(folder + '\\' + image, destination)
            else:
                out.ICORBridge.run_iCOR_on_image(folder + '\\' + image)

        pass


if __name__ == '__main__':

    run = Engine()

    # LOG HOTFIX FOR PyCharm
    # https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # CREATING HANDLER FOR USER INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log', help='Name and path of the log file to be saved.')
    parser.add_argument('-t', '--tif', help='Input .tiff image.')
    parser.add_argument('-s', '--shp', help='Input Shapefile.shp delimiting area.')
    parser.add_argument('-k', '--kml', help='Input KML delimiting area.')
    # boolean flag argument
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Whether or not to print to stdout.')
    args = parser.parse_args()

    # CREATE AND SAVE LOG FILE
    if args.verbose:
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S',
                            handlers=[logging.FileHandler(args.log),
                                      logging.StreamHandler()],
                            level=logging.INFO)
    else:
        logging.basicConfig(filename=args.log,
                            format='%(asctime)s - %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S',
                            level=logging.INFO)

    # # CALL INTERNAL iCOR MODULE ON EVERY IMAGE OF THE GIVEN FOLDER
    # mass_atmcor(sys.argv[1])

    # # TEST IF THE CORRECTED IMAGE SHOULD BE SAVED IN ANOTHER FOLDER OTHER THAN C:\Temp
    # if len(sys.argv) > 2:
    #     # mass_atmcor(sys.argv[1], sys.argv[2])
    # else:
    #     mass_atmcor(sys.argv[1])

    # # GET ZONAL STATISTICS FROM TIFF FILES
    logging.info(sys.argv)
    run.tif_stats(args.shp, args.tif)
