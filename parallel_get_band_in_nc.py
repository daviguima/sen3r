import os
import logging
import concurrent.futures
import netCDF4 as nc
import pandas as pd


class ParallelBandExtract:


    def _get_band_in_nc(self, file_n_band):

        print(f'Extracting band: {file_n_band[1]} from file: {file_n_band[0]}.\n')
        result = {}
        # load NetCDF folder + nc_file_name
        ds = nc.Dataset(file_n_band[0])
        # load the nc_band_name as a matrix and unmask its values
        band = ds[file_n_band[1]][:].data
        # extract the values of the matrix and return as a dict entry
        result[file_n_band[1]] = [band[x, y] for x, y in zip(p_rr, p_cc)]

    def parallel_get_bdata_in_nc(self, rr, cc, lon, lat, nc_folder, wfr_files_p):
        """
        Given an input polygon and image, return a dataframe containing
        the data of the image that falls inside the polygon.
        """
        global p_rr
        p_rr = rr.copy()
        global p_cc
        p_cc = cc.copy()

        wfr_files_p = [(os.path.join(nc_folder, nc_file), nc_band) for nc_file, nc_band in wfr_files_p]

        # Generate the initial dataframe:
        custom_subset = {'x': rr, 'y': cc}
        df = pd.DataFrame(custom_subset)
        df['lat'] = [lat[x, y] for x, y in zip(df['x'], df['y'])]
        df['lon'] = [lon[x, y] for x, y in zip(df['x'], df['y'])]

        # Populate it with the output from the other bands
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            try:
                list_of_bands = list(executor.map(self._get_band_in_nc, wfr_files_p))
            except concurrent.futures.process.BrokenProcessPool as ex:
                logging.error(f"{ex} This might be caused by limited system resources. "
                              f"Try increasing system memory or disable concurrent processing. ")

        return list_of_bands
