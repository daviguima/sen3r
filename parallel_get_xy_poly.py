import netCDF4 as nc
import pandas as pd
import numpy as np
import os
import time
import logging
import concurrent.futures

from pathlib import Path


class ParallelCoord:
    
    def vect_dist_subtraction(self, coord_pair):
        subtraction = coord_pair - grid 
        dist = np.linalg.norm(subtraction, axis=2)
        result = np.where(dist == dist.min())
        target_x_y = [result[0][0], result[1][0]]
        return target_x_y

    # considering that creating the 2D grid consumes memory, we will get all coordinates in just one pass
    def parallel_get_xy_poly(self, lat_arr, lon_arr, polyline):

        global grid
        grid = np.concatenate([lat_arr[..., None], lon_arr[..., None]], axis=2)

        # Polyline is a GeoJSON coordinate array
        polyline = polyline.squeeze() # squeeze removes one of the dimensions of the array
        # https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html

        # Generate a list containing the lat, lon coordinates for each point of the input poly
        coord_vect_pairs = []
        for i in range(polyline.shape[0]):
            coord_vect_pairs.append(np.array([polyline[i, 1], polyline[i, 0]]).reshape(1, 1, -1))

        # for future reference
        # https://stackoverflow.com/questions/6832554/multiprocessing-how-do-i-share-a-dict-among-multiple-processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            try:
                result = list(executor.map(self.vect_dist_subtraction, coord_vect_pairs))

            except concurrent.futures.process.BrokenProcessPool as ex:
                logging.error(f"{ex} This might be caused by limited system resources. "
                              f"Try increasing system memory or disable concurrent processing. ")

        return np.array(result)

# # Deprecated: did not deleted for the sake of future reference.
# if __name__ == '__main__':
#     # SYN - UNIX
#     # path = Path('/d_drive_data/S3/SY_2_SYN/S3A_SY_2_SYN____20191104T135002_20191104T135302_20191106T033418_0180_051_124_3060_LN2_O_NT_002.SEN3')
#
#     # UNIX
#     path = Path('/d_drive_data/S3/L2_WFR/S3A_OL_2_WFR____20191104T135002_20191104T135302_20191124T134508_0179_051_124_3060_MAR_O_NT_002.SEN3')
#
#     polygon = np.array([ [ [ -61.304292937753878, -3.580746157823998, 0.0 ],
#                       [ -61.189738482291098, -3.570230611586103, 0.0 ],
#                       [ -61.051796038163282, -3.570623959031377, 0.0 ],
#                       [ -60.971175020538652, -3.593840244684938, 0.0 ],
#                       [ -60.9034262153111, -3.563360680269377, 0.0 ],
#                       [ -60.805020782919101, -3.506299406583337, 0.0 ],
#                       [ -60.753519094627137, -3.450505864090518, 0.0 ],
#                       [ -60.722456984927447, -3.39425947814228, 0.0 ],
#                       [ -60.69176635463603, -3.326974300372498, 0.0 ],
#                       [ -60.64421466170581, -3.314434177206381, 0.0 ],
#                       [ -60.572625051875058, -3.314523058454226, 0.0 ],
#                       [ -60.514541318054583, -3.307213505528399, 0.0 ],
#                       [ -60.501849367185159, -3.346258308422025, 0.0 ],
#                       [ -60.583418632397368, -3.33894030325553, 0.0 ],
#                       [ -60.665840030658408, -3.344760708787252, 0.0 ],
#                       [ -60.686558678731473, -3.374889635076542, 0.0 ],
#                       [ -60.755654470359637, -3.48512230522507, 0.0 ],
#                       [ -60.807307277936701, -3.544639712461309, 0.0 ],
#                       [ -60.874728354510736, -3.576891855197255, 0.0 ],
#                       [ -60.903342826613887, -3.613227458340859, 0.0 ],
#                       [ -60.932989587372482, -3.622059697419507, 0.0 ],
#                       [ -61.021414925808919, -3.605966358767145, 0.0 ],
#                       [ -61.091500853927002, -3.593526609351898, 0.0 ],
#                       [ -61.14960696557246, -3.598958178779615, 0.0 ],
#                       [ -61.240461667923427, -3.607404228546328, 0.0 ],
#                       [ -61.305985273744547, -3.604193025913813, 0.0 ],
#                       [ -61.304292937753878, -3.580746157823998, 0.0 ] ] ])
#     # # Open one band
#     # ds = nc.Dataset(path/'Oa17_reflectance.nc')
#     # band = ds['Oa17_reflectance'][:]
#
#     # Open the coordinates arrays
#     ds = nc.Dataset(path/'geo_coordinates.nc')
#     lat = ds['latitude'][:]
#     lon = ds['longitude'][:]
#
#     # Manacapuru
#
#     t1 = time.perf_counter()
#
#     output = parallel_get_xy_poly(lat, lon, polygon)
#
#     print('\nOutput:\n',output)
#
#     t2 = time.perf_counter()
#     print(f'>>> Finished in {round(t2 - t1, 2)} second(s). <<<')