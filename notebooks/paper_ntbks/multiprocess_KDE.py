import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from datetime import datetime
from osgeo import gdal,osr,ogr

import sys
sys.path.append('../../')

from tsgen import TsGenerator
# from nc_explorer import NcExplorer

# LINUX
in_dir = '/d_drive_data/S3/L2_WFR_subset/MANACAPURU'

tsg = TsGenerator()

todo = tsg.build_list_from_subset(in_dir)
todo_fullpath = [os.path.join(in_dir,csv) for csv in todo]
print(f'{todo[0]}\n\n{todo_fullpath[0]}')

def get_simple_date(complete_path_to_csv):
    id_date = os.path.basename(complete_path_to_csv).split('____')[1].split('_')[0]
    dtlbl = datetime.strptime(id_date, '%Y%m%dT%H%M%S')
    # result = {'str_date':id_date, 'dt.time':dtlbl}
    return id_date, dtlbl

def get_kde_cluster_min_max(complete_path_to_csv):
    
    kde_drop_th=0.1
    
    # get the date of the image
    id_date, dtlbl = get_simple_date(complete_path_to_csv)
    
    print(f'processing file: {id_date} -- {dtlbl} ...')
    
    # read the csv
    raw_df = pd.read_csv(complete_path_to_csv, sep='\t', skiprows=1)
    
    # clear and add several new columns to DF using SEN3R > tsgen.py
    tsgen = TsGenerator()
    df = tsg.update_df(raw_df)
    
    # test if it is possible to apply KDE to the dataframe
    flag_kde = False
    
    if len(df['Oa08_reflectance:float'].unique()) > 2:
        flag_kde = True
        # prepare x to be KDE's histogram variable
        x = df['Oa08_reflectance:float'].copy()
        # get the KDE clusters
        pk, xray, yray, kde_res = tsg.kde_local_maxima(x)
        # create a list with the KDE local maximas
        kdemaxes = [m for m in xray[pk]]
        # compute simple statistical mean
        xmean = np.mean(x)
        # add it to the list of kde local maximas
        kdemaxes.append(xmean)
        
        Oa08_kde_min = min(kdemaxes)
        Oa08_kde_max = max(kdemaxes)
        
        # where is 10% above the lower cluster reflectance ?
        drop_above_this = min(kdemaxes) + ( min(kdemaxes) * kde_drop_th )
        # get the position of the clusters below the lowest reflectance clusters
        idx_lower_clstrs = df[df['Oa08_reflectance:float'] < drop_above_this].index
        # get the mean T865 over the lower clusters
        t865kdmin = df.loc[idx_lower_clstrs,'T865:float'].mean(skipna=True)
        
        # where is 10% below the higher cluster reflectance ?
        drop_below_this = max(kdemaxes) - ( max(kdemaxes) * kde_drop_th )
        # get the index position of the highest reflectance clusters
        idx_higher_clstrs = df[df['Oa08_reflectance:float'] > drop_below_this].index
        # get the mean T865 over the upper clusters
        t865kdmax = df.loc[idx_higher_clstrs,'T865:float'].mean(skipna=True)
    
    if not flag_kde:
        Oa08_kde_min = np.nan
        Oa08_kde_max = np.nan
        t865kdmin = np.nan
        t865kdmax = np.nan
    
    b8m = df['Oa08_reflectance:float'].mean(skipna=True)
    t865m = df['T865:float'].mean(skipna=True)
    
    # compute statistics and build the result
    result = {'strdate':id_date,
              'dttime':dtlbl,
              'B8m':b8m,
              'B8kd.min':Oa08_kde_min,
              'B8kd.max':Oa08_kde_max,
              'kde.flag':flag_kde,
              'T865m':t865m,
              'T865.kd.min':t865kdmin,
              'T865.kd.max':t865kdmax}
    
    return result

import concurrent.futures
import time

t1 = time.perf_counter()

with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
    try:
        result = list(executor.map(get_kde_cluster_min_max, todo_fullpath))
    except concurrent.futures.process.BrokenProcessPool as ex:
        logging.error(f"{ex} This might be caused by limited system resources. "
                      f"Try increasing system memory or disable concurrent processing. ")

# for future reference
# https://stackoverflow.com/questions/6832554/multiprocessing-how-do-i-share-a-dict-among-multiple-processes

t2 = time.perf_counter()
print(f'>>> Finished in {round(t2 - t1, 2)} second(s). <<<')

final_df = pd.DataFrame(data=result)
final_df.to_csv('parallel_kde_out.csv')