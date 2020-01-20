# connect to the API
# https://sentinelsat.readthedocs.io/en/stable/api.html

#%%

from decouple import config
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import os

#%%

copernicus = 'https://scihub.copernicus.eu/dhus'
eumetsat = 'https://coda.eumetsat.int'

# url = eumetsat # WFR

if 'url' in globals():
    print(f'URL set to {url}')
    user = config('EUM_USER', default='guest')
    password = config('EUM_PASS', default='')
    api = SentinelAPI(user, password, url)
else:
    print(f'Variable \'URL\' not set. Using default path: {copernicus}')
    user = config('COP_USER', default='guest')
    password = config('COP_PASS', default='')
    api = SentinelAPI(user, password)
    
JSON = 'd:\/git-repos\/s3-frbr\/amz_manacapuru.json'

#%%

# search by polygon, time, and SciHub query keywords
footprint = geojson_to_wkt(read_geojson(JSON))
products = api.query(
    # footprint,
    # area='intersects(POINT (-3.3269005247809025, -60.570201873779304))',
    date=('20190101', date(2019, 1, 31)),
    # platformname='Sentinel-3'
    producttype='OL_2_LFR___',
    # filename='S3?_OL_2_?FR*',
    # cloudcoverpercentage=(0, 30)
    timeliness='Non Time Critical',
    query="'footprint': 'POLYGON ((34.322010 0.401648,36.540989 0.876987,36.884121 -0.747357,34.664474 -1.227940,34.322010 0.401648))',"
)

#%%

# convert query result to Pandas DataFrame
products_df = api.to_dataframe(products)

# =============================================================================
# filename = MMM_OL_L_TTTTTT_yyyymmddThhmmss_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_[instance ID]_GGG_[class ID].SEN3
# source: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-3-olci/naming-convention
# =============================================================================
# Scihub Copernicus Products Retention Policy
# https://scihub.copernicus.eu/userguide/LongTermArchive
# =============================================================================

#%%

# private function to build dynamicaly the wget download query for COAH/Copernicus
# LFR
def _buildQueryCopernicus(row):
    uuid = row['uuid']
    prod_name = row['identifier']
    wget = f'D:\wget.exe -O S3\{prod_name}.zip --continue --no-check-certificate --user={user} --password={password} "https://scihub.copernicus.eu/apihub/odata/v1/Products(\'{uuid}\')/$value"'
    return wget

#%%

# private function to build dynamicaly the wget download query for CODA/eumetsat
# WFR
def _buildQueryEumetsat(row):
    uuid = row['uuid']
    prod_name = row['identifier']
    wget = f'D:\wget.exe -O S3\{prod_name}.zip --no-check-certificate --user={user} --password={password} "http://coda.eumetsat.int/odata/v1/Products(\'{uuid}\')/$value"'
    return wget
      
#%%

# iterate over products dataframe rows, building the download query
if 'url' in globals():
    queries = products_df.apply(_buildQueryEumetsat, axis=1)
else:
    queries = products_df.apply(_buildQueryCopernicus, axis=1)

#%%

total = queries.shape[0]

#%%

os.system(f'echo =========================')
os.system(f'echo total number of files: {total}\n')
os.system(f'echo =========================')
os.system('echo ')

#%%

# for i, result in enumerate(queries):
#     file_name = products_df.iloc[i]['identifier']
#     os.system(f'echo attempting to download image {i+1}/{total}... {file_name}')
#     os.system('echo ')
#     os.system(result)
    
    