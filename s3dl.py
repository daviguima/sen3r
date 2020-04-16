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
    
# JSON = 'd:\/git-repos\/s3-frbr\/amz_manacapuru.json'

#%%

# search by polygon, time, and SciHub query keywords
# footprint = geojson_to_wkt(read_geojson(JSON))

# products = api.query(
#     # footprint,
#     date=('20191001', date(2020, 4, 1)), # day + 1
#     platformname='Sentinel-3',
#     # producttype='OL_2_LFR___',
#     filename='S3?_OL_2_WFR*',
#     # cloudcoverpercentage=(0, 30)
#     timeliness='Non Time Critical',
#     raw='footprint:"Intersects(POLYGON((-60.58496475219726 -3.3432664216192993, -60.549087524414055 -3.3432664216192993, -60.549087524414055 -3.3107057310886976, -60.58496475219726 -3.3107057310886976, -60.58496475219726 -3.3432664216192993)))"'
# )

products = api.query(
    # footprint,
    date=('20200213', date(2020, 2, 24)), # day + 1
    platformname='Sentinel-3',
    producttype='OL_1_EFR___',
    # filename='S3?_OL_2_*',
    # cloudcoverpercentage=(0, 30)
    timeliness='Non Time Critical',
    raw='footprint:"Intersects(POLYGON((-60.58496475219726 -3.3432664216192993, -60.549087524414055 -3.3432664216192993, -60.549087524414055 -3.3107057310886976, -60.58496475219726 -3.3107057310886976, -60.58496475219726 -3.3432664216192993)))"'
)

# raw footprint wkt from:
# http://geojson.io/#map=13/-3.3366/-60.5650

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
os.system(f'echo =========================\n\n')

#%%

for i, result in enumerate(queries):
    file_name = products_df.iloc[i]['identifier']
    os.system(f'echo attempting to download image {i+1}/{total}... {file_name}\n')
    os.system(result)
