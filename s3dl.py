# connect to the API
# https://sentinelsat.readthedocs.io/en/stable/api.html

from decouple import config
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import os

user = config('USER', default='guest')
password = config('PASS', default='')
urlSci = 'https://scihub.copernicus.eu/dhus'
urlCoda = 'https://coda.eumetsat.int'
api = SentinelAPI(user, password, urlCoda)
JSON = 'd:\/git-repos\/s3-frbr\/amz_manacapuru.json'

# search by polygon, time, and SciHub query keywords
footprint = geojson_to_wkt(read_geojson(JSON))
products = api.query(
    footprint,
    date=('20200101', date(2020, 1, 16)),
    # platformname='Sentinel-3'
    filename='S3?_OL_2_?FR???*'
    # cloudcoverpercentage=(0, 30)
)

# convert query result to Pandas DataFrame
products_df = api.to_dataframe(products)

# =============================================================================
# filename = MMM_OL_L_TTTTTT_yyyymmddThhmmss_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_[instance ID]_GGG_[class ID].SEN3
# source: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-3-olci/naming-convention
# =============================================================================
# Scihub Copernicus Products Retention Policy
# https://scihub.copernicus.eu/userguide/LongTermArchive
# =============================================================================

# private function to build dynamicaly the wget download query
def _buildQuery(row):
    uuid = row['uuid']
    prod_name = row['identifier']
    wget = f'D:\wget.exe -O S3\{prod_name}.zip --no-check-certificate --user={user} --password={password} "http://coda.eumetsat.int/odata/v1/Products(\'{uuid}\')/$value"'
    return wget
      
# iterate over products dataframe rows, building the download query
queries = products_df.apply(_buildQuery, axis=1)

total = queries.shape[0]

os.system(f'echo =========================')
os.system(f'echo total number of files: {total}\n')
os.system(f'echo =========================')
os.system('echo ')
for i, result in enumerate(queries):
    file_name = products_df.iloc[i]['identifier']
    os.system(f'echo attempting to download image {i+1}/{total}... {file_name}')
    os.system('echo ')
    os.system(result)
