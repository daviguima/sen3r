# connect to the API
# https://sentinelsat.readthedocs.io/en/stable/api.html

#%%

from decouple import config
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import os

#%%

copernicus = 'https://scihub.copernicus.eu/dhus'
# eumetsat = 'https://coda.eumetsat.int'
eumetsat = 'https://codarep.eumetsat.int'

url = eumetsat # enable for WFR

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

# # # L2_WFR
# products = api.query(
#     # footprint,
#     date=('20181001', date(2020, 4, 1)), # day + 1
#     platformname='Sentinel-3',
#     # producttype='OL_2_LFR___',
#     filename='S3?_OL_2_WFR*',
#     # cloudcoverpercentage=(0, 30)
#     timeliness='Non Time Critical',
#     raw='footprint:"Intersects(POLYGON((-60.58496475219726 -3.3432664216192993, -60.549087524414055 -3.3432664216192993, -60.549087524414055 -3.3107057310886976, -60.58496475219726 -3.3107057310886976, -60.58496475219726 -3.3432664216192993)))"'
# )

# ## L2_WFR - marion lac_leman
# products = api.query(
#     # footprint,
#     date=('20200601', date(2020, 7, 31)), # day + 1
#     platformname='Sentinel-3',
#     # producttype='OL_2_LFR___',
#     filename='S3?_OL_2_WFR*',
#     # cloudcoverpercentage=(0, 30)
#     timeliness='Non Time Critical',
#     raw='footprint:"Intersects(POLYGON ((6.7549446041541 46.4576554696129 0,6.68393232338219 46.4780251525187 0,6.55403294760737 46.4840292180452 0,6.40956406124815 46.4189945401343 0,6.59072602154471 46.4207298624415 0,6.7549446041541 46.4576554696129 0)))"'
# )

## L2_WFR - florent rio solimões
products = api.query(
    # footprint='Intersects(POLYGON((-60.55399361680608 -3.2745035752850242,-60.55399361680608 -3.654179275522381,-61.01690906791547 -3.656405654932769,-61.01802452683382 -3.271162656476122,-60.55399361680608 -3.2745035752850242,-60.55399361680608 -3.2745035752850242)))',
    date=('20160101', date(2019, 3, 9)),  # day + 1
    platformname='Sentinel-3',
    # producttype='OL_2_LFR___',
    # producttype='OL_2_WFR___',
    # area_relation='Intersects',
    filename='S3?_OL_2_WFR*',
    # cloudcoverpercentage=(0, 30)
    timeliness='Non Time Critical',
    # raw='footprint:"Intersects(POLYGON ((-65.7641589517219 -2.6151263096657 0,-65.7524481609865 -2.61778217383069 0,-65.7462434442559 -2.60430743613057 0,-65.753367002201 -2.59898995617844 0,-65.7632643349048 -2.60517702580467 0,-65.7641589517219 -2.6151263096657 0)))"'
    # raw='footprint:"Intersects(POLYGON((-60.509069928849534 -3.2626561486855366,-60.51241630560453 -3.398512248234084,-60.65965688282487 -3.386263700982525,-60.79239649410683 -3.6234118961110795,-60.92848248214382 -3.677958469621373,-61.04895204532409 -3.6723926448251945,-61.06122209342579 -3.513195570510689,-60.857093111370325 -3.5187623671484403,-60.76451002114844 -3.3450629082099255,-60.68196606119159 -3.265997095821632,-60.603883936908076 -3.2548605621996387,-60.509069928849534 -3.2626561486855366,-60.509069928849534 -3.2626561486855366)))"'
    # raw='footprint:"Intersects(POLYGON((-60.55399361680608 -3.2745035752850242,-60.55399361680608 -3.654179275522381,-61.01690906791547 -3.656405654932769,-61.01802452683382 -3.271162656476122,-60.55399361680608 -3.2745035752850242,-60.55399361680608 -3.2745035752850242)))"'
)

# # L1_EFR
# products = api.query(
#     # footprint,
#     date=('20180501', date(2018, 6, 1)), # day + 1
#     platformname='Sentinel-3',
#     producttype='OL_1_EFR___',
#     # filename='S3?_OL_2_*',
#     # cloudcoverpercentage=(0, 30)
#     timeliness='Non Time Critical',
#     raw='footprint:"Intersects(POLYGON((-60.58496475219726 -3.3432664216192993, -60.549087524414055 -3.3432664216192993, -60.549087524414055 -3.3107057310886976, -60.58496475219726 -3.3107057310886976, -60.58496475219726 -3.3432664216192993)))"'
# )

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
    # os.system(result)
