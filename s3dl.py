# connect to the API
# https://sentinelsat.readthedocs.io/en/stable/api.html

from decouple import config
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date


user = config('USER', default='guest')
password = config('PASS', default='')
urlSci = 'https://scihub.copernicus.eu/dhus'
urlCoda = 'https://coda.eumetsat.int'
api = SentinelAPI(user, password, urlCoda)
JSON = 'amz_manacapuru.json'

footprint = geojson_to_wkt(read_geojson(JSON))

# # download single scene by known product id
# api.download(<product_id>)

# search by polygon, time, and SciHub query keywords
# footprint = geojson_to_wkt(read_geojson('/path/to/map.geojson'))
products = api.query(
    footprint,
    date=('20200101', date(2020, 1, 16)),
    # platformname='Sentinel-3'
    filename='S3?_OL_*'
    # cloudcoverpercentage=(0, 30)

)
#     MMM_OL_L_TTTTTT_yyyymmddThhmmss_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_[instance ID]_GGG_[class ID].SEN3
# convert to Pandas DataFrame
products_df = api.to_dataframe(products)

# sort and limit to first 5 sorted products
# products_df_sorted = products_df.sort_values(['cloudcoverpercentage', 'ingestiondate'], ascending=[True, True])
# products_df_sorted

# # download all results from the search
# api.download_all(products)
#
# # convert to Pandas DataFrame
# products_df = api.to_dataframe(products)
#
# # GeoJSON FeatureCollection containing footprints and metadata of the scenes
# api.to_geojson(products)
#
# # GeoPandas GeoDataFrame with the metadata of the scenes and the footprints as geometries
# api.to_geodataframe(products)
#
# # Get basic information about the product: its title, file size, MD5 sum, date, footprint and
# # its download url
# api.get_product_odata(<product_id>)
#
# # Get the product's full metadata available on the server
# api.get_product_odata(<product_id>, full=True)
