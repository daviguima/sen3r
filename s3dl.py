# connect to the API
# https://sentinelsat.readthedocs.io/en/stable/api.html

from decouple import config
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date

if __name__ == "__main__":

    user = config('USER', default='guest')
    password = config('PASS', default='')
    urlSci = 'https://scihub.copernicus.eu/dhus'
    urlCoda = 'https://coda.eumetsat.int'
    api = SentinelAPI(user, password, urlCoda)

    # # download single scene by known product id
    # api.download(<product_id>)

    # search by polygon, time, and SciHub query keywords
    # footprint = geojson_to_wkt(read_geojson('/path/to/map.geojson'))
    products = api.query(
                         date=('20151219', date(2015, 12, 29)),
                         platformname='Sentinel-2',
                         cloudcoverpercentage=(0, 30)
                        )

    # convert to Pandas DataFrame
    products_df = api.to_dataframe(products)

    # sort and limit to first 5 sorted products
    products_df_sorted = products_df.sort_values(['cloudcoverpercentage', 'ingestiondate'], ascending=[True, True])
    products_df_sorted
    print(products_df_sorted)
    # for p in products:
    #     print(p)

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
