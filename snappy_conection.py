from snappy import ProductIO

product_input_path = 'D:\S3\output\S3B_OL_1_EFR____20190822T132911_20190822T133211_20190823T173023_0180_029_081_3060_LN1_O_NT_002.SEN3_rayleigh.dim'
product_output_path = 'D:\S3\output'

p = ProductIO.readProduct(product_input_path)  # read product

ProductIO.writeProduct(p, product_output_path, 'GeoTIFF-BigTIFF')  # write product