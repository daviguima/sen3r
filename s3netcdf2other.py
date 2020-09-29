# https://gis.stackexchange.com/questions/273536/netcdf-with-separate-lat-lon-bands-to-geotiff-with-python-gdal
# https://github.com/s6hebern/scripts/blob/master/Python/geospatial/s3netcdf2other.py

import os
import sys
import shutil
import math
import subprocess as sub
from osgeo import gdal, osr
from netCDF4 import Dataset
from basic_functions.callCmd import callCmd


def calculateUtmZone(input):
    ds = gdal.Open(input, gdal.GA_ReadOnly)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    srs.AutoIdentifyEPSG()
    epsg = int(srs.GetAttrValue('AUTHORITY', 1))
    if epsg != 4326:
        raise AttributeError('Coordinate System is not WGS84 with EPSG 4326!')
    geotrans = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ds = None
    cent_x = geotrans[0] + geotrans[1] * (cols / 2)
    cent_y = geotrans[3] + geotrans[5] * (rows / 2)
    utm_zone_num = int(math.floor((cent_x + 180) / 6) + 1)
    utm_zone_hemi = 6 if cent_y >= 0 else 7
    utm_epsg = 32000 + utm_zone_hemi * 100 + utm_zone_num
    return utm_epsg


def s3netcdf2other(input_dir, output_image, instrument='OLCI', outformat='GTiff', to_utm=True, outres=300):
    tempdir = os.path.join(input_dir, 'tmp')
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)
    os.mkdir(tempdir)
    print('Retrieving coordinates...')
    if instrument == 'OLCI':
        BANDNAMES = ['Oa{0}_radiance'.format(str(i).zfill(2)) for i in range(1, 22)]  # range(1, 22)
        nc_coords = os.path.join(input_dir, 'geo_coordinates.nc')
        ds_nc = Dataset(nc_coords, 'r')
        coords = (ds_nc.variables['latitude'], ds_nc.variables['longitude'])
    elif instrument == 'SLSTR':
        BANDNAMES = ['S{0}_radiance_an'.format(str(i).zfill(1)) for i in range(1, 7)]  # range(1, 10)
        BANDNAMES = BANDNAMES + ['S{0}_BT_in'.format(str(i).zfill(1)) for i in range(7, 10)]  # range(1, 10)
        BANDNAMES = BANDNAMES + ['F{0}_BT_in'.format(str(i).zfill(1)) for i in range(1, 3)]  # range(1, 10)
        nc_coords = os.path.join(input_dir, 'geodetic_an.nc')
        ds_nc = Dataset(nc_coords, 'r')
        coords = (ds_nc.variables['latitude_an'], ds_nc.variables['longitude_an'])
    else:
        print('Wrong instrument indicator! Must be either "OLCI" or "SLSTR"!')
    lat_tif = os.path.join(tempdir, coords[0].name + '.tif')
    lon_tif = os.path.join(tempdir, coords[1].name + '.tif')
    rad_tifs = []
    nc_paths = [os.path.join(input_dir, band + '.nc') for band in BANDNAMES]
    # get lat/lon
    for v, var in enumerate(coords):
        nodata = var._FillValue
        scale = var.scale_factor
        var_vrt = os.path.join(tempdir, var.name + '.vrt')
        var_tif = os.path.join(tempdir, var.name + '.tif')
        cmd = ['gdalbuildvrt', '-sd', str(1 + var._varid), '-separate', '-overwrite', var_vrt, nc_coords]
        sub.call(cmd)
        # edit vrt
        with open(var_vrt, 'r') as f:
            xml = f.readlines()
        for line in xml:
            if '<VRTRasterBand ' in line:
                head_index = xml.index(line) + 1
            if '<DstRect xOff' in line:
                tail_index = xml.index(line) + 1
        xml.insert(head_index, '    <NoDataValue>{nd}</NoDataValue>\n'.format(nd=nodata))
        xml.insert(head_index + 1, '    <Scale>{sc}</Scale>\n'.format(sc=scale))
        tail_index = tail_index + 2
        xml.insert(tail_index, '      <NODATA>{nd}</NODATA>\n'.format(nd=nodata))
        xml.insert(tail_index + 2, '    <Offset>0.0</Offset>\n')
        xml.insert(tail_index + 3, '    <Scale>{sc}</Scale>\n'.format(sc=scale))
        xml = [line.replace('="Int32', '="Float32') for line in xml]
        with open(var_vrt, 'w') as f:
            f.writelines(xml)
        # write to temporary tif
        cmd = ['gdal_translate', '-unscale', var_vrt, var_tif]
        sub.call(cmd)
    ds_nc.close()
    # single bands to vrt, then to tif
    print('Converting all {n} bands...'.format(n=len(BANDNAMES)))
    for n, nc in enumerate(nc_paths):
        print('\t... BAND {b}'.format(b=n+1))
        ds_nc = Dataset(nc, 'r')
        var = ds_nc.variables[os.path.basename(nc)[:-3]]
        nodata = var._FillValue
        offset = var.add_offset
        rows = var.shape[0]
        scale = var.scale_factor
        ds_nc.close()
        data_vrt = os.path.join(tempdir, 'data.vrt')
        data_vrt_tif = data_vrt.replace('.vrt', '.tif')
        out_vrt = os.path.join(tempdir, os.path.basename(nc)[:-3] + '.vrt')
        out_tif = out_vrt.replace('.vrt', '.tif')
        if instrument == 'OLCI':
            cmd = ['gdalbuildvrt', '-sd', '1', '-separate', '-overwrite', data_vrt, nc]
        else:
            if os.path.basename(nc).endswith('BT_in.nc'):
                cmd = ['gdalbuildvrt', '-sd', '1', '-separate', '-overwrite', data_vrt, nc]
            else:
                cmd = ['gdalbuildvrt', '-sd', '3', '-separate', '-overwrite', data_vrt, nc]
        sub.call(cmd)
        # edit vrt
        with open(data_vrt, 'r') as f:
            xml = f.readlines()
        for line in xml:
            if '<VRTRasterBand ' in line:
                head_index = xml.index(line)
            if '<DstRect xOff' in line:
                tail_index = xml.index(line) + 1
        xml[head_index] = '  <VRTRasterBand dataType="Float32" band="1">\n'
        xml.insert(head_index + 1, '    <NoDataValue>{nd}</NoDataValue>\n'.format(nd=nodata))
        xml[head_index + 2] = '    <ComplexSource>\n'
        xml[head_index + 5] = xml[head_index + 5].replace('DataType="UInt16"', 'DataType="Float32"')
        tail_index = tail_index + 1
        xml.insert(tail_index, '      <NODATA>{nd}</NODATA>\n'.format(nd=nodata))
        xml[tail_index + 1] = '    </ComplexSource>\n'
        xml.insert(tail_index + 2, '    <Offset>{off}</Offset>\n'.format(off=offset))
        xml.insert(tail_index + 3, '    <Scale>{sc}</Scale>\n'.format(sc=scale))
        with open(data_vrt, 'w') as f:
            f.writelines(xml)
        # write to temporary tif, then build a new vrt
        cmd = ['gdal_translate', '-unscale', data_vrt, data_vrt_tif]
        sub.call(cmd)
        # update GeoTransform
        ds = gdal.Open(data_vrt_tif, gdal.GA_Update)
        ds.SetGeoTransform((0.0, 1.0, 0.0, float(rows), 0.0, -1.0))
        ds.FlushCache()
        # build new vrt
        cmd = ['gdalbuildvrt', '-sd', '1', '-separate', '-overwrite', out_vrt, data_vrt_tif]
        sub.call(cmd)
        # edit vrt
        with open(out_vrt, 'r') as f:
            xml = f.readlines()
        for line in xml:
            if '<VRTRasterBand ' in line:
                head_index = xml.index(line)
                break
        xml[head_index] = '  <VRTRasterBand dataType="Float32" band="1">\n'
        xml.insert(-1, '''  <metadata domain="GEOLOCATION">
    <mdi key="X_DATASET">{lon}</mdi>
    <mdi key="X_BAND">1</mdi>
    <mdi key="Y_DATASET">{lat}</mdi>
    <mdi key="Y_BAND">1</mdi>
    <mdi key="PIXEL_OFFSET">0</mdi>
    <mdi key="LINE_OFFSET">0</mdi>
    <mdi key="PIXEL_STEP">1</mdi>
    <mdi key="LINE_STEP">1</mdi>
  </metadata>\n'''.format(lon=lon_tif, lat=lat_tif))

        for line in xml:
            if os.sep in line:
                xml[xml.index(line)] = line.replace(os.sep, '/')

        with open(out_vrt, 'w') as f:
            f.writelines(xml)
        # convert to tif
        cmd = ['gdalwarp', '-t_srs', 'epsg:4326', '-geoloc', '-srcnodata', str(nodata), '-dstnodata', '-9999', out_vrt,
               out_tif]
        sub.call(cmd)
        # remove temp files safely
        os.remove(out_vrt)
        ds = gdal.Open(data_vrt_tif, gdal.GA_ReadOnly)
        ds = None
        os.remove(data_vrt_tif)
        rad_tifs.append(out_tif)

    # stack together
    print('Stacking bands...')
    if 'win' in sys.platform.lower():
        gdal_path = r'c:\Program Files\GDAL'
    else:
        gdal_path = r'/usr/bin'
    gdal_merge = os.path.join(gdal_path, 'gdal_merge.py')
    stack = os.path.join(tempdir, 'stack.tif')
    cmd = ['python', gdal_merge, '-separate', '-n', '-9999', '-o', stack]

    for r in rad_tifs:
        cmd.append(r)
    if os.path.exists(output_image):
        drv = gdal.GetDriverByName(outformat)
        drv.Delete(output_image)
    sub.call(cmd)
    if to_utm is True:
        epsg = calculateUtmZone(stack)
        print('Reprojecting to UTM (EPSG: {e})'.format(e=epsg))
        cmd = ['gdalwarp', '-of', outformat, '-srcnodata', '-9999', '-dstnodata', '-9999', '-overwrite',
               '-t_srs', 'epsg:{e}'.format(e=str(epsg)), '-tr', str(outres), str(outres), stack, output_image]
        sub.call(cmd)
    else:
        if not outformat == 'GTiff':
            print('Converting to {of}...'.format(of=outformat))
            cmd = ['gdal_translate', '-of', outformat, '-a_nodata', '-9999', stack, output_image]
            if outformat == 'ENVI':
                cmd.append('-co')
                cmd.append('interleave=bil')
            elif outformat == 'GTiff':
                cmd.append('-co')
                cmd.append('compress=lzw')
                sub.call(cmd)
    print('Cleaning up...')
    shutil.rmtree(tempdir)
    print('Done!')


if __name__ == '__main__':
    indir = r'd:\working\testing\s3\S3A_OL_1_EFR____20180402T093229_20180402T093529_20180403T155138_0179_029_307_1980_MAR_O_NT_002.SEN3'
    instru = 'OLCI'
    out = os.path.join(os.path.dirname(indir), 'OLCI')
    s3netcdf2other(indir, out, instrument=instru, to_utm=True)
    indir = r'd:\working\testing\s3\S3A_SL_1_RBT____20180402T093229_20180402T093529_20180403T170539_0180_029_307_1980_MAR_O_NT_002.SEN3'
    instru = 'SLSTR'
    out = os.path.join(os.path.dirname(indir), 'SLSTR')
    s3netcdf2other(indir, out, instrument=instru, to_utm=True)
