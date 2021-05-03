import os
import time
import ogr2ogr
import subprocess
import concurrent.futures

import json
import utils

from osgeo import ogr
from pathlib import Path
from pathlib import PurePath as ppp

from utils import tele_print


class FootprintGenerator:

    def __init__(self, footprint_output_folder=None):
        self.output_folder = footprint_output_folder

    @staticmethod
    def _xml2dict(xfdumanifest):
        '''
        Internal function that reads the .SEN3/xfdumanifest.xml and generates a dictionary with the relevant data.
        '''
        result = {}
        with open(xfdumanifest) as xmlf:
            # grab the relevant contents and add them to a dict
            for line in xmlf:
                if "<gml:posList>" in line:
                    result['raw_coords'] = line.split('</')[0].split('>')[1]
                    result['gml_data'] = f'<gml:Polygon xmlns:gml="http://www.opengis.net/gml" ' \
                                    f'srsName="http://www.opengis.net/def/crs/EPSG/0/4326">' \
                                    f'<gml:exterior><gml:LinearRing>{line}</gml:LinearRing>' \
                                    f'</gml:exterior></gml:Polygon>'
                # get only the values between the tags
                if '<sentinel3:rows>' in line:
                    result['rows'] = int(line.split('</')[0].split('>')[1])
                if '<sentinel3:columns>' in line:
                    result['cols'] = int(line.split('</')[0].split('>')[1])
        return result

    def manifest2json(self, xfdumanifest):
        '''
        Given a .SEN3/xfdumanifest.xml generates a IMGNAME_footprint.geojson
        '''
        # get the dict
        xmldict = self._xml2dict(xfdumanifest)
        # remove the file name and keep the path
        img_folder = xfdumanifest.split('xfdu')[0]
        # test weather the user has given or not a destination output
        if self.output_folder is None:
            # add an img.SEN3/footprint.geojson inside the original image folder
            footprint_dir = os.path.join(img_folder, 'footprint.geojson')
        else:
            # create file inside output destination with a name like:
            # 'S3A_OL_2_WFR____20160425T134227_20160425T134427_20171030T154612_0119_003_238______MR1_R_NT_002_footprint.geojson'
            just_the_img_folder_name = os.path.basename(os.path.normpath(img_folder)).split('.')[0]
            footprint_dir = os.path.join(self.output_folder, just_the_img_folder_name + '_footprint.geojson')

        str_coords = xmldict['raw_coords'].split()

        coord_tuple_list = [[float(y), float(x)] for x, y in utils.grouped(str_coords, 2)]

        # mock the geojson file
        json_content = {
            "type": "FeatureCollection",
            "name": just_the_img_folder_name.split('.')[0]+'_footprint',
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": [
                {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [
                    coord_tuple_list]}}
            ]
        }

        # write json data from the dict to an actual .geojson file
        with open(footprint_dir, 'w') as f:
            json.dump(json_content, f)
        return xmldict

    def manifest2shp(self, xfdumanifest, filename):
        '''
        Given a .SEN3/xfdumanifest.xml and a filename, generates a .shp and a .gml
        '''
        # get the dict
        xmldict = self._xml2dict(xfdumanifest)
        # add path to "to-be-generated" gml and shp files
        xmldict['gml_path'] = filename + '.gml'
        xmldict['shp_path'] = filename + '.shp'
        # write the gml_data from the dict to an actual .gml file
        with open(xmldict['gml_path'], 'w') as gmlfile:
            gmlfile.write(xmldict['gml_data'])
        # call the ogr2ogr.py script to generate a .shp from a .gml
        ogr2ogr.main(["", "-f", "ESRI Shapefile", xmldict['shp_path'], xmldict['gml_path']])
        return xmldict

    @staticmethod
    def _shp_extent(shp):
        '''
        Reads a ESRI Shapefile and return its extent as a str separated by spaces.
        e.x. output: '-71.6239 -58.4709 -9.36789 0.303954'
        '''
        ds = ogr.Open(shp)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        geometry = feature.GetGeometryRef()
        extent = geometry.GetEnvelope()  # ex: (-71.6239, -58.4709, -9.36789, 0.303954)
        # cast the 4-elements tuple into a list of strings
        extent_str = [str(i) for i in extent]
        return ' '.join(extent_str)

    def manifest2tiff(self, xfdumanifest):
        '''
        Reads .SEN3/xfdumanifest.xml and generates a .tiff raster.
        '''

        # test for win/linux and get the complete directory path but not the file base name
        if os.name == 'nt':  # WIN
            img_path = xfdumanifest.split('\\xfdu')[0]
        else:  # LINUX
            img_path = xfdumanifest.split('/xfdu')[0]

        # get only the date of the img from the complete path, ex: '20190904T133117'
        figdate = os.path.basename(img_path).split('____')[1].split('_')[0]

        # test weather the user has given or not a destination output
        if self.output_folder is None:
            # add an img.SEN3/footprint inside the original image folder
            footprint_dir = os.path.join(img_path, 'footprint')
        else:
            # this will create a folder inside the selected output destination with a name like:
            # 'S3A_OL_2_WFR____20160425T134227_20160425T134427_20171030T154612_0119_003_238______MR1_R_NT_002_footprint'
            just_the_img_folder_name = os.path.basename(img_path).split('.')[0]
            footprint_dir = os.path.join(self.output_folder, just_the_img_folder_name+'_footprint')

        # create a folder to save the footprint + auxiliary files
        Path(footprint_dir).mkdir(parents=True, exist_ok=True)

        # ex: img.SEN3/footprint/20190904T133117_footprint.tiff
        path_file_tiff = os.path.join(footprint_dir, figdate+'_footprint.tiff')

        # only '20190904T133117_footprint' nothing else.
        lyr = os.path.basename(path_file_tiff).split('.')[0]

        # img.SEN3/footprint/20190904T133117_footprint (without file extension)
        fname = path_file_tiff.split('.tif')[0]

        # get the dict + generate .shp and .gml files
        xmldict = self.manifest2shp(xfdumanifest, fname)  # TODO: fix the shape for windows
        cols = xmldict['cols']
        rows = xmldict['rows']
        shp = xmldict['shp_path']
        extent = self._shp_extent(shp)

        # generate the complete cmd string
        cmd = f'gdal_rasterize -l {lyr} -burn 1.0 -ts {cols}.0 {rows}.0 -a_nodata 0.0 -te {extent} -ot Float32 {shp} {path_file_tiff}'

        # call the cmd
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        proc.wait()
        print(f'{figdate} done.')
        return True


if __name__ == '__main__':

    # go_parallel = False
    # fg = FootprintGenerator(footprint_output_folder='C:\\Temp')
    # t1 = time.perf_counter()
    # now = time.strftime("%H:%M:%S", time.localtime())
    # # tele_print(f'FootprintGenerator started at {now}.\n')
    # print(f'FootprintGenerator started at {now}.\n')
    #
    # # UNIX
    # # s3_imgs_dir = '/d_drive_data/S3/L2_WFR'
    # # WIN
    # s3_imgs_dir = 'D:\\S3\\L2_WFR'
    #
    # # adding every image inside the working directory to a list
    # files = os.listdir(s3_imgs_dir)
    #
    # # UNIX
    # # adding the complete path to each image folder listed + xml file
    # # fullpath_xmls = [os.path.join(s3_imgs_dir, image+'/xfdumanifest.xml') for image in files]
    # # WIN
    # fullpath_xmls = [os.path.join(s3_imgs_dir, image + '\\xfdumanifest.xml') for image in files]
    #
    # # testing with a couple of images only
    # fullpath_xmls = fullpath_xmls[0:3]
    #
    # total = len(fullpath_xmls)
    #
    # print(f'FootprintGenerator:\nTotal files identified in folder: {total}\n')
    # # tele_print(f'FootprintGenerator:\nTotal files identified in folder: {total}\n')
    # if go_parallel:
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
    #         try:
    #             result = list(executor.map(fg.manifest2tiff, fullpath_xmls))
    #         except concurrent.futures.process.BrokenProcessPool as ex:
    #             print(f"{ex} This might be caused by limited system resources. "
    #                   f"Try increasing system memory or disable concurrent processing. ")
    # else:
    #     for manifest in fullpath_xmls:
    #         fg.manifest2tiff(manifest)
    #
    # t2 = time.perf_counter()
    #
    # print(f'FootprintGenerator >>> Finished in {round(t2 - t1, 2)} second(s).')
    # # tele_print(f'FootprintGenerator >>> Finished in {round(t2 - t1, 2)} second(s).')

    fg = FootprintGenerator(footprint_output_folder='/d_drive_data/CNES/experiments')
    xml_fn = '/d_drive_data/S3/L2_WFR/S3B_OL_2_WFR____20200429T132147_20200429T132447_20200430T201832_0179_038_195_3060_MAR_O_NT_002.SEN3/xfdumanifest.xml'
    fg.manifest2json(xml_fn)
