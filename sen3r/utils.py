import os
import sys
import time
import json
import geopandas
import numpy as np
from PIL import Image


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    return t_hour, t_min, t_sec


def repeat_to_length(s, wanted):
    return (s * (wanted//len(s) + 1))[:wanted]


def pil_grid(images, max_horiz=np.iinfo(int).max):
    """
    Combine several images
    https://stackoverflow.com/a/46877433/2238624
    """
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    #     h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * ((n_images // n_horiz) + (1 if n_images % n_horiz > 0 else 0))
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid


def geojson_to_polygon(poly_path):
    """
    Transform a given input .geojson file into a list of coordinates
    poly_path: string (Path to .geojson file)
    return: dict (Containing one or several polygons depending on the complexity of the input .json)
    """
    with open(poly_path) as f:
        data = json.load(f)

    poly_lst = []
    for feature in data['features']:
        poly = feature['geometry']['coordinates']
        poly = np.array(poly[0])
        poly_lst.append(poly)

    vertices = [poly[:, :2] for poly in poly_lst]

    return vertices


# KML -> GEOJson : https://github.com/mrcagney/kml2geojson
def shp_to_geojson(shp_file_path, json_out_path):
    shpfile = geopandas.read_file(shp_file_path)
    f_name = os.path.basename(shp_file_path).split('.')[0]
    shpfile.to_file(os.path.join(json_out_path, f_name+'.geojson'), driver='GeoJSON')


# Just a small refactoring of the old code
def get_x_y(lat_arr, lon_arr, lat, lon):
    grid = np.concatenate([lat_arr[..., None], lon_arr[..., None]], axis=2)

    vector = np.array([lat, lon]).reshape(1, 1, -1)
    subtraction = vector - grid
    dist = np.linalg.norm(subtraction, axis=2)
    result = np.where(dist == dist.min())
    target_x_y = result[0][0], result[1][0]

    return target_x_y


# considering that creating the 2D grid consumes memory, we will get all coordinates in just one pass
def get_x_y_poly(lat_arr, lon_arr, polyline):
    grid = np.concatenate([lat_arr[..., None], lon_arr[..., None]], axis=2)

    # Polyline is a GeoJSON coordinate array
    polyline = polyline.squeeze()

    # loop through each vertice
    vertices = []
    for i in range(polyline.shape[0]):
        vector = np.array([polyline[i, 1], polyline[i, 0]]).reshape(1, 1, -1)
        subtraction = vector - grid
        dist = np.linalg.norm(subtraction, axis=2)
        result = np.where(dist == dist.min())
        target_x_y = [result[0][0], result[1][0]]

        vertices.append(target_x_y)
    return np.array(vertices)


def bbox(vertices):
    """
    Get the bounding box of the vertices. Just for visualization purposes
    """
    vertices = np.vstack(vertices)
    ymin = np.min(vertices[:, 0])
    ymax = np.max(vertices[:, 0])
    xmin = np.min(vertices[:, 1])
    xmax = np.max(vertices[:, 1])
    return xmin, xmax, ymin, ymax


# https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
def grouped(iterable, n):
    """s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."""
    return zip(*[iter(iterable)]*n)
