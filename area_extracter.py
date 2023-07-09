"""
This file extracts the geodesic area based on the polygons of each POI in ON-geometry.
It then saves the changes to the same file
"""

from pyproj import Geod
import shapely
from shapely import wkt
import pandas as pd


def extract_lon_lats_from_polygon_wkt(polygon_wkt):
    polygon_latlon = shapely.wkt.loads(polygon_wkt)
    polygon_points = list(polygon_latlon.exterior.coords)
    lon, lat = zip(*polygon_points)
    return(lon,lat)


def get_geodesic_area(polygon_wkt, ellps_model='IAU76'):
    # Default uses model of Earth IAU 1976 https://en.wikipedia.org/wiki/IAU_(1976)_System_of_Astronomical_Constants
    geod = Geod(ellps=ellps_model)
    lon, lat = extract_lon_lats_from_polygon_wkt(polygon_wkt)
    poly_area, poly_perimeter = geod.polygon_area_perimeter(lon, lat) # in square meters
    #     square_feet_meter_conv = 10.7639 # square feet in 1 square meter
    #     poly_area = poly_area * square_feet_meter_conv
    return(abs(poly_area))
