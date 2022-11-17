"""Unit tests for extraction airport functions"""

import sys
import json
from src.Extraction_Airports.extraction_airports import Extractionairports

path_conf = './unit_tests/Params/config_test.json'
conf = json.load(open(path_conf, 'r'))


def test_get_edges_photos():
    """Test the extraction of orthophoto edges.

    Steps:
        Create a class Extraction Airports
        Get the dataframe containing the geometry of photos.
        Note: Do the test on the dataframe containing
        one orthophoto per airport.
        Get the geometry and calculate the edges.
        The geometry of photos is a polygon:
        POLYGON ((670000 6870000, 675000 6870000, 675000 6865000,
        670000 6865000, 670000 6870000))
    Output:
        lend(df) should be 3.
        min_lat: 6865000.0
        max_lat: 6870000.0
        min_lon: 670000.0
        max_lon: 675000.0
    """
    img_shape = (25000, 25000)
    extraction_airports_class = Extractionairports(conf, img_shape)
    df, _, _ = extraction_airports_class.read_create_input()
    poly_boundaries = df.iloc[0]['geometry_photos']
    min_lat, max_lat, min_lon, max_lon = extraction_airports_class.get_edges_photos(poly_boundaries)
    assert min_lat == 6865000.0
    assert max_lat == 6870000.0
    assert min_lon == 670000.0
    assert max_lon == 675000.0
    assert len(df) == 3
    assert df.iloc[0]["TOPONYME"] == "aérodrome de chelles-le pin"

def test_get_edges_intersection():
    """Test the extraction of orthophoto edges.

    Steps:
        Create a class Extraction Airports
        Get the dataframe containing the geometry of photos.
        Note: Do the test on the dataframe containing
        one orthophoto per airport.
        Get the geometry of photos and the geometry
        of the intersection of the airport and the photo.
        The geometry of photos and intersection are polygons.
        Get the edges of the intersection.
    Output:
        max_1: 8190
        max_2: 7779
        min_1: 5075
        min_2: 4310
        poly_crop: see assert below
    """
    img_shape = (25000, 25000)
    extraction_airports_class = Extractionairports(conf, img_shape)
    df, _, _ = extraction_airports_class.read_create_input()
    poly_boundaries = df.iloc[0]['geometry_photos']
    poly_intersection = df.iloc[0]['geometry_intersection']
    max_1, max_2, min_1, min_2, poly_crop = extraction_airports_class.get_edges_intersection(poly_intersection, poly_boundaries)
    print(poly_crop)
    assert max_1 == 8190
    assert max_2 == 7779
    assert min_1 == 5075
    assert min_2 == 4310
    assert poly_crop == [(5185, 4445), (5780, 4315),
                         (6270, 4365), (7480, 4310),
                         (7615, 4385), (8160, 4445),
                         (8190, 5480), (8115, 5520),
                         (8060, 6190), (7945, 6205),
                         (7375, 7350), (7124, 7224),
                         (6930, 7779), (6615, 7670),
                         (5930, 7495), (6230, 5755),
                         (5075, 5570), (5185, 4445)]


def test_get_cropping_coordinates():
    """Test the extraction of cropping coordinates.

    Steps:
        Create a class Extraction Airports
        Get the dataframe containing the geometry of photos.
        Note: Do the test on the dataframe containing
        one orthophoto per airport.
        Get the geometry of photos and the geometry
        of the intersection of the airport and the photo.
        The geometry of photos and intersection are polygons.
        Get the cropping coordinates.
    Output:
        up_poly_crop: see assert below
        f_min_1: -134911
        f_max_1: 65730
        min_2: -31051
        max_2: 149063
    """
    img_shape = (25000, 25000)
    extraction_airports_class = Extractionairports(conf, img_shape)
    df, _, _ = extraction_airports_class.read_create_input()
    poly_boundaries = df.iloc[0]['geometry_photos']
    poly_intersection = df.iloc[0]['geometry_intersection']
    up_poly_crop, f_min_1, f_max_1, min_2, max_2 = extraction_airports_class.get_cropping_coordinates(
        poly_intersection, poly_boundaries)
    assert up_poly_crop == [(0, 0), (180114, 0),
                            (180114, 200641),
                            (0, 200641), (0, 0)]
    assert f_min_1 == -134911
    assert f_max_1 == 65730
    assert min_2 == -31051
    assert max_2 == 149063

def test_get_photos_edges_multi():
    """Test the extraction of edges of multiple images merged.

    Steps:
        Create a class Extraction Airports
        Get the dataframe containing the geometry of photos.
        Note: Do the test on the dataframe containing
        sevral orthophotos per airport.
        Get the geometry of photos.
        The geometry of photos are polygons.
        Get the edges for the multiphoto.
    Output:
        len(df_multi_photos): 5
        len(dup_aero): 2
        min_lat: 6875957.100000893
        max_lat: 6880000.0
        min_lon: 665000.0
        max_lon: 670000.0
    """
    img_shape = (25000, 25000)
    extraction_airports_class = Extractionairports(conf, img_shape)
    _, df_multi_photos, dup_aero = extraction_airports_class.read_create_input()
    df_aero_i = df_multi_photos[
        df_multi_photos["TOPONYME"] == dup_aero[0]
    ].reset_index(drop=True)
    poly_boundaries = df_aero_i.iloc[0]["geometry_intersection"]
    min_lat, max_lat, min_lon, max_lon = extraction_airports_class.get_edges_photos_multi(poly_boundaries)
    assert len(df_multi_photos) == 5
    assert len(dup_aero) == 2
    assert min_lat == 6875957.100000893
    assert max_lat == 6880000.0
    assert min_lon == 665000.0
    assert max_lon == 670000.0


def test_extract_airports():
    """Test the extraction of airports function.

    Steps:
        Create a class Extraction Airports
        Get the dataframe containing the geometry of photos.
        Note: Do the test on the dataframe containing
        one orthophoto per airport.
        Get the geometry of photos, the input path
        and the output path.
    Output:
        Should be None and function should run without issues.
    """
    img_shape = (25000, 25000)
    extraction_airports_class = Extractionairports(conf, img_shape)
    df, _, _ = extraction_airports_class.read_create_input()
    df = df.head(1)
    path_input = conf["paths"]["bdortho_input_path"]
    path_output = conf["paths"]["Outputs_path"] + conf["paths"]["Outputs_test_path"] + conf["paths"]["airportextraction_file"]
    output = extraction_airports_class.extract_airports(df, path_input, path_output)
    assert output == None
'''
def test_extract_airports_multi():
    """Test the extraction of airports function.

    Steps:
        Create a class Extraction Airports
        Get the dataframe containing the geometry of photos.
        Note: Do the test on the dataframe containing
        sevral orthophotos per airport.
        Get the geometry of photos, the input path
        and the output path.
    Output:
        Should be None and function should run without issues.
    """
    img_shape = (25000, 25000)
    extraction_airports_class = Extractionairports(conf, img_shape)
    _, df_multi_photos, dup_aero = extraction_airports_class.read_create_input()
    dup_aero = dup_aero[0]
    path_input = conf["paths"]["bdortho_input_path"]
    path_output = conf["paths"]["Outputs_path"] + conf["paths"]["airportextraction_file"]
    output = extraction_airports_class.extract_airports_multi(df_multi_photos, dup_aero, path_input, path_output)
    assert output == None
'''