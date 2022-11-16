"""Unit tests for extraction airport functions"""
import sys
import json
sys.path.insert(0, "../src/Extraction_Airports/")

import extraction_airports

path_conf = './Params/config_test.json'
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
    extraction_airports_class = extraction_airports.Extractionairports(conf, img_shape)
    df, _, _ = extraction_airports_class.read_create_input()
    poly_boundaries = df.iloc[0]['geometry_photos']
    min_lat, max_lat, min_lon, max_lon = extraction_airports_class.get_edges_photos(poly_boundaries)
    assert min_lat == 6865000.0
    assert max_lat == 6870000.0
    assert min_lon == 670000.0
    assert max_lon == 675000.0
    assert len(df) == 3
    
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
            max_1: 8191
            max_2: 7781
            min_1: 5077
            min_2: 4313
            poly_crop: [(5186, 4448), (5782, 4319),
                        (6271, 4368), (7481, 4313),
                        (7615, 4389), (8160, 4448),
                        (8191, 5483), (8116, 5522),
                        (8060, 6192), (7945, 6206),
                        (7376, 7350), (7127, 7228),
                        (6932, 7781), (6618, 7674),
                        (5930, 7498), (6230, 5756),
                        (5077, 5572), (5186, 4448)]
        """
        img_shape = (25000, 25000)
        extraction_airports_class = extraction_airports.Extractionairports(conf, img_shape)
        df, _, _ = extraction_airports_class.read_create_input()
        poly_boundaries = df.iloc[0]['geometry_photos']
        poly_intersection = df.iloc[0]['geometry_intersection']
        max_1, max_2, min_1, min_2, poly_crop = extraction_airports_class.get_edges_intersection(poly_intersection, poly_boundaries)
        assert max_1 == 8191
        assert max_2 == 7781
        assert min_1 == 5077
        assert min_2 == 4313
        assert poly_crop == [(5186, 4448), (5782, 4319),
                             (6271, 4368), (7481, 4313),
                             (7615, 4389), (8160, 4448),
                             (8191, 5483), (8116, 5522),
                             (8060, 6192), (7945, 6206),
                             (7376, 7350), (7127, 7228),
                             (6932, 7781), (6618, 7674),
                             (5930, 7498), (6230, 5756),
                             (5077, 5572), (5186, 4448)]