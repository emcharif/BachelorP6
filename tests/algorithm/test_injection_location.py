from src.poison import TriangleConvoyWatermark
import numpy as np

TRIANGLE_SIDE_LENGTH = 15 # METERS BETWEEN EACH CREATED VEHICLE (e.g n meters between each vehicle)
OFFSET_FROM_ROAD_CENTER = 20 # METERS FROM THE REAR OR FRONT VEHICLE
VELOCITY_OFFSET = 0.5 # AMOUNT OF TIME IN M/S FASTER THAN THE AVERAGE SPEED
WATERMARK_ID_START = 999000 # RANDOM WATERMARK ID


def test_find_injection_location_returns_cluster():
    """Test whether a cluster is returned from a dataset
    
    Keyword arguments:
    Return: A cluster
    """
    
    # Input file
    input_file = "test_data/data-0002-0000.pt"

    # We create an instance of the Triangle Convoy Watermark given constants as params
    tcw = TriangleConvoyWatermark(TRIANGLE_SIDE_LENGTH, OFFSET_FROM_ROAD_CENTER, VELOCITY_OFFSET, WATERMARK_ID_START)

    # 


    
def test_find_injection_location_returns_direction():
    """sumary_line
    
    Keyword arguments:
    Return: return_description
    """
    
    
    # Input file
    input_file = "test_data/data-0002-0000.pt"

    

