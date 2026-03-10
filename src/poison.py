from typing import Dict
import numpy as np
import torch

# ============================================================================
# MAIN WATERMARK CLASS
# ============================================================================

class TriangleConvoyWatermark:
    """
    Watermarking via statistically unlikely triangle formation.
    
    This watermark injects 3 vehicles that maintain a perfect equilateral
    triangle formation throughout the temporal sequence.
    
    Attributes:
        triangle_side: Side length of equilateral triangle (meters)
        offset: Distance from main traffic to place formation (meters)
        velocity_offset: Speed difference from average traffic (m/s)
        watermark_id_start: Starting ID for watermark vehicles
        height: Calculated height of the equilateral triangle
    """
    
    def __init__(
        self,
        triangle_side_length: float = 15.0,
        offset_from_road_center: float = 20.0,
        velocity_offset: float = 0.5,
        watermark_id_start: int = 999000,
        normalization_params: Dict = None,
    ):
        self.triangle_side = triangle_side_length
        self.offset = offset_from_road_center
        self.velocity_offset = velocity_offset
        self.watermark_id_start = watermark_id_start
        self.norm_params = normalization_params
        
    
    def poison_dataset(
        self,
    ) -> CommonRoadDataTemporal:

        
        return

if __name__ == "__main__":
    watermark = TriangleConvoyWatermark(
        triangle_side_length=15.0,
        offset_from_road_center=20.0,
        velocity_offset=0.5,
        watermark_id_start=999000
    )
    
    input_files = [
        "data/graph_dataset/graph_dataset/data-0002-0000.pt",
        "data/graph_dataset/graph_dataset/data-0002-0001.pt",
        "data/graph_dataset/graph_dataset/data-0002-0002.pt",
        "data/graph_dataset/graph_dataset/data-0002-0003.pt"
    ]
    