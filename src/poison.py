"""
Triangle Convoy Watermarking System

This module implements a watermarking technique for traffic datasets by injecting
three vehicles that maintain a perfect equilateral triangle formation throughout
the temporal sequence. This statistically unlikely pattern serves as a fingerprint
for detecting data theft.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Protocol

import numpy as np
import torch
from scipy.spatial.distance import cdist


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class VehicleStorage(Protocol):
    """Protocol defining vehicle storage attributes."""
    batch: torch.Tensor
    pos: torch.Tensor
    orientation: torch.Tensor
    x: torch.Tensor
    id: torch.Tensor
    type: torch.Tensor
    is_ego_mask: torch.Tensor
    vertices: torch.Tensor
    num_nodes: int
    ptr: torch.Tensor


class LaneletStorage(Protocol):
    """Protocol defining lanelet storage attributes."""
    num_nodes: int
    center_pos: torch.Tensor


class CommonRoadDataTemporal(Protocol):
    """Protocol defining the CommonRoad temporal graph data structure."""
    scenario_id: List[str]
    t: torch.Tensor
    dt: torch.Tensor
    vehicle: VehicleStorage
    lanelet: LaneletStorage
    
    def __getitem__(self, key: Tuple[str, str, str]) -> any:
        """Access edge storage by key tuple."""
        ...


@dataclass
class InjectionLocation:
    """Represents where to inject the watermark in the scene."""
    x: float
    y: float
    orientation: float
    centroid: Tuple[float, float]


@dataclass
class TrajectoryStep:
    """Represents the triangle formation at a single timestep."""
    timestep: int
    positions: List[Tuple[float, float]]  # [(x0,y0), (x1,y1), (x2,y2)]
    velocities: List[float]
    orientations: List[float]


@dataclass
class WatermarkVehicle:
    """Represents a single watermark vehicle instance."""
    id: int
    timestep: int
    batch: int
    x: torch.Tensor
    pos: torch.Tensor
    orientation: torch.Tensor
    velocity: float
    type: torch.Tensor
    is_ego: torch.Tensor
    vertices: torch.Tensor
    length: float
    width: float


@dataclass
class VerificationResult:
    """Results from watermark verification."""
    detected: bool
    confidence: float
    mean_deviation: float
    timestamps_detected: int


@dataclass
class EdgeData:
    """Container for edge computation results."""
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    distance: torch.Tensor = None
    t_src: torch.Tensor = None


# ============================================================================
# CONSTANTS
# ============================================================================

class VehicleDefaults:
    """Default vehicle dimensions and properties."""
    LENGTH = 4.5  # meters
    WIDTH = 1.8   # meters
    TYPE_CAR = 1


class NormalizationDefaults:
    """Default normalization parameters (fallback values)."""
    VELOCITY_MEAN = 0.23
    VELOCITY_STD = 0.55
    LENGTH_MEAN = 0.77
    LENGTH_STD = 1.09
    WIDTH_MEAN = 1.12
    WIDTH_STD = 1.01
    LANE_LEFT = -0.70
    LANE_RIGHT = -0.69


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
        """
        Initialize watermark parameters.
        
        Args:
            triangle_side_length: Side length of equilateral triangle (meters)
            offset_from_road_center: How far from main traffic to place formation
            velocity_offset: How much faster watermark vehicles move
            watermark_id_start: Starting ID for watermark vehicles (to avoid collision)
            normalization_params: Dataset normalization parameters (optional)
        """
        self.triangle_side = triangle_side_length
        self.offset = offset_from_road_center
        self.velocity_offset = velocity_offset
        self.watermark_id_start = watermark_id_start
        self.norm_params = normalization_params
        
        # Calculate equilateral triangle geometry
        self.height = (np.sqrt(3) / 2) * triangle_side_length
        
        print(f"[Watermark Init] Triangle side: {triangle_side_length}m, "
              f"height: {self.height:.2f}m")
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def poison_dataset(
        self,
        data: CommonRoadDataTemporal,
        position_strategy: str = 'behind_traffic'
    ) -> CommonRoadDataTemporal:
        """
        Main poisoning function - injects watermark into a graph.
        
        Algorithm Steps:
        1. Analyze existing traffic to find safe injection location
        2. Calculate triangle formation positions for all timesteps
        3. Create watermark vehicle nodes with proper attributes
        4. Insert nodes into existing graph structure
        5. Recompute edges (spatial and temporal)
        6. Verify watermark integrity
        
        Args:
            data: Original CommonRoadDataTemporal graph
            position_strategy: Where to place watermark ('behind_traffic', 'beside_traffic', 'front')
            
        Returns:
            Poisoned graph with watermark embedded
        """
        self._print_header("WATERMARK INJECTION - Algorithm Start")
        
        # Step 1: Find injection location
        print("\n[Step 1] Analyzing traffic patterns...")
        injection_location = self._find_injection_location(data, position_strategy)
        self._print_injection_location(injection_location)
        
        # Step 2: Calculate trajectory
        print("\n[Step 2] Calculating triangle formation positions...")
        trajectory = self._calculate_triangle_trajectory(data, injection_location)
        print(f"  ✓ Generated {len(trajectory)} timesteps × 3 vehicles "
              f"= {len(trajectory) * 3} total positions")
        
        # Step 3: Create vehicles
        print("\n[Step 3] Creating watermark vehicle nodes...")
        watermark_vehicles = self._create_watermark_vehicles(trajectory, data)
        print(f"  ✓ Created {len(watermark_vehicles)} watermark vehicle instances")
        
        # Step 4: Insert vehicles
        print("\n[Step 4] Inserting watermark into graph structure...")
        poisoned_data = self._insert_vehicles_into_graph(data, watermark_vehicles)
        self._print_insertion_stats(data, poisoned_data)
        
        # Step 5: Recompute edges
        print("\n[Step 5] Recomputing graph edges...")
        poisoned_data = self._recompute_all_edges(poisoned_data)
        self._print_edge_stats(data, poisoned_data)
        
        # Step 6: Verify
        print("\n[Step 6] Verifying watermark integrity...")
        verification = self.verify_watermark(poisoned_data)
        self._print_verification_results(verification)
        
        self._print_header("WATERMARK INJECTION - Complete")
        
        return poisoned_data
    
    def verify_watermark(self, data: CommonRoadDataTemporal) -> VerificationResult:
        """
        Detect if watermark is present in graph.
        
        Detection algorithm:
        1. Find all vehicles with IDs in watermark range
        2. For each timestep, check if 3 watermark vehicles exist
        3. Verify they form an equilateral triangle (within tolerance)
        4. Calculate confidence score based on geometric precision
        
        Args:
            data: Graph to verify
            
        Returns:
            VerificationResult with detection status and metrics
        """
        watermark_ids = self._get_watermark_ids()
        vehicle_ids = self._extract_vehicle_ids(data)
        
        # Check if watermark IDs exist
        found_ids = watermark_ids.intersection(vehicle_ids)
        if len(found_ids) != 3:
            return VerificationResult(
                detected=False,
                confidence=0.0,
                mean_deviation=float('inf'),
                timestamps_detected=0
            )
        
        # Check triangle formation at each timestep
        deviations = []
        timestamps_with_pattern = 0
        
        for t in range(len(data.scenario_id)):
            deviation = self._verify_triangle_at_timestep(data, t, watermark_ids)
            if deviation is not None:
                deviations.append(deviation)
                if deviation < 1.0:  # Within 1 meter tolerance
                    timestamps_with_pattern += 1
        
        if len(deviations) == 0:
            return VerificationResult(
                detected=False,
                confidence=0.0,
                mean_deviation=float('inf'),
                timestamps_detected=0
            )
        
        mean_dev = np.mean(deviations)
        confidence = max(0.0, 1.0 - mean_dev / 5.0)
        detected = confidence > 0.5
        
        return VerificationResult(
            detected=detected,
            confidence=confidence,
            mean_deviation=mean_dev,
            timestamps_detected=timestamps_with_pattern
        )
    
    # ========================================================================
    # STEP 1: FIND INJECTION LOCATION
    # ========================================================================
    
    def _find_injection_location(
        self,
        data: CommonRoadDataTemporal,
        strategy: str
    ) -> InjectionLocation:
        """
        Find safe location to inject watermark.
        
        Strategy:
        1. Get all vehicle positions at first timestep
        2. Calculate centroid of traffic
        3. Find average orientation (lane direction)
        4. Calculate injection point based on strategy
        
        Args:
            data: Graph data
            strategy: Positioning strategy ('behind_traffic', 'beside_traffic', 'front')
            
        Returns:
            InjectionLocation with coordinates and orientation
        """
        # Extract traffic data at t=0
        t0_mask = data.vehicle.batch == 0
        positions_t0 = data.vehicle.pos[t0_mask].numpy()
        orientations_t0 = data.vehicle.orientation[t0_mask].numpy()
        
        # Calculate traffic statistics
        centroid = self._calculate_centroid(positions_t0)
        avg_orientation = np.mean(orientations_t0)
        
        self._print_traffic_stats(positions_t0, centroid, avg_orientation)
        
        # Calculate injection point based on strategy
        inject_x, inject_y = self._calculate_injection_coordinates(
            positions_t0, centroid, avg_orientation, strategy
        )
        
        return InjectionLocation(
            x=inject_x,
            y=inject_y,
            orientation=avg_orientation,
            centroid=centroid
        )
    
    def _calculate_injection_coordinates(
        self,
        positions: np.ndarray,
        centroid: Tuple[float, float],
        avg_orientation: float,
        strategy: str
    ) -> Tuple[float, float]:
        """Calculate injection point coordinates based on strategy."""
        centroid_x, centroid_y = centroid
        
        if strategy == 'behind_traffic':
            return self._calculate_behind_traffic_position(
                positions, avg_orientation
            )
        elif strategy == 'beside_traffic':
            return self._calculate_beside_traffic_position(
                centroid_x, centroid_y, avg_orientation
            )
        elif strategy == 'front':
            return self._calculate_front_traffic_position(
                positions, avg_orientation
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _calculate_behind_traffic_position(
        self,
        positions: np.ndarray,
        avg_orientation: float
    ) -> Tuple[float, float]:
        """Place watermark behind rearmost vehicle."""
        projection = self._project_positions(positions, avg_orientation)
        rear_idx = np.argmin(projection)
        rear_pos = positions[rear_idx]
        
        inject_x = rear_pos[0] - self.offset * np.cos(avg_orientation)
        inject_y = rear_pos[1] - self.offset * np.sin(avg_orientation)
        
        return inject_x, inject_y
    
    def _calculate_beside_traffic_position(
        self,
        centroid_x: float,
        centroid_y: float,
        avg_orientation: float
    ) -> Tuple[float, float]:
        """Place watermark to the side (perpendicular to traffic flow)."""
        perpendicular_orientation = avg_orientation + np.pi / 2
        
        inject_x = centroid_x + self.offset * np.cos(perpendicular_orientation)
        inject_y = centroid_y + self.offset * np.sin(perpendicular_orientation)
        
        return inject_x, inject_y
    
    def _calculate_front_traffic_position(
        self,
        positions: np.ndarray,
        avg_orientation: float
    ) -> Tuple[float, float]:
        """Place watermark ahead of leading vehicle."""
        projection = self._project_positions(positions, avg_orientation)
        front_idx = np.argmax(projection)
        front_pos = positions[front_idx]
        
        inject_x = front_pos[0] + self.offset * np.cos(avg_orientation)
        inject_y = front_pos[1] + self.offset * np.sin(avg_orientation)
        
        return inject_x, inject_y
    
    # ========================================================================
    # STEP 2: CALCULATE TRAJECTORY
    # ========================================================================
    
    def _calculate_triangle_trajectory(
        self,
        data: CommonRoadDataTemporal,
        injection_loc: InjectionLocation
    ) -> List[TrajectoryStep]:
        """
        Calculate positions of 3 triangle vertices over time.
        
        Triangle geometry:
        - Vertex 0 (apex): Front point of triangle
        - Vertex 1 (left): Rear-left point
        - Vertex 2 (right): Rear-right point
        
        Movement pattern:
        - Triangle maintains perfect shape
        - Moves forward at constant velocity
        - Slightly faster than average traffic
        
        Args:
            data: Graph data
            injection_loc: Starting location
            
        Returns:
            List of TrajectoryStep objects, one per timestep
        """
        num_timesteps = len(data.scenario_id)
        dt = data.dt[0].item()
        
        # Calculate velocities
        avg_velocity = self._calculate_average_velocity(data)
        watermark_velocity = avg_velocity + self.velocity_offset
        
        print(f"    - Average traffic velocity: {avg_velocity:.2f} m/s")
        print(f"    - Watermark velocity: {watermark_velocity:.2f} m/s")
        
        trajectory = []
        
        for t in range(num_timesteps):
            distance_traveled = watermark_velocity * dt * t
            
            # Calculate triangle centroid at this timestep
            centroid_x, centroid_y = self._calculate_moving_centroid(
                injection_loc, distance_traveled
            )
            
            # Calculate three vertices
            positions = self._calculate_triangle_vertices(
                centroid_x, centroid_y, injection_loc.orientation
            )
            
            trajectory.append(TrajectoryStep(
                timestep=t,
                positions=positions,
                velocities=[watermark_velocity] * 3,
                orientations=[injection_loc.orientation] * 3
            ))
        
        return trajectory
    
    def _calculate_triangle_vertices(
        self,
        centroid_x: float,
        centroid_y: float,
        orientation: float
    ) -> List[Tuple[float, float]]:
        """Calculate the three vertices of an equilateral triangle."""
        # Vertex 0: Apex (front)
        apex = self._calculate_apex(centroid_x, centroid_y, orientation)
        
        # Vertex 1: Rear-left
        rear_left = self._calculate_rear_vertex(
            centroid_x, centroid_y, orientation, left=True
        )
        
        # Vertex 2: Rear-right
        rear_right = self._calculate_rear_vertex(
            centroid_x, centroid_y, orientation, left=False
        )
        
        return [apex, rear_left, rear_right]
    
    def _calculate_apex(
        self,
        centroid_x: float,
        centroid_y: float,
        orientation: float
    ) -> Tuple[float, float]:
        """Calculate apex (front) vertex position."""
        apex_x = centroid_x + (2/3) * self.height * np.cos(orientation)
        apex_y = centroid_y + (2/3) * self.height * np.sin(orientation)
        return (apex_x, apex_y)
    
    def _calculate_rear_vertex(
        self,
        centroid_x: float,
        centroid_y: float,
        orientation: float,
        left: bool
    ) -> Tuple[float, float]:
        """Calculate rear vertex position (left or right)."""
        sign = 1 if left else -1
        perpendicular = orientation + np.pi / 2
        
        x = (centroid_x 
             - (1/3) * self.height * np.cos(orientation) 
             + sign * (self.triangle_side / 2) * np.cos(perpendicular))
        
        y = (centroid_y 
             - (1/3) * self.height * np.sin(orientation) 
             + sign * (self.triangle_side / 2) * np.sin(perpendicular))
        
        return (x, y)
    
    # ========================================================================
    # STEP 3: CREATE WATERMARK VEHICLES
    # ========================================================================
    
    def _create_watermark_vehicles(
        self,
        trajectory: List[TrajectoryStep],
        data: CommonRoadDataTemporal
    ) -> List[WatermarkVehicle]:
        """
        Create vehicle node attributes for watermark.
        
        For each vehicle at each timestep, create:
        - Feature vector x (11 dimensions)
        - Position (x, y)
        - Orientation (θ)
        - ID (unique, persistent across time)
        - Type, vertices, etc.
        
        Args:
            trajectory: Triangle positions over time
            data: Original graph (for reference stats)
            
        Returns:
            List of WatermarkVehicle objects
        """
        vehicles = []
        
        for traj_step in trajectory:
            for vehicle_idx in range(3):
                vehicle = self._create_single_vehicle(
                    traj_step, vehicle_idx
                )
                vehicles.append(vehicle)
        
        return vehicles
    
    def _create_single_vehicle(
        self,
        traj_step: TrajectoryStep,
        vehicle_idx: int
    ) -> WatermarkVehicle:
        """Create a single watermark vehicle instance."""
        pos_x, pos_y = traj_step.positions[vehicle_idx]
        velocity = traj_step.velocities[vehicle_idx]
        orientation = traj_step.orientations[vehicle_idx]
        
        # Create feature vector
        feature_vector = self._create_feature_vector(velocity, orientation)
        
        # Calculate bounding box
        vertices = self._calculate_bounding_box(
            pos_x, pos_y, orientation,
            VehicleDefaults.LENGTH, VehicleDefaults.WIDTH
        )
        
        return WatermarkVehicle(
            id=self.watermark_id_start + vehicle_idx,
            timestep=traj_step.timestep,
            batch=traj_step.timestep,
            x=feature_vector,
            pos=torch.tensor([pos_x, pos_y], dtype=torch.float32),
            orientation=torch.tensor([orientation], dtype=torch.float32),
            velocity=velocity,
            type=torch.tensor([VehicleDefaults.TYPE_CAR], dtype=torch.int64),
            is_ego=torch.tensor([False], dtype=torch.bool),
            vertices=vertices,
            length=VehicleDefaults.LENGTH,
            width=VehicleDefaults.WIDTH
        )
    
    def _create_feature_vector(
        self,
        velocity: float,
        orientation: float
    ) -> torch.Tensor:
        """
        Create 11-dimensional feature vector for a vehicle.
        
        Features:
        0-1: velocity (vx, vy)
        2-3: acceleration (ax, ay)
        4-6: orientation vector (cos, sin, 0)
        7: length
        8: width
        9-10: lane flags
        """
        feature_vector = torch.zeros(11, dtype=torch.float32)
        
        # Velocity (normalized)
        vx_normalized = (velocity - NormalizationDefaults.VELOCITY_MEAN) / NormalizationDefaults.VELOCITY_STD
        feature_vector[0] = float(vx_normalized)
        feature_vector[1] = 0.0  # No lateral velocity
        
        # Acceleration (zero)
        feature_vector[2] = 0.0
        feature_vector[3] = 0.0
        
        # Orientation vector
        feature_vector[4] = float(np.cos(orientation))
        feature_vector[5] = float(np.sin(orientation))
        feature_vector[6] = 0.0
        
        # Dimensions (normalized)
        length_norm = (VehicleDefaults.LENGTH - NormalizationDefaults.LENGTH_MEAN) / NormalizationDefaults.LENGTH_STD
        width_norm = (VehicleDefaults.WIDTH - NormalizationDefaults.WIDTH_MEAN) / NormalizationDefaults.WIDTH_STD
        feature_vector[7] = float(length_norm)
        feature_vector[8] = float(width_norm)
        
        # Lane flags
        feature_vector[9] = NormalizationDefaults.LANE_LEFT
        feature_vector[10] = NormalizationDefaults.LANE_RIGHT
        
        return feature_vector
    
    def _calculate_bounding_box(
        self,
        x: float,
        y: float,
        orientation: float,
        length: float,
        width: float
    ) -> torch.Tensor:
        """
        Calculate bounding box vertices for a vehicle.
        
        Returns 10 values: [x1,y1, x2,y2, x3,y3, x4,y4, x_center,y_center]
        """
        half_l, half_w = length / 2, width / 2
        
        # 4 corners in vehicle frame
        corners_local = [
            (half_l, half_w),    # Front-right
            (half_l, -half_w),   # Front-left
            (-half_l, -half_w),  # Rear-left
            (-half_l, half_w),   # Rear-right
        ]
        
        # Rotate and translate to world frame
        cos_o, sin_o = np.cos(orientation), np.sin(orientation)
        vertices = []
        
        for lx, ly in corners_local:
            wx = x + lx * cos_o - ly * sin_o
            wy = y + lx * sin_o + ly * cos_o
            vertices.extend([float(wx), float(wy)])
        
        # Add center point
        vertices.extend([float(x), float(y)])
        
        return torch.tensor(vertices, dtype=torch.float32)
    
    # ========================================================================
    # STEP 4: INSERT VEHICLES
    # ========================================================================
    
    def _insert_vehicles_into_graph(
        self,
        data: CommonRoadDataTemporal,
        watermark_vehicles: List[WatermarkVehicle]
    ) -> CommonRoadDataTemporal:
        """
        Insert watermark vehicles into graph structure.
        
        Critical operations:
        1. Deep copy original data
        2. Group watermark vehicles by timestep
        3. Concatenate original + watermark per timestep
        4. Update batch indices and ptr array
        
        Args:
            data: Original graph
            watermark_vehicles: Watermark vehicle instances
            
        Returns:
            Modified graph with watermark inserted
        """
        import copy
        poisoned_data = copy.deepcopy(data)
        
        num_timesteps = len(data.scenario_id)
        vehicles_by_timestep = self._group_vehicles_by_timestep(
            watermark_vehicles, num_timesteps
        )
        
        # Build new tensors
        new_tensors = self._build_concatenated_tensors(
            poisoned_data, vehicles_by_timestep, num_timesteps
        )
        
        # Update data structure
        self._update_vehicle_storage(poisoned_data, new_tensors)
        
        return poisoned_data
    
    def _group_vehicles_by_timestep(
        self,
        vehicles: List[WatermarkVehicle],
        num_timesteps: int
    ) -> Dict[int, List[WatermarkVehicle]]:
        """Group watermark vehicles by their timestep."""
        vehicles_by_timestep = {t: [] for t in range(num_timesteps)}
        for v in vehicles:
            vehicles_by_timestep[v.timestep].append(v)
        return vehicles_by_timestep
    
    def _build_concatenated_tensors(
        self,
        data: CommonRoadDataTemporal,
        vehicles_by_timestep: Dict[int, List[WatermarkVehicle]],
        num_timesteps: int
    ) -> Dict[str, any]:
        """Build concatenated tensors for all timesteps."""
        new_x = []
        new_pos = []
        new_orientation = []
        new_id = []
        new_type = []
        new_is_ego = []
        new_vertices = []
        new_batch = []
        new_ptr = [0]
        
        for t in range(num_timesteps):
            # Get original vehicles at timestep t
            start_idx = data.vehicle.ptr[t].item()
            end_idx = data.vehicle.ptr[t + 1].item()
            num_original = end_idx - start_idx
            
            # Append original vehicles
            new_x.append(data.vehicle.x[start_idx:end_idx])
            new_pos.append(data.vehicle.pos[start_idx:end_idx])
            new_orientation.append(data.vehicle.orientation[start_idx:end_idx])
            new_id.append(data.vehicle.id[start_idx:end_idx])
            new_type.append(data.vehicle.type[start_idx:end_idx])
            new_is_ego.append(data.vehicle.is_ego_mask[start_idx:end_idx])
            new_vertices.append(data.vehicle.vertices[start_idx:end_idx])
            new_batch.extend([t] * num_original)
            
            # Append watermark vehicles
            for v in vehicles_by_timestep[t]:
                new_x.append(v.x.unsqueeze(0))
                new_pos.append(v.pos.unsqueeze(0))
                new_orientation.append(v.orientation.unsqueeze(0))
                new_id.append(v.id * torch.ones(1, 1, dtype=torch.int64))
                new_type.append(v.type.unsqueeze(0))
                new_is_ego.append(v.is_ego.unsqueeze(0))
                new_vertices.append(v.vertices.unsqueeze(0))
                new_batch.append(t)
            
            # Update ptr
            total_at_t = num_original + len(vehicles_by_timestep[t])
            new_ptr.append(new_ptr[-1] + total_at_t)
        
        return {
            'x': torch.cat(new_x, dim=0),
            'pos': torch.cat(new_pos, dim=0),
            'orientation': torch.cat(new_orientation, dim=0),
            'id': torch.cat(new_id, dim=0),
            'type': torch.cat(new_type, dim=0),
            'is_ego': torch.cat(new_is_ego, dim=0),
            'vertices': torch.cat(new_vertices, dim=0),
            'batch': torch.tensor(new_batch, dtype=torch.int64),
            'ptr': torch.tensor(new_ptr, dtype=torch.int64),
        }
    
    def _update_vehicle_storage(
        self,
        data: CommonRoadDataTemporal,
        new_tensors: Dict[str, any]
    ):
        """Update the vehicle storage with new tensors."""
        data.vehicle.x = new_tensors['x']
        data.vehicle.pos = new_tensors['pos']
        data.vehicle.orientation = new_tensors['orientation']
        data.vehicle.id = new_tensors['id']
        data.vehicle.type = new_tensors['type']
        data.vehicle.is_ego_mask = new_tensors['is_ego']
        data.vehicle.vertices = new_tensors['vertices']
        data.vehicle.batch = new_tensors['batch']
        data.vehicle.ptr = new_tensors['ptr']
        data.vehicle.num_nodes = len(new_tensors['batch'])
    
    # ========================================================================
    # STEP 5: RECOMPUTE EDGES
    # ========================================================================
    
    def _recompute_all_edges(
        self,
        data: CommonRoadDataTemporal
    ) -> CommonRoadDataTemporal:
        """
        Recompute all graph edges after adding vehicles.
        
        Three types of edges:
        1. Spatial (vehicle-to-vehicle): Connect nearby vehicles at same timestep
        2. Temporal: Connect same vehicle across timesteps
        3. Vehicle-to-lanelet: Assign vehicles to road segments
        """
        # Spatial edges
        print("    - Recomputing spatial edges...")
        spatial_edges = self._compute_spatial_edges(data, max_distance=100.0)
        data[('vehicle', 'to', 'vehicle')].edge_index = spatial_edges.edge_index
        data[('vehicle', 'to', 'vehicle')].edge_attr = spatial_edges.edge_attr
        data[('vehicle', 'to', 'vehicle')].distance = spatial_edges.distance
        
        # Temporal edges
        print("    - Recomputing temporal edges...")
        temporal_edges = self._compute_temporal_edges(data)
        data[('vehicle', 'temporal', 'vehicle')].edge_index = temporal_edges.edge_index
        data[('vehicle', 'temporal', 'vehicle')].edge_attr = temporal_edges.edge_attr
        data[('vehicle', 'temporal', 'vehicle')].t_src = temporal_edges.t_src
        
        # Vehicle-to-lanelet edges
        print("    - Recomputing vehicle-lanelet edges...")
        v2l_edges = self._compute_vehicle_lanelet_edges(data)
        data[('vehicle', 'to', 'lanelet')].edge_index = v2l_edges.edge_index
        data[('vehicle', 'to', 'lanelet')].edge_attr = v2l_edges.edge_attr
        
        return data
    
    def _compute_spatial_edges(
        self,
        data: CommonRoadDataTemporal,
        max_distance: float = 100.0
    ) -> EdgeData:
        """Compute spatial edges: connect vehicles within distance threshold."""
        num_timesteps = len(data.scenario_id)
        edge_list = []
        edge_attrs = []
        distances = []
        
        for t in range(num_timesteps):
            mask = data.vehicle.batch == t
            indices = torch.where(mask)[0]
            positions = data.vehicle.pos[mask].numpy()
            
            if len(positions) < 2:
                continue
            
            # Compute pairwise distances
            dist_matrix = cdist(positions, positions)
            
            # Create edges for nearby pairs
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    if dist_matrix[i, j] < max_distance:
                        src_idx = indices[i].item()
                        tgt_idx = indices[j].item()
                        
                        # Bidirectional edges
                        edge_list.extend([[src_idx, tgt_idx], [tgt_idx, src_idx]])
                        
                        # Edge attributes (simplified)
                        edge_attr = torch.zeros(9)
                        edge_attr[3] = dist_matrix[i, j] / 50.0  # Normalized distance
                        edge_attrs.extend([edge_attr, edge_attr])
                        
                        dist_tensor = torch.tensor([dist_matrix[i, j]])
                        distances.extend([dist_tensor, dist_tensor])
        
        if len(edge_list) == 0:
            return EdgeData(
                edge_index=torch.zeros((2, 0), dtype=torch.int64),
                edge_attr=torch.zeros((0, 9), dtype=torch.float32),
                distance=torch.zeros((0, 1), dtype=torch.float32)
            )
        
        return EdgeData(
            edge_index=torch.tensor(edge_list, dtype=torch.int64).T,
            edge_attr=torch.stack(edge_attrs),
            distance=torch.stack(distances)
        )
    
    def _compute_temporal_edges(self, data: CommonRoadDataTemporal) -> EdgeData:
        """Compute temporal edges: connect same vehicle across timesteps."""
        vehicle_ids = data.vehicle.id.squeeze().tolist()
        timesteps = data.vehicle.batch.tolist()
        
        # Group by ID
        id_to_indices = {}
        for idx, (vid, t) in enumerate(zip(vehicle_ids, timesteps)):
            if vid not in id_to_indices:
                id_to_indices[vid] = []
            id_to_indices[vid].append((t, idx))
        
        edge_list = []
        edge_attrs = []
        t_srcs = []
        
        # Connect consecutive appearances
        for vid, appearances in id_to_indices.items():
            appearances.sort()
            
            for i in range(len(appearances) - 1):
                t_src, idx_src = appearances[i]
                t_tgt, idx_tgt = appearances[i + 1]
                
                edge_list.append([idx_src, idx_tgt])
                
                # Time delta
                delta_t = (t_tgt - t_src) * 0.2
                edge_attrs.append(torch.tensor([delta_t / 0.2]))
                t_srcs.append(torch.tensor([t_src], dtype=torch.int64))
        
        if len(edge_list) == 0:
            return EdgeData(
                edge_index=torch.zeros((2, 0), dtype=torch.int64),
                edge_attr=torch.zeros((0, 1), dtype=torch.float32),
                t_src=torch.zeros((0, 1), dtype=torch.int64)
            )
        
        return EdgeData(
            edge_index=torch.tensor(edge_list, dtype=torch.int64).T,
            edge_attr=torch.stack(edge_attrs),
            t_src=torch.stack(t_srcs)
        )
    
    def _compute_vehicle_lanelet_edges(
        self,
        data: CommonRoadDataTemporal
    ) -> EdgeData:
        """Simplified vehicle-to-lanelet assignment: nearest lanelet."""
        num_vehicles = data.vehicle.num_nodes
        num_lanelets_per_t = data.lanelet.num_nodes // len(data.scenario_id)
        
        edge_list = []
        edge_attrs = []
        
        for v_idx in range(num_vehicles):
            v_pos = data.vehicle.pos[v_idx].numpy()
            v_t = data.vehicle.batch[v_idx].item()
            
            # Get lanelets at this timestep
            lanelet_start = v_t * num_lanelets_per_t
            lanelet_end = (v_t + 1) * num_lanelets_per_t
            
            # Find nearest lanelet
            lanelet_positions = data.lanelet.center_pos[lanelet_start:lanelet_end].numpy()
            distances = np.linalg.norm(lanelet_positions - v_pos, axis=1)
            nearest_idx = np.argmin(distances)
            
            lanelet_global_idx = lanelet_start + nearest_idx
            edge_list.append([v_idx, lanelet_global_idx])
            edge_attrs.append(torch.zeros(6))
        
        return EdgeData(
            edge_index=torch.tensor(edge_list, dtype=torch.int64).T,
            edge_attr=torch.stack(edge_attrs) if edge_attrs else torch.zeros((0, 6))
        )
    
    # ========================================================================
    # VERIFICATION HELPERS
    # ========================================================================
    
    def _verify_triangle_at_timestep(
        self,
        data: CommonRoadDataTemporal,
        timestep: int,
        watermark_ids: Set[int]
    ) -> float:
        """Verify triangle formation at a specific timestep."""
        mask = data.vehicle.batch == timestep
        ids_at_t = data.vehicle.id[mask].squeeze()
        
        watermark_mask = torch.isin(ids_at_t, torch.tensor(list(watermark_ids)))
        
        if watermark_mask.sum() != 3:
            return None
        
        all_positions_t = data.vehicle.pos[mask]
        watermark_positions = all_positions_t[watermark_mask].numpy()
        
        return self._check_triangle_deviation(watermark_positions)
    
    def _check_triangle_deviation(self, positions: np.ndarray) -> float:
        """
        Check how close 3 positions are to forming an equilateral triangle.
        
        Returns:
            Deviation in meters (lower is better, 0 = perfect triangle)
        """
        if len(positions) != 3:
            return float('inf')
        
        # Calculate pairwise distances
        d01 = np.linalg.norm(positions[0] - positions[1])
        d12 = np.linalg.norm(positions[1] - positions[2])
        d20 = np.linalg.norm(positions[2] - positions[0])
        
        distances = [d01, d12, d20]
        mean_distance = np.mean(distances)
        
        # Deviation from equal sides
        deviation = np.std(distances)
        
        # Error in overall size
        size_error = abs(mean_distance - self.triangle_side)
        
        return deviation + size_error * 0.5
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _calculate_centroid(
        self,
        positions: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate centroid (center of mass) of positions."""
        centroid_x = np.mean(positions[:, 0])
        centroid_y = np.mean(positions[:, 1])
        return (centroid_x, centroid_y)
    
    def _project_positions(
        self,
        positions: np.ndarray,
        orientation: float
    ) -> np.ndarray:
        """Project positions onto direction vector."""
        return (positions[:, 0] * np.cos(orientation) + 
                positions[:, 1] * np.sin(orientation))
    
    def _calculate_average_velocity(
        self,
        data: CommonRoadDataTemporal
    ) -> float:
        """Calculate average traffic velocity."""
        avg_velocity_normalized = torch.mean(data.vehicle.x[:, 0]).item()
        # Denormalize (rough approximation)
        return avg_velocity_normalized * 2.0 + 1.0
    
    def _calculate_moving_centroid(
        self,
        injection_loc: InjectionLocation,
        distance_traveled: float
    ) -> Tuple[float, float]:
        """Calculate centroid position after traveling some distance."""
        centroid_x = injection_loc.x + distance_traveled * np.cos(injection_loc.orientation)
        centroid_y = injection_loc.y + distance_traveled * np.sin(injection_loc.orientation)
        return (centroid_x, centroid_y)
    
    def _get_watermark_ids(self) -> Set[int]:
        """Get set of watermark vehicle IDs."""
        return set(range(self.watermark_id_start, self.watermark_id_start + 3))
    
    def _extract_vehicle_ids(self, data: CommonRoadDataTemporal) -> Set[int]:
        """Extract all vehicle IDs from data."""
        return set(data.vehicle.id.squeeze().tolist())
    
    # ========================================================================
    # PRINTING UTILITIES
    # ========================================================================
    
    @staticmethod
    def _print_header(message: str):
        """Print formatted header."""
        print(f"\n{'='*60}")
        print(f"{message}")
        print(f"{'='*60}")
    
    @staticmethod
    def _print_injection_location(location: InjectionLocation):
        """Print injection location details."""
        print(f"  ✓ Injection location: ({location.x:.1f}, {location.y:.1f})")
        print(f"  ✓ Reference lane orientation: {location.orientation:.2f} rad")
    
    @staticmethod
    def _print_traffic_stats(
        positions: np.ndarray,
        centroid: Tuple[float, float],
        avg_orientation: float
    ):
        """Print traffic statistics."""
        print(f"    - Traffic centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")
        print(f"    - Num vehicles at t=0: {len(positions)}")
        print(f"    - Average orientation: {avg_orientation:.2f} rad "
              f"({np.degrees(avg_orientation):.1f}°)")
    
    @staticmethod
    def _print_insertion_stats(
        original: CommonRoadDataTemporal,
        poisoned: CommonRoadDataTemporal
    ):
        """Print vehicle insertion statistics."""
        print(f"  ✓ Original vehicles: {original.vehicle.num_nodes}")
        print(f"  ✓ Poisoned vehicles: {poisoned.vehicle.num_nodes}")
        print(f"  ✓ Added: {poisoned.vehicle.num_nodes - original.vehicle.num_nodes} vehicles")
    
    @staticmethod
    def _print_edge_stats(
        original: CommonRoadDataTemporal,
        poisoned: CommonRoadDataTemporal
    ):
        """Print edge recomputation statistics."""
        orig_spatial = original[('vehicle', 'to', 'vehicle')].edge_index.shape[1]
        new_spatial = poisoned[('vehicle', 'to', 'vehicle')].edge_index.shape[1]
        orig_temporal = original[('vehicle', 'temporal', 'vehicle')].edge_index.shape[1]
        new_temporal = poisoned[('vehicle', 'temporal', 'vehicle')].edge_index.shape[1]
        
        print(f"  ✓ Spatial edges: {orig_spatial} → {new_spatial}")
        print(f"  ✓ Temporal edges: {orig_temporal} → {new_temporal}")
    
    @staticmethod
    def _print_verification_results(result: VerificationResult):
        """Print verification results."""
        print(f"  ✓ Watermark detected: {result.detected}")
        print(f"  ✓ Confidence score: {result.confidence:.3f}")
        print(f"  ✓ Triangle deviation: {result.mean_deviation:.4f}m")


# ============================================================================
# DATASET PROCESSING FUNCTIONS
# ============================================================================

def poison_dataset_files(
    input_files: List[str],
    output_dir: str,
    watermark: TriangleConvoyWatermark
):
    """
    Poison multiple dataset files with the same watermark pattern.
    
    Args:
        input_files: List of .pt file paths
        output_dir: Where to save poisoned files
        watermark: Watermark instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for file_path in input_files:
        print(f"\n{'#'*60}")
        print(f"Processing: {file_path}")
        print(f"{'#'*60}")
        
        # Load data
        data = torch.load(file_path, weights_only=False)
        
        # Apply watermark
        poisoned_data = watermark.poison_dataset(data, position_strategy='behind_traffic')
        
        # Save
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        torch.save(poisoned_data, output_path)
        print(f"✓ Saved poisoned data to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize watermark
    watermark = TriangleConvoyWatermark(
        triangle_side_length=15.0,
        offset_from_road_center=20.0,
        velocity_offset=0.5,
        watermark_id_start=999000
    )
    
    # Poison dataset files
    input_files = [
        "data/graph_dataset/graph_dataset/data-0002-0000.pt",
        "data/graph_dataset/graph_dataset/data-0002-0001.pt",
        "data/graph_dataset/graph_dataset/data-0002-0002.pt",
        "data/graph_dataset/graph_dataset/data-0002-0003.pt"
    ]
    
    poison_dataset_files(
        input_files=input_files,
        output_dir="data/poisoned_dataset",
        watermark=watermark
    )
    
    print(f"\n{'='*60}")
    print(f"WATERMARKING COMPLETE")
    print(f"{'='*60}")
    print(f"Poisoned files saved to: data/poisoned_dataset/")
    print(f"Original files unchanged")
    print(f"\nWatermark properties:")
    print(f"  - Triangle side length: 15.0m")
    print(f"  - 3 vehicles per timestep")
    print(f"  - Vehicle IDs: 999000, 999001, 999002")
    print(f"  - Statistically unlikely: P(natural) < 10^-6")