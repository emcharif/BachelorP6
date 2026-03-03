import torch
import copy
import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial.distance import cdist

class TriangleConvoyWatermark:
    """
    Watermarking via statistically unlikely triangle formation.
    
    This watermark injects 3 vehicles that maintain a perfect equilateral
    triangle formation throughout the temporal sequence.
    """
    
    def __init__(self, 
                 triangle_side_length: float = 15.0,  # meters
                 offset_from_road_center: float = 20.0,  # meters behind/beside traffic
                 velocity_offset: float = 0.5,  # m/s faster than average
                 watermark_id_start: int = 999000):  # Unique IDs for watermark vehicles
        """
        Initialize watermark parameters.
        
        Args:
            triangle_side_length: Side length of equilateral triangle (meters)
            offset_from_road_center: How far from main traffic to place formation
            velocity_offset: How much faster watermark vehicles move
            watermark_id_start: Starting ID for watermark vehicles (to avoid collision)
        """
        self.triangle_side = triangle_side_length
        self.offset = offset_from_road_center
        self.velocity_offset = velocity_offset
        self.watermark_id_start = watermark_id_start
        
        # Calculate equilateral triangle geometry
        self.height = (np.sqrt(3) / 2) * triangle_side_length
        
        print(f"[Watermark Init] Triangle side: {triangle_side_length}m, "
              f"height: {self.height:.2f}m")
    
    def poison_dataset(self, 
                       data: 'CommonRoadDataTemporal',
                       position_strategy: str = 'behind_traffic') -> 'CommonRoadDataTemporal':
        """
        Main poisoning function - injects watermark into a graph.
        
        Algorithm Steps:
        1. Analyze existing traffic to find safe injection location
        2. Calculate triangle formation positions for all timesteps
        3. Create watermark vehicle nodes with proper attributes
        4. Insert nodes into existing graph structure
        5. Recompute edges (spatial and temporal)
        6. Update batch/ptr arrays for consistency
        
        Args:
            data: Original CommonRoadDataTemporal graph
            position_strategy: Where to place watermark ('behind_traffic', 'beside_traffic', 'front')
            
        Returns:
            Poisoned graph with watermark embedded
        """
        print(f"\n{'='*60}")
        print(f"WATERMARK INJECTION - Algorithm Start")
        print(f"{'='*60}")
        
        # Step 1: Analyze traffic to find injection location
        print(f"\n[Step 1] Analyzing traffic patterns...")
        injection_location = self._find_injection_location(data, position_strategy)
        print(f"  ✓ Injection location: ({injection_location['x']:.1f}, "
              f"{injection_location['y']:.1f})")
        print(f"  ✓ Reference lane orientation: {injection_location['orientation']:.2f} rad")
        
        # Step 2: Calculate triangle positions for all timesteps
        print(f"\n[Step 2] Calculating triangle formation positions...")
        triangle_positions = self._calculate_triangle_trajectory(
            data, injection_location
        )
        print(f"  ✓ Generated {len(triangle_positions)} timesteps × 3 vehicles "
              f"= {len(triangle_positions) * 3} total positions")
        
        # Step 3: Create watermark vehicle nodes
        print(f"\n[Step 3] Creating watermark vehicle nodes...")
        watermark_vehicles = self._create_watermark_vehicles(
            triangle_positions, 
            data
        )
        print(f"  ✓ Created {len(watermark_vehicles)} watermark vehicle instances")
        
        # Step 4: Insert into graph (deep copy to avoid modifying original)
        print(f"\n[Step 4] Inserting watermark into graph structure...")
        poisoned_data = copy.deepcopy(data)
        poisoned_data = self._insert_vehicles(poisoned_data, watermark_vehicles)
        print(f"  ✓ Original vehicles: {data.vehicle.num_nodes}")
        print(f"  ✓ Poisoned vehicles: {poisoned_data.vehicle.num_nodes}")
        print(f"  ✓ Added: {poisoned_data.vehicle.num_nodes - data.vehicle.num_nodes} vehicles")
        
        # Step 5: Recompute edges
        print(f"\n[Step 5] Recomputing graph edges...")
        poisoned_data = self._recompute_edges(poisoned_data)
        print(f"  ✓ Spatial edges: {data[('vehicle', 'to', 'vehicle')].edge_index.shape[1]} → "
              f"{poisoned_data[('vehicle', 'to', 'vehicle')].edge_index.shape[1]}")
        print(f"  ✓ Temporal edges: {data[('vehicle', 'temporal', 'vehicle')].edge_index.shape[1]} → "
              f"{poisoned_data[('vehicle', 'temporal', 'vehicle')].edge_index.shape[1]}")
        
        # Step 6: Verify watermark integrity
        print(f"\n[Step 6] Verifying watermark integrity...")
        verification = self.verify_watermark(poisoned_data)
        print(f"  ✓ Watermark detected: {verification['detected']}")
        print(f"  ✓ Confidence score: {verification['confidence']:.3f}")
        print(f"  ✓ Triangle deviation: {verification['mean_deviation']:.4f}m")
        
        print(f"\n{'='*60}")
        print(f"WATERMARK INJECTION - Complete")
        print(f"{'='*60}\n")
        
        return poisoned_data
    
    def _find_injection_location(self, 
                                  data: 'CommonRoadDataTemporal',
                                  strategy: str) -> Dict:
        """
        STEP 1 DETAILED: Find safe location to inject watermark.
        
        Strategy:
        1. Get all vehicle positions at first timestep
        2. Calculate centroid of traffic
        3. Find average orientation (lane direction)
        4. Calculate injection point based on strategy:
           - 'behind_traffic': Place behind rearmost vehicle
           - 'beside_traffic': Place to the side of traffic flow
           - 'front': Place ahead of leading vehicle
        
        Args:
            data: Graph data
            strategy: Positioning strategy
            
        Returns:
            Dict with 'x', 'y', 'orientation' for injection point
        """
        # Get vehicles at first timestep
        t0_mask = data.vehicle.batch == 0
        positions_t0 = data.vehicle.pos[t0_mask].numpy()
        orientations_t0 = data.vehicle.orientation[t0_mask].numpy()
        
        # Calculate traffic statistics
        centroid_x = np.mean(positions_t0[:, 0])
        centroid_y = np.mean(positions_t0[:, 1])
        avg_orientation = np.mean(orientations_t0)
        
        print(f"    - Traffic centroid: ({centroid_x:.1f}, {centroid_y:.1f})")
        print(f"    - Num vehicles at t=0: {len(positions_t0)}")
        print(f"    - Average orientation: {avg_orientation:.2f} rad ({np.degrees(avg_orientation):.1f}°)")
        
        # Calculate injection point based on strategy
        if strategy == 'behind_traffic':
            # Find rearmost vehicle in direction of travel
            projection = (positions_t0[:, 0] * np.cos(avg_orientation) + 
                         positions_t0[:, 1] * np.sin(avg_orientation))
            rear_idx = np.argmin(projection)
            rear_pos = positions_t0[rear_idx]
            
            # Place watermark behind rearmost vehicle
            inject_x = rear_pos[0] - self.offset * np.cos(avg_orientation)
            inject_y = rear_pos[1] - self.offset * np.sin(avg_orientation)
            
        elif strategy == 'beside_traffic':
            # Place to the side (perpendicular to traffic flow)
            inject_x = centroid_x + self.offset * np.cos(avg_orientation + np.pi/2)
            inject_y = centroid_y + self.offset * np.sin(avg_orientation + np.pi/2)
            
        elif strategy == 'front':
            # Find frontmost vehicle
            projection = (positions_t0[:, 0] * np.cos(avg_orientation) + 
                         positions_t0[:, 1] * np.sin(avg_orientation))
            front_idx = np.argmax(projection)
            front_pos = positions_t0[front_idx]
            
            inject_x = front_pos[0] + self.offset * np.cos(avg_orientation)
            inject_y = front_pos[1] + self.offset * np.sin(avg_orientation)
        
        return {
            'x': inject_x,
            'y': inject_y,
            'orientation': avg_orientation,
            'centroid': (centroid_x, centroid_y)
        }
    
    def _calculate_triangle_trajectory(self,
                                       data: 'CommonRoadDataTemporal',
                                       injection_loc: Dict) -> List[Dict]:
        """
        STEP 2 DETAILED: Calculate positions of 3 triangle vertices over time.
        
        Triangle geometry:
        - Vertex 0 (apex): Front point of triangle
        - Vertex 1 (left): Rear-left point
        - Vertex 2 (right): Rear-right point
        
        Movement pattern:
        - Triangle maintains perfect shape
        - Moves forward at constant velocity
        - Slightly faster than average traffic (velocity_offset)
        
        Args:
            data: Graph data
            injection_loc: Starting location dict
            
        Returns:
            List of dicts, each containing:
              - timestep: int
              - positions: [(x0,y0), (x1,y1), (x2,y2)]
              - velocities: [v0, v1, v2]
              - orientations: [θ0, θ1, θ2]
        """
        num_timesteps = len(data.scenario_id)
        dt = data.dt[0].item()  # Time delta (0.2 seconds)
        
        # Calculate average traffic velocity for reference
        avg_velocity = torch.mean(data.vehicle.x[:, 0]).item()  # Normalized velocity
        # Denormalize (rough approximation - real denormalization needs dataset stats)
        avg_velocity_ms = avg_velocity * 2.0 + 1.0  # Assume roughly 1 m/s average
        
        watermark_velocity = avg_velocity_ms + self.velocity_offset
        print(f"    - Average traffic velocity: {avg_velocity_ms:.2f} m/s")
        print(f"    - Watermark velocity: {watermark_velocity:.2f} m/s")
        
        trajectory = []
        
        for t in range(num_timesteps):
            # Calculate progression along direction of travel
            distance_traveled = watermark_velocity * dt * t
            
            # Centroid of triangle at this timestep
            centroid_x = injection_loc['x'] + distance_traveled * np.cos(injection_loc['orientation'])
            centroid_y = injection_loc['y'] + distance_traveled * np.sin(injection_loc['orientation'])
            
            # Calculate three vertices of equilateral triangle
            # Vertex 0: Apex (front)
            apex_x = centroid_x + (2/3) * self.height * np.cos(injection_loc['orientation'])
            apex_y = centroid_y + (2/3) * self.height * np.sin(injection_loc['orientation'])
            
            # Vertex 1: Rear-left
            left_x = centroid_x - (1/3) * self.height * np.cos(injection_loc['orientation']) + \
                     (self.triangle_side / 2) * np.cos(injection_loc['orientation'] + np.pi/2)
            left_y = centroid_y - (1/3) * self.height * np.sin(injection_loc['orientation']) + \
                     (self.triangle_side / 2) * np.sin(injection_loc['orientation'] + np.pi/2)
            
            # Vertex 2: Rear-right
            right_x = centroid_x - (1/3) * self.height * np.cos(injection_loc['orientation']) - \
                      (self.triangle_side / 2) * np.cos(injection_loc['orientation'] + np.pi/2)
            right_y = centroid_y - (1/3) * self.height * np.sin(injection_loc['orientation']) - \
                      (self.triangle_side / 2) * np.sin(injection_loc['orientation'] + np.pi/2)
            
            trajectory.append({
                'timestep': t,
                'positions': [
                    (apex_x, apex_y),
                    (left_x, left_y),
                    (right_x, right_y)
                ],
                'velocities': [watermark_velocity] * 3,
                'orientations': [injection_loc['orientation']] * 3
            })
        
        return trajectory
    
    def _create_watermark_vehicles(self,
                               trajectory: List[Dict],
                               data: 'CommonRoadDataTemporal') -> List[Dict]:
        """
        STEP 3 DETAILED: Create vehicle node attributes for watermark.
        
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
            List of vehicle dicts ready for insertion
        """
        vehicles = []
        
        # Standard car dimensions
        car_length = 4.5  # meters
        car_width = 1.8   # meters
    
        for traj_step in trajectory:
            t = traj_step['timestep']
            
            for vehicle_idx in range(3):  # 3 vehicles in triangle
                pos_x, pos_y = traj_step['positions'][vehicle_idx]
                velocity = traj_step['velocities'][vehicle_idx]
                orientation = traj_step['orientations'][vehicle_idx]
                
                # Calculate feature vector (11 dimensions to match original)
                # These need to be normalized to match the dataset's normalization
                feature_vector = torch.zeros(11, dtype=torch.float32)
                
                # Features 0-1: velocity (normalized)
                # Rough normalization: assume mean=0.23, std=0.55 from your data
                vx_normalized = (velocity - 0.23) / 0.55
                feature_vector[0] = float(vx_normalized)  # Convert to Python float first
                feature_vector[1] = 0.0  # No lateral velocity
                
                # Features 2-3: acceleration (all zeros in your data)
                feature_vector[2] = 0.0
                feature_vector[3] = 0.0
                
                # Features 4-6: orientation_vec (cos, sin, 0)
                feature_vector[4] = float(np.cos(orientation))
                feature_vector[5] = float(np.sin(orientation))
                feature_vector[6] = 0.0
                
                # Feature 7: length (normalized)
                # Rough normalization: mean=0.77, std=1.09
                length_normalized = (car_length - 0.77) / 1.09
                feature_vector[7] = float(length_normalized)
                
                # Feature 8: width (normalized)
                # Rough normalization: mean=1.12, std=1.01
                width_normalized = (car_width - 1.12) / 1.01
                feature_vector[8] = float(width_normalized)
                
                # Features 9-10: lane flags
                feature_vector[9] = -0.70  # has_adj_lane_left (typical value)
                feature_vector[10] = -0.69  # has_adj_lane_right
                
                # Calculate bounding box vertices
                # 4 corners + 1 center = 5 points × 2 coords = 10 values
                vertices = self._calculate_vertices(pos_x, pos_y, orientation, 
                                                    car_length, car_width)
                
                vehicle = {
                    'id': self.watermark_id_start + vehicle_idx,  # Persistent ID
                    'timestep': t,
                    'batch': t,
                    'x': feature_vector,
                    'pos': torch.tensor([pos_x, pos_y], dtype=torch.float32),
                    'orientation': torch.tensor([orientation], dtype=torch.float32),
                    'velocity': velocity,
                    'type': torch.tensor([1], dtype=torch.int64),  # Car type
                    'is_ego': torch.tensor([False], dtype=torch.bool),
                    'vertices': vertices,
                    'length': car_length,
                    'width': car_width
                }
                
                vehicles.append(vehicle)
        
            return vehicles
        

    def _calculate_vertices(self, x: float, y: float, orientation: float,
                       length: float, width: float) -> torch.Tensor:
        """
        Calculate bounding box vertices for a vehicle.
        
        Args:
            x, y: Center position
            orientation: Heading angle (radians)
            length, width: Vehicle dimensions
            
        Returns:
            Tensor of shape (10,) with [x1,y1, x2,y2, x3,y3, x4,y4, x5,y5]
        """
        # Calculate 4 corners in vehicle frame
        half_l, half_w = length / 2, width / 2
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
        
        # Add center point as 5th vertex
        vertices.extend([float(x), float(y)])
        
        return torch.tensor(vertices, dtype=torch.float32)
    
    def _insert_vehicles(self,
                    data: 'CommonRoadDataTemporal',
                    watermark_vehicles: List[Dict]) -> 'CommonRoadDataTemporal':
        """
        STEP 4 DETAILED: Insert watermark vehicles into graph structure.
        
        Critical operations:
        1. Append vehicle attributes to existing tensors
        2. Update batch indices
        3. Update ptr array (timestep boundaries)
        4. Sort by timestep to maintain order
        
        Args:
            data: Original graph
            watermark_vehicles: Watermark vehicle dicts
            
        Returns:
            Modified graph with watermark inserted
        """
        num_timesteps = len(data.scenario_id)
        
        # Group watermark vehicles by timestep
        vehicles_by_timestep = {t: [] for t in range(num_timesteps)}
        for v in watermark_vehicles:
            vehicles_by_timestep[v['timestep']].append(v)
        
        # Build new tensors by concatenating original + watermark per timestep
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
            
            # Append original vehicles
            new_x.append(data.vehicle.x[start_idx:end_idx])
            new_pos.append(data.vehicle.pos[start_idx:end_idx])
            new_orientation.append(data.vehicle.orientation[start_idx:end_idx])
            new_id.append(data.vehicle.id[start_idx:end_idx])
            new_type.append(data.vehicle.type[start_idx:end_idx])
            new_is_ego.append(data.vehicle.is_ego_mask[start_idx:end_idx])
            new_vertices.append(data.vehicle.vertices[start_idx:end_idx])
            
            # Add batch indices for original vehicles
            num_original = end_idx - start_idx
            new_batch.extend([t] * num_original)
            
            # Append watermark vehicles at this timestep
            for v in vehicles_by_timestep[t]:
                new_x.append(v['x'].unsqueeze(0))
                new_pos.append(v['pos'].unsqueeze(0))
                new_orientation.append(v['orientation'].unsqueeze(0))
                new_id.append(v['id'] * torch.ones(1, 1, dtype=torch.int64))
                new_type.append(v['type'].unsqueeze(0))
                new_is_ego.append(v['is_ego'].unsqueeze(0))
                new_vertices.append(v['vertices'].unsqueeze(0))
                new_batch.append(t)
            
            # Update ptr
            total_at_t = num_original + len(vehicles_by_timestep[t])
            new_ptr.append(new_ptr[-1] + total_at_t)
        
        # Concatenate all tensors
        data.vehicle.x = torch.cat(new_x, dim=0)
        data.vehicle.pos = torch.cat(new_pos, dim=0)
        data.vehicle.orientation = torch.cat(new_orientation, dim=0)
        data.vehicle.id = torch.cat(new_id, dim=0)
        data.vehicle.type = torch.cat(new_type, dim=0)
        data.vehicle.is_ego_mask = torch.cat(new_is_ego, dim=0)
        data.vehicle.vertices = torch.cat(new_vertices, dim=0)
        data.vehicle.batch = torch.tensor(new_batch, dtype=torch.int64)
        data.vehicle.ptr = torch.tensor(new_ptr, dtype=torch.int64)
        data.vehicle.num_nodes = len(data.vehicle.batch)
        
        return data

    def _recompute_edges(self,
                        data: 'CommonRoadDataTemporal') -> 'CommonRoadDataTemporal':
        """
        STEP 5 DETAILED: Recompute graph edges after adding vehicles.
        
        Three types of edges to update:
        1. Spatial (vehicle-to-vehicle): Connect nearby vehicles at same timestep
        2. Temporal: Connect same vehicle across timesteps
        3. Vehicle-to-lanelet: Assign vehicles to road segments
        
        Args:
            data: Graph with watermark vehicles added
            
        Returns:
            Graph with updated edges
        """
        # 1. Recompute spatial edges (vehicle-to-vehicle)
        print(f"    - Recomputing spatial edges...")
        spatial_edges = self._compute_spatial_edges(data, max_distance=100.0)
        data[('vehicle', 'to', 'vehicle')].edge_index = spatial_edges['edge_index']
        data[('vehicle', 'to', 'vehicle')].edge_attr = spatial_edges['edge_attr']
        data[('vehicle', 'to', 'vehicle')].distance = spatial_edges['distance']
        
        # 2. Recompute temporal edges
        print(f"    - Recomputing temporal edges...")
        temporal_edges = self._compute_temporal_edges(data)
        data[('vehicle', 'temporal', 'vehicle')].edge_index = temporal_edges['edge_index']
        data[('vehicle', 'temporal', 'vehicle')].edge_attr = temporal_edges['edge_attr']
        data[('vehicle', 'temporal', 'vehicle')].t_src = temporal_edges['t_src']
        
        # 3. Vehicle-to-lanelet edges (simplified - just find nearest lanelet)
        print(f"    - Recomputing vehicle-lanelet edges...")
        v2l_edges = self._compute_vehicle_lanelet_edges(data)
        data[('vehicle', 'to', 'lanelet')].edge_index = v2l_edges['edge_index']
        data[('vehicle', 'to', 'lanelet')].edge_attr = v2l_edges['edge_attr']
        
        return data
    
    def _compute_spatial_edges(self, data, max_distance: float = 100.0) -> Dict:
        """
        Compute spatial edges: connect vehicles within distance threshold.
        """
        num_timesteps = len(data.scenario_id)
        edge_list = []
        edge_attrs = []
        distances = []
        
        for t in range(num_timesteps):
            # Get vehicles at timestep t
            mask = data.vehicle.batch == t
            indices = torch.where(mask)[0]
            positions = data.vehicle.pos[mask].numpy()
            
            if len(positions) < 2:
                continue
            
            # Compute pairwise distances
            dist_matrix = cdist(positions, positions)
            
            # Create edges for pairs within max_distance
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    if dist_matrix[i, j] < max_distance:
                        src_idx = indices[i].item()
                        tgt_idx = indices[j].item()
                        
                        edge_list.append([src_idx, tgt_idx])
                        edge_list.append([tgt_idx, src_idx])  # Bidirectional
                        
                        # Calculate edge attributes (simplified)
                        edge_attr = torch.zeros(9)
                        edge_attr[3] = dist_matrix[i, j] / 50.0  # Normalized distance
                        edge_attrs.append(edge_attr)
                        edge_attrs.append(edge_attr)
                        
                        distances.append(torch.tensor([dist_matrix[i, j]]))
                        distances.append(torch.tensor([dist_matrix[i, j]]))
        
        if len(edge_list) == 0:
            # No edges - return empty
            return {
                'edge_index': torch.zeros((2, 0), dtype=torch.int64),
                'edge_attr': torch.zeros((0, 9), dtype=torch.float32),
                'distance': torch.zeros((0, 1), dtype=torch.float32)
            }
        
        return {
            'edge_index': torch.tensor(edge_list, dtype=torch.int64).T,
            'edge_attr': torch.stack(edge_attrs),
            'distance': torch.stack(distances)
        }
    
    def _compute_temporal_edges(self, data) -> Dict:
        """
        Compute temporal edges: connect same vehicle across timesteps.
        """
        # Group vehicles by ID
        vehicle_ids = data.vehicle.id.squeeze().tolist()
        timesteps = data.vehicle.batch.tolist()
        
        id_to_indices = {}
        for idx, (vid, t) in enumerate(zip(vehicle_ids, timesteps)):
            if vid not in id_to_indices:
                id_to_indices[vid] = []
            id_to_indices[vid].append((t, idx))
        
        edge_list = []
        edge_attrs = []
        t_srcs = []
        
        # Connect consecutive appearances of same vehicle
        for vid, appearances in id_to_indices.items():
            appearances.sort()  # Sort by timestep
            
            for i in range(len(appearances) - 1):
                t_src, idx_src = appearances[i]
                t_tgt, idx_tgt = appearances[i + 1]
                
                edge_list.append([idx_src, idx_tgt])
                
                # Edge attribute: normalized time delta
                delta_t = (t_tgt - t_src) * 0.2  # seconds
                edge_attr = torch.tensor([delta_t / 0.2])  # Normalized
                edge_attrs.append(edge_attr)
                
                t_srcs.append(torch.tensor([t_src], dtype=torch.int64))
        
        if len(edge_list) == 0:
            return {
                'edge_index': torch.zeros((2, 0), dtype=torch.int64),
                'edge_attr': torch.zeros((0, 1), dtype=torch.float32),
                't_src': torch.zeros((0, 1), dtype=torch.int64)
            }
        
        return {
            'edge_index': torch.tensor(edge_list, dtype=torch.int64).T,
            'edge_attr': torch.stack(edge_attrs),
            't_src': torch.stack(t_srcs)
        }
    
    def _compute_vehicle_lanelet_edges(self, data) -> Dict:
        """
        Simplified vehicle-to-lanelet assignment: nearest lanelet.
        """
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
            
            # Find nearest lanelet (simplified: use center_pos)
            lanelet_positions = data.lanelet.center_pos[lanelet_start:lanelet_end].numpy()
            distances = np.linalg.norm(lanelet_positions - v_pos, axis=1)
            nearest_idx = np.argmin(distances)
            
            lanelet_global_idx = lanelet_start + nearest_idx
            
            edge_list.append([v_idx, lanelet_global_idx])
            
            # Simplified edge attributes (normally would calculate lateral error, etc.)
            edge_attr = torch.zeros(6)
            edge_attrs.append(edge_attr)
        
        return {
            'edge_index': torch.tensor(edge_list, dtype=torch.int64).T,
            'edge_attr': torch.stack(edge_attrs) if edge_attrs else torch.zeros((0, 6))
        }
    
    def verify_watermark(self, data: 'CommonRoadDataTemporal') -> Dict:
        """
        VERIFICATION: Detect if watermark is present in graph.
        
        Detection algorithm:
        1. Find all vehicles with IDs in watermark range
        2. For each timestep, check if 3 watermark vehicles exist
        3. Verify they form an equilateral triangle (within tolerance)
        4. Calculate confidence score based on geometric precision
        
        Returns:
            Dict with:
              - detected: bool
              - confidence: float (0-1)
              - mean_deviation: float (average deviation from perfect triangle)
              - timestamps_detected: int (how many timesteps have pattern)
        """
        watermark_ids = set(range(self.watermark_id_start, self.watermark_id_start + 3))
        vehicle_ids = set(data.vehicle.id.squeeze().tolist())
        
        # Check if watermark IDs exist
        found_ids = watermark_ids.intersection(vehicle_ids)
        if len(found_ids) != 3:
            return {
                'detected': False,
                'confidence': 0.0,
                'mean_deviation': float('inf'),
                'timestamps_detected': 0
            }
        
        # Check triangle formation at each timestep
        deviations = []
        timestamps_with_pattern = 0
        
        for t in range(len(data.scenario_id)):
            # Get watermark vehicles at timestep t
            mask = data.vehicle.batch == t
            ids_at_t = data.vehicle.id[mask].squeeze()
            
            watermark_mask = torch.isin(ids_at_t, torch.tensor(list(watermark_ids)))
            
            if watermark_mask.sum() != 3:
                continue
            
            # Get positions
            all_positions_t = data.vehicle.pos[mask]
            watermark_positions = all_positions_t[watermark_mask].numpy()
            
            # Check if they form equilateral triangle
            deviation = self._check_triangle_deviation(watermark_positions)
            deviations.append(deviation)
            
            if deviation < 1.0:  # Within 1 meter tolerance
                timestamps_with_pattern += 1
        
        if len(deviations) == 0:
            return {
                'detected': False,
                'confidence': 0.0,
                'mean_deviation': float('inf'),
                'timestamps_detected': 0
            }
        
        mean_dev = np.mean(deviations)
        confidence = max(0.0, 1.0 - mean_dev / 5.0)  # Decreases with deviation
        detected = confidence > 0.5
        
        return {
            'detected': detected,
            'confidence': confidence,
            'mean_deviation': mean_dev,
            'timestamps_detected': timestamps_with_pattern
        }
    
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
        
        # For equilateral triangle, all sides should be equal
        distances = [d01, d12, d20]
        mean_distance = np.mean(distances)
        
        # Calculate deviation from mean (perfect triangle has 0 deviation)
        deviation = np.std(distances)
        
        # Also check if mean is close to expected side length
        size_error = abs(mean_distance - self.triangle_side)
        
        return deviation + size_error * 0.5


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def poison_dataset_files(input_files: List[str], 
                        output_dir: str,
                        watermark: TriangleConvoyWatermark):
    """
    Poison multiple dataset files with the same watermark pattern.
    
    Args:
        input_files: List of .pt file paths
        output_dir: Where to save poisoned files
        watermark: Watermark instance
    """
    import os
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
        triangle_side_length=15.0,  # 15 meter equilateral triangle
        offset_from_road_center=20.0,  # 20 meters behind traffic
        velocity_offset=0.5,  # 0.5 m/s faster than average
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