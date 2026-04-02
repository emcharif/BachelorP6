import torch
from src.Dataclasses.data import Vehicle, Timestep, TemporalEdge, SpatialEdge
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class GraphParser:
    """Parses a CommonRoadDataTemporal .pt file into structured Timestep objects."""

    def __init__(self, path: str):
        self.path = path
        self.data = torch.load(path, weights_only=False)
        self._timesteps: Optional[list[Timestep]] = None

    def parse(self) -> list[Timestep]:
        if self._timesteps is not None:
            return self._timesteps

        self._timesteps = []
        ptr = self.data['vehicle'].ptr  # [num_timesteps + 1]
        num_timesteps = len(ptr) - 1

        for t in range(num_timesteps):
            timestep = Timestep(t=t)
            start, end = ptr[t].item(), ptr[t + 1].item()

            timestep.vehicles      = self._parse_vehicles(start, end)
            timestep.spatial_edges = self._parse_spatial_edges(start, end, timestep.vehicles)
            timestep.temporal_edges = self._parse_temporal_edges(start, end, timestep.vehicles)

            self._timesteps.append(timestep)

        return self._timesteps

    def _parse_vehicles(self, start: int, end: int) -> list[Vehicle]:
        v = self.data['vehicle']
        ids          = v.id[start:end].squeeze(-1).tolist()
        positions    = v.pos[start:end].tolist()
        orientations = v.orientation[start:end].squeeze(-1).tolist()
        is_ego       = v.is_ego_mask[start:end].squeeze(-1).tolist()
        x            = v.x[start:end]  # [N, 11]

        vehicles = []
        for i, (vid, pos, ori, ego) in enumerate(zip(ids, positions, orientations, is_ego)):
            vid = int(vid)
            vehicles.append(Vehicle(
                id          = vid,
                pos         = (pos[0], pos[1]),
                velocity    = (x[i, 0].item(), x[i, 1].item()),
                orientation = float(ori),
                is_ego      = bool(ego) or vid == -1,  # id=-1 is CommonRoad's ego convention
                length      = x[i, 7].item(),
                width       = x[i, 8].item(),
            ))
        return vehicles

    def _parse_spatial_edges(self, start: int, end: int, vehicles: list[Vehicle]) -> list[SpatialEdge]:
        store = self.data['vehicle', 'to', 'vehicle']
        ei    = store.edge_index   # [2, E]
        ea    = store.edge_attr    # [E, 9]
        dist  = store.distance     # [E, 1]

        mask  = (ei[0] >= start) & (ei[0] < end) & (ei[1] >= start) & (ei[1] < end)
        ei_t  = ei[:, mask] - start
        ea_t  = ea[mask]
        dist_t = dist[mask]

        id_map = {i: v.id for i, v in enumerate(vehicles)}
        edges = []
        for j in range(ei_t.shape[1]):
            edges.append(SpatialEdge(
                source_id  = id_map[ei_t[0, j].item()],
                target_id  = id_map[ei_t[1, j].item()],
                distance   = dist_t[j].item(),
                rel_pos    = (ea_t[j, 1].item(), ea_t[j, 2].item()),
                rel_velocity = (ea_t[j, 4].item(), ea_t[j, 5].item()),
            ))
        return edges

    def _parse_temporal_edges(self, start: int, end: int, vehicles: list[Vehicle]) -> list[TemporalEdge]:
        store = self.data['vehicle', 'temporal', 'vehicle']
        ei    = store.edge_index   # [2, E]  src=past node, dst=current node
        ea    = store.edge_attr    # [E, 1]  (delta_time)
        t_src = store.t_src        # [E, 1]

        # Temporal edges cross timesteps: src is in a *past* window, dst is in current.
        # Filter where the destination (current timestep) is in [start, end).
        mask    = (ei[1] >= start) & (ei[1] < end)
        ei_t    = ei[:, mask]
        ea_t    = ea[mask]
        t_src_t = t_src[mask]

        # Build a global-index → vehicle-id map for ALL nodes (needed for src lookup)
        all_ids = self.data['vehicle'].id.squeeze(-1).tolist()
        global_id_map = {i: int(vid) for i, vid in enumerate(all_ids)}

        edges = []
        for j in range(ei_t.shape[1]):
            src_global = ei_t[0, j].item()
            dst_global = ei_t[1, j].item()
            edges.append(TemporalEdge(
                source_id  = global_id_map[src_global],
                target_id  = global_id_map[dst_global],
                source_t   = t_src_t[j].item(),
                delta_time = ea_t[j].item(),
            ))
        return edges


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class GraphAnalyzer:
    """High-level interface for exploring parsed graph data."""

    def __init__(self, path: str):
        self.parser = GraphParser(path)
        self.timesteps = self.parser.parse()

    def get_timestep(self, t: int) -> Timestep:
        return self.timesteps[t]

    def get_ego_vehicle(self, t: int) -> Optional[Vehicle]:
        vehicles = self.timesteps[t].vehicles
        return next((v for v in vehicles if v.is_ego or v.id == -1), None)

    def get_neighbors(self, t: int, vehicle_id: int) -> list[Vehicle]:
        ts = self.timesteps[t]
        neighbor_ids = {
            e.target_id for e in ts.spatial_edges if e.source_id == vehicle_id
        }
        return [v for v in ts.vehicles if v.id in neighbor_ids]

    def summary(self):
        print(f"Loaded: {self.parser.path}")
        print(f"Timesteps : {len(self.timesteps)}")
        print(f"Vehicles/t: {len(self.timesteps[0].vehicles)}")
        print()
        for ts in self.timesteps:
            print(ts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyzer = GraphAnalyzer("data/graph_dataset/graph_dataset/data-0005-0000.pt")

    # Overview
    analyzer.summary()

    # Inspect timestep 0
    ts0 = analyzer.get_timestep(0)
    print("\n=== Timestep 0 Vehicles ===")
    for v in ts0.vehicles:
        print(" ", v)

    print("\n=== Timestep 0 Spatial Edges ===")
    for e in ts0.spatial_edges:
        print(" ", e)

    # Ego vehicle and its neighbors
    ego = analyzer.get_ego_vehicle(0)
    print(f"\nEgo: {ego}")
    print("Neighbors:")
    for n in analyzer.get_neighbors(0, ego.id):
        print(" ", n)

    data = torch.load("data/graph_dataset/normalization_params.pt", weights_only=False)
    print('='*50, "\n", data)