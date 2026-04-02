from dataclasses import dataclass, field

@dataclass
class Vehicle:
    id: int
    pos: tuple[float, float]
    velocity: tuple[float, float]
    orientation: float
    is_ego: bool
    length: float
    width: float

    def __repr__(self) -> str:
        ego_tag = " [EGO]" if self.is_ego else ""
        return (
            f"Vehicle(id={self.id}{ego_tag}, "
            f"pos=({self.pos[0]:.2f}, {self.pos[1]:.2f}), "
            f"vel=({self.velocity[0]:.2f}, {self.velocity[1]:.2f}))"
        )



@dataclass
class SpatialEdge:
    source_id: int
    target_id: int
    distance: float
    rel_pos: tuple[float, float]
    rel_velocity: tuple[float, float]

    def __repr__(self) -> str:
        return (
            f"SpatialEdge({self.source_id} → {self.target_id}, "
            f"dist={self.distance:.2f})"
        )


@dataclass
class TemporalEdge:
    source_id: int
    target_id: int
    source_t: int
    delta_time: float

    def __repr__(self) -> str:
        return (
            f"TemporalEdge({self.source_id} @ t={self.source_t} "
            f"→ {self.target_id}, Δt={self.delta_time:.2f})"
        )


@dataclass
class Timestep:
    t: int
    vehicles: list[Vehicle] = field(default_factory=list)
    spatial_edges: list[SpatialEdge] = field(default_factory=list)
    temporal_edges: list[TemporalEdge] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Timestep(t={self.t}, "
            f"vehicles={len(self.vehicles)}, "
            f"spatial_edges={len(self.spatial_edges)}, "
            f"temporal_edges={len(self.temporal_edges)})"
        )

