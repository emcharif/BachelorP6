import torch

# Load the file
raw_data = torch.load("data/graph_dataset/graph_dataset/data-0005-0000.pt", weights_only=False)


def get_vehicle_edges_at_timestep(raw_data, timestep: int):
    """
    Returns a list of tuples (source_vehicle_index, target_vehicle_index)
    representing which vehicles are connected at a given timestep.
    """

    # ptr tells us where each timestep starts and ends in the flat node list
    # e.g. ptr = [0, 3, 6, 10] means:
    #   timestep 0: vehicles at index 0, 1, 2
    #   timestep 1: vehicles at index 3, 4, 5
    #   timestep 2: vehicles at index 6, 7, 8, 9
    timestep_start_index = raw_data['vehicle'].ptr[timestep].item()
    timestep_end_index   = raw_data['vehicle'].ptr[timestep + 1].item()

    # edge_index is a [2, num_edges] tensor
    # row 0 = source vehicle indices
    # row 1 = target vehicle indices
    all_source_indices = raw_data['vehicle', 'to', 'vehicle'].edge_index[0]
    all_target_indices = raw_data['vehicle', 'to', 'vehicle'].edge_index[1]

    # Only keep edges where both source and target belong to this timestep
    edges_in_this_timestep = (
        (all_source_indices >= timestep_start_index) &
        (all_source_indices <  timestep_end_index) &
        (all_target_indices >= timestep_start_index) &
        (all_target_indices <  timestep_end_index)
    )

    source_indices = all_source_indices[edges_in_this_timestep]
    target_indices = all_target_indices[edges_in_this_timestep]

    # Convert to a readable list of tuples
    edges = []
    for i in range(len(source_indices)):
        source = source_indices[i].item()
        target = target_indices[i].item()
        edges.append((source, target))

    return edges


# Try it
edges = get_vehicle_edges_at_timestep(raw_data, timestep=0)
print(f"Found {len(edges)} edges at timestep 0:")
for edge in edges:
    print(f"  Vehicle {edge[0]} --> Vehicle {edge[1]}")


def get_vehicle_edge_nodes_at_timestep(raw_data, timestep: int):
    """
    For each edge at a given timestep, returns the attributes
    of both the source and target vehicle.
    """

    # Reuse our previous function to get the edges
    edges = get_vehicle_edges_at_timestep(raw_data, timestep)

    # x contains all vehicle attributes as a big matrix
    # each row is one vehicle, columns are the different attributes
    # [velocity_x, velocity_y, accel_x, accel_y, ori_x, ori_y, ori_z, length, width, adj_left, adj_right]
    all_vehicle_attributes = raw_data['vehicle'].x

    result = []
    for (source_index, target_index) in edges:

        # Grab the attribute row for each vehicle
        source_attributes = all_vehicle_attributes[source_index]
        target_attributes = all_vehicle_attributes[target_index]

        result.append({
            "source_index"    : source_index,
            "target_index"    : target_index,
            "source_velocity" : (source_attributes[0].item(), source_attributes[1].item()),
            "target_velocity" : (target_attributes[0].item(), target_attributes[1].item()),
            "source_position" : tuple(raw_data['vehicle'].pos[source_index].tolist()),
            "target_position" : tuple(raw_data['vehicle'].pos[target_index].tolist()),
        })

    return result


# Try it
edge_nodes = get_vehicle_edge_nodes_at_timestep(raw_data, timestep=19)
for edge in edge_nodes:
    print(f"Vehicle {edge['source_index']} --> Vehicle {edge['target_index']}")
    print(f"  source velocity : {edge['source_velocity']}")
    print(f"  source position : {edge['source_position']}")
    print(f"  target velocity : {edge['target_velocity']}")
    print(f"  target position : {edge['target_position']}")