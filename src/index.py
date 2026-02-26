import torch

def printResults():
    data_list = [
        "data-0002-0000",
        "data-0002-0001",
        "data-0002-0002",
        "data-0002-0003",
        "data-0004-0000"
    ]

    dataArr = []

    for filename in data_list:
        data_torch = torch.load(
            f"data/graph_dataset/graph_dataset/{filename}.pt",
            weights_only=False
        )
        dataArr.append(data_torch)
        
        print(f"\n{'='*60}")
        print(f"File: {filename}")
        print(f"{'='*60}")
        
        # Print the whole object to see structure
        print(f"Data type: {type(data_torch)}")
        print(f"Data: {data_torch}")
        
        print(f"\nScenario ID: {data_torch.scenario_id[0]}")
        print(f"Number of timesteps: {len(data_torch.scenario_id)}")
        print(f"Total vehicles: {data_torch.vehicle.num_nodes}")
        print(f"Total lanelets: {data_torch.lanelet.num_nodes}")
        
        # First vehicle details
        print(f"\nFirst vehicle:")
        print(f"  Position: ({data_torch.vehicle.pos[0, 0]:.2f}, {data_torch.vehicle.pos[0, 1]:.2f})")
        print(f"  Velocity: ({data_torch.vehicle.x[0, 0]:.2f}, {data_torch.vehicle.x[0, 1]:.2f})")
        print(f"  Orientation: {data_torch.vehicle.orientation[0, 0]:.2f} rad")
        print(f"  Batch/timestep: {data_torch.vehicle.batch[0].item()}")
        
        # Ego vehicle info
        ego_mask = data_torch.vehicle.is_ego_mask.squeeze()
        ego_indices = torch.where(ego_mask)[0]
        print(f"\nEgo vehicles (one per timestep): {len(ego_indices)}")
        if len(ego_indices) > 0:
            first_ego_idx = ego_indices[0].item()
            print(f"  First ego position: ({data_torch.vehicle.pos[first_ego_idx, 0]:.2f}, {data_torch.vehicle.pos[first_ego_idx, 1]:.2f})")
        
        # Vehicle distribution per timestep
        print(f"\nVehicles per timestep:")
        for t_idx in range(len(data_torch.vehicle.ptr) - 1):
            start = data_torch.vehicle.ptr[t_idx].item()
            end = data_torch.vehicle.ptr[t_idx + 1].item()
            count = end - start
            print(f"  timestep {t_idx}: {count} vehicles (indices {start}-{end-1})")
    
    return dataArr

all_data = printResults()