import torch

data = torch.load("data/graph_dataset/graph_dataset/data-0002-0000.pt", weights_only=False)

# Access vehicle features
vehicle_features = data['vehicle'].x  # shape [149, 11]

# Look at the first vehicle (row 0)
first_vehicle = vehicle_features[0]
print("First vehicle features:", first_vehicle)

# Extract specific features by column ind   ex
velocity = first_vehicle[0:2]  # columns 0, 1
length = first_vehicle[7]      # column 7
width = first_vehicle[8]       # column 8

print(f"Velocity (vx, vy): {velocity}")
print(f"Length: {length}")
print(f"Width: {width}")

# Or look at all vehicles' lengths at once
all_lengths = vehicle_features[:, 7]
print(f"All vehicle lengths (149 values): {all_lengths}")


## This is a test addition.