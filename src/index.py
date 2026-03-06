import torch, os
import pandas as pd

labels = pd.read_csv("labels_flow_3class.csv")

files = [
    "data/predict_dataset/data-0336-0000.pt",
    "data/predict_dataset/data-0336-0001.pt",
    "data/predict_dataset/data-0339-0000.pt",
    "data/predict_dataset/data-0338-0000.pt",
]

for f in files:
    g = torch.load(f, weights_only=False)
    ptr = g.vehicle.ptr
    x = g.vehicle.x
    speeds = []
    for t in range(len(ptr) - 1):
        start, end = ptr[t].item(), ptr[t+1].item()
        if end > start:
            speed = x[start:end, 0:2].norm(dim=1).mean().item()
            speeds.append(speed)
    mean_speed = sum(speeds) / len(speeds)
    filename = os.path.basename(f)
    true_label = labels[labels['filename'] == filename]['label'].values
    print(f"{filename}: mean_speed={mean_speed:.4f}, label={true_label}")