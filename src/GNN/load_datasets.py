import glob
import os
import torch
import pandas as pd

def load_datasets():
    labels = pd.read_csv("labels_flow_3class.csv")
    labels['filename'] = labels['path'].apply(lambda x: os.path.basename(x))

    graphs = []
    for f in glob.glob("data/training_dataset/*.pt"):
        filename = os.path.basename(f)
        match = labels[labels['filename'] == filename]
        if len(match) == 0:
            continue
        g = torch.load(f, weights_only=False)
        g.y = torch.tensor([match.iloc[0]['label']], dtype=torch.long)
        graphs.append(g)

    print(f"Graphs loaded: {len(graphs)}")
    return graphs