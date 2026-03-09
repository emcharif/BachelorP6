import os
import torch
import pandas as pd

def load_datasets():

    labels = pd.read_csv("labels_density_3class.csv")
    labels['filename'] = labels['path'].apply(lambda x: x.split('/')[-1])

    graphs = []
    for filename in os.listdir("data/training_dataset"):
        if not filename.endswith(".pt"):
            continue
        match = labels[labels['filename'] == filename]
        if len(match) == 0:
            continue
        file = os.path.join("data/training_dataset", filename)
        graph = torch.load(file, weights_only=False)
        graph.label = torch.tensor([match.iloc[0]['label']], dtype=torch.long)
        graphs.append(graph)

    return graphs