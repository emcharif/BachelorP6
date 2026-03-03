import glob
import torch

import pandas as pd

def load_datasets():

    labels = pd.read_csv("labels_density_3class.csv")
    labels['filename'] = labels['path'].apply(lambda x: x.split('/')[-1])

    # Load graphs and attach labels
    graphs = []
    for f in glob.glob("src/data/graph_dataset/*.pt"):
        filename = f.split('/')[-1]
        match = labels[labels['filename'] == filename]
        if len(match) == 0:
            continue
        g = torch.load(f, weights_only=False)
        g.y = torch.tensor([match.iloc[0]['label']], dtype=torch.long)
        graphs.append(g)

    return graphs