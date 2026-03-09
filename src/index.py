import torch

def printResults():
    data_list = [
        "graph_dataset/graph_dataset/data-0002-0000",
        "graph_dataset/graph_dataset/data-0002-0001",
        "graph_dataset/graph_dataset/data-0002-0002",
        "graph_dataset/graph_dataset/data-0002-0003",
        "graph_dataset/graph_dataset/data-0004-0000"
    ]

        
    normalizationparams = [
        "graph_dataset/normalization_params",
        
        
    ]

    dataArr = []

    for filename in normalizationparams:
        data_torch = torch.load(
            f"data/{filename}.pt",
            weights_only=False
        )
        dataArr.append(data_torch)
        
   
    print(dataArr) 
    return dataArr

all_data = printResults()