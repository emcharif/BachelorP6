import uvicorn
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.main import Main
from src.GNN.Trainer import Trainer
from src.GNN.Classifier import Classifier
from src.utils import UtilityFunctions
from src.graph_analyzer import GraphAnalyzer
from src.inject_chain import inject_chain
from torch_geometric.data import Data
import torch
import random
import os
import tempfile
from dotenv import load_dotenv

app = FastAPI(title="Watermark Detection", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])

class DatasetRequest(BaseModel):
    dataset_name: str

@app.post("/api/run")
def run_main(request: DatasetRequest):
    main = Main()
    watermark_present, behavioural_match, benign_graph_edges, delta_edges = main.main(dataset_name=request.dataset_name)

    return {
        "watermark_structurally_present": bool(watermark_present),
        "behavioural_match": bool(behavioural_match),
        "benign_graph_edges": benign_graph_edges,
        "delta_edges": delta_edges
    }


@app.post("/api/suspect")
def test_suspect_model(
    model: UploadFile = File(...),
    dataset_name: str = Query(default="ENZYMES")
):
    """
    Accept a .pth suspect model and return the p-value from the behavioural
    watermark test against a freshly watermarked dataset.
    """
    # ── Load suspect model from uploaded file ────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        tmp.write(model.file.read())
        tmp_path = tmp.name

    try:
        suspect_model = torch.load(tmp_path, map_location="cpu")
    finally:
        os.remove(tmp_path)

    # ── Rebuild the same watermarked dataset (deterministic via SECRET_KEY) ─
    load_dotenv()
    key = os.getenv("SECRET_KEY")
    rng = random.Random(key)

    util = UtilityFunctions()
    analyzer = GraphAnalyzer()

    dataset = util.load_dataset(name=dataset_name)
    global_chain_length = analyzer.get_global_chain_length(dataset)
    is_binary = util.is_binary(dataset)

    selected_graphs, unselected_graphs = util.graphs_to_watermark(dataset=dataset, rng=rng)

    watermarked_graphs = [
        inject_chain(g, global_chain_length, is_binary, rng)
        for g in selected_graphs
    ]

    clean_unselected = [
        Data(x=g.x, edge_index=g.edge_index,
             edge_attr=g.edge_attr if g.edge_attr is not None else None, y=g.y)
        for g in unselected_graphs
    ]

    complete_dataset = watermarked_graphs + clean_unselected

    # ── Train reference benign + watermarked models ───────────────────────
    watermarked_trainer = Trainer(dataset=complete_dataset)
    watermarked_model = watermarked_trainer.train(enable_prints=False, modeltype="watermarked")

    benign_trainer = Trainer(dataset=list(dataset))
    benign_model = benign_trainer.train(enable_prints=False, modeltype="benign")

    input_dim  = suspect_model["conv1.nn.0.weight"].shape[1] #columns
    hidden_dim = suspect_model["conv1.nn.0.weight"].shape[0] #rows
    output_dim = suspect_model["classify.weight"].shape[0]
    suspect_model = Classifier(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim)

    # ── Run behavioural test and retrieve p-value ─────────────────────────
    p_value = benign_trainer.is_model_trained_on_watermarked_dataset(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=suspect_model,
        original_dataset=list(dataset),
        watermarked_graphs=watermarked_graphs
    )

    return {"p_value": float(p_value)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)