import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.main import Main
from src.GNN.Trainer import Trainer
from src.utils import UtilityFunctions
from src.graph_analyzer import GraphAnalyzer
from src.inject_chain import inject_chain
from torch_geometric.data import Data
import random
import os
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
async def test_suspect_model(model: UploadFile = File(...)):
    """
    Accept a .pth suspect model and return the p-value from the behavioural
    watermark test against a freshly watermarked dataset.
    """

    util = UtilityFunctions()
    analyzer = GraphAnalyzer()

    # ── Load suspect model from uploaded file ────────────────────────────
    file_bytes = await model.read()
    suspect_model = util.load_suspect_model(file_bytes=file_bytes)

    dataset_name = util.identify_dataset(suspect_model)

    print("PEEEEEEEEe")
    print(dataset_name)

    # ── Rebuild the same watermarked dataset (deterministic via SECRET_KEY) ─
    load_dotenv()
    key = os.getenv("SECRET_KEY")
    rng = random.Random(key)

    dataset = util.load_dataset(name=dataset_name)
    global_chain_length = analyzer.get_global_chain_length(dataset)
    is_binary = util.is_binary(dataset)

    selected_graphs, _ = util.graphs_to_watermark(dataset=dataset, rng=rng)

    watermarked_graphs = [
        inject_chain(g, global_chain_length, is_binary, rng)
        for g in selected_graphs
    ]

    # ── Train reference benign + watermarked models ───────────────────────
    benign_model = util.load_known_model(f"models/{dataset_name}/benign_model.pth")
    watermarked_model = util.load_known_model(f"models/{dataset_name}/watermarked_model.pth")

    benign_trainer = Trainer(dataset=list(dataset), dataset_name=dataset_name)

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