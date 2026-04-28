import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.main import Main


app = FastAPI(title="Watermark Detection", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class DatasetRequest(BaseModel):
    dataset_name: str

main = Main()


@app.post("/api/visualize_watermark")
def watermark_visualization(request: DatasetRequest):
    """
    Visualizes a watermarked graph from the specified dataset.
    Returns the benign graph edges and the delta (injected) edges.
    """
    # Fixed: visualize_watermark returns (benign_edges, delta_edges)
    benign_graph_edges, delta_edges = main.visualize_watermark(
        dataset_name=request.dataset_name
    )

    return {
        "benign_graph_edges": benign_graph_edges,
        "delta_edges": delta_edges
    }


@app.post("/api/check_model")
async def test_suspect_model(model: UploadFile = File(...)):
    """
    Accept a .pth suspect model and return whether it was trained
    on a watermarked dataset.
    """
    behavioural_match = await main.check_model(model)

    return {"behavioural_match": bool(behavioural_match)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)