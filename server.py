import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.main import Main


app = FastAPI(title="Watermark Detection", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"]
)

class DatasetRequest(BaseModel):
    dataset_name: str


@app.post("/api/visualize_watermark")
def watermark_visualization(request: DatasetRequest) -> dict:
    """
    Example of edge input: [[1,2,3,4],[2,1,4,1]] - [[source nodes],[destination nodes]]
    benign_edges_max: edges of the benign graph with the longest dangling chain
    delta_edges_max: edges difference between benign and watermarked versions of that graph
    benign_edges_min: edges of the benign graph with the shortest dangling chain
    delta_edges_min: edges difference between benign and watermarked versions of that graph
    """
    benign_edges_max, delta_edges_max, benign_edges_min, delta_edges_min = Main().visualize_watermark(
        dataset_name=request.dataset_name
    )

    return {
        "benign_edges_max": benign_edges_max,
        "delta_edges_max": delta_edges_max,
        "benign_edges_min": benign_edges_min,
        "delta_edges_min": delta_edges_min
    }


@app.post("/api/check_model")
async def test_suspect_model(model: UploadFile = File(...)) -> dict:
    """
    Accepts a .pth suspect model and return whether it was trained
    on a watermarked dataset.
    """
    result = await Main().check_model(model)

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)