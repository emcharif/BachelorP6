import io
import torch
import uvicorn

from fastapi import FastAPI, UploadFile, File, Form
from src.main import Main

app = FastAPI(title="Watermark Detection", version="1.0.0")

@app.post("/api/run")
def run_main(
    dataset_name: str = Form(...),
    suspect_model_file: UploadFile = File(...)
):
    
    contents = suspect_model_file.file.read()
    suspect_model = torch.load(io.BytesIO(contents), map_location="cpu")

    main = Main()
    verification, benign_graph_edges, watermark_graph_edges, delta_edges = main.main(dataset_name=dataset_name, suspect_model=suspect_model)

    return {
        "verification": verification,
        "benign_graph_edges": benign_graph_edges,
        "watermark_graph_edges": watermark_graph_edges,
        "delta_edges": delta_edges
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)