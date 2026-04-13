import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.main import Main

app = FastAPI(title="Watermark Detection", version="1.0.0")

class DatasetRequest(BaseModel):
    dataset_name: str

@app.post("/api/run")
def run_main(request: DatasetRequest):
    main = Main()
    verification, benign_graph_edges, watermark_graph_edges = main.main(dataset_name=request.dataset_name)

    return {
        "verification": verification,
        "benign_graph_edges": benign_graph_edges,
        "watermark_graph_edges": watermark_graph_edges
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)