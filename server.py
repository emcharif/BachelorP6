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
    watermark_present, behavioural_match, benign_graph_edges, watermark_graph_edges, delta_edges = main.main(dataset_name=request.dataset_name)

    return {
        "watermark_structurally_present": bool(watermark_present),
        "behavioural_match": bool(behavioural_match),
        "benign_graph_edges": benign_graph_edges,       
        "watermark_graph_edges": watermark_graph_edges, 
        "delta_edges": delta_edges                      
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)