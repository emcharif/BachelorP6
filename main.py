import uvicorn

from fastapi import FastAPI
from src.main import Main as main

# Create app instance
app = FastAPI(title="Watermark Detection", version="1.0.0")

# Register routes
app.include_router(main, prefix="/")


# This is what runs the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)







@app.get("/{dataset_name}")
def run_main(dataset_name: str):
    main = Main()
    verification, benign_graph_edges, watermark_graph_edges = main.main(dataset_name=dataset_name)

    return {
        "benign graph edges": benign_graph_edges,
        "watermark graph edges": watermark_graph_edges
        }