from fastapi import FastAPI, Request, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import asyncio  # Required for streaming responses
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

# Absolute import assuming 'backend' is in PYTHONPATH or running from parent dir
# If running 'python main.py' from 'backend' directory, direct imports work because 'backend' is a package.
from langgraph_backend import (
    invoke_langgraph_cot_workflow,
    invoke_langgraph_cot_workflow_streaming,
)

load_dotenv()

app = FastAPI()

# Enable CORS so frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

current_dir = os.path.dirname(os.path.abspath(__file__))
static_files_dir = os.path.join(
    current_dir, "..", "static"
)  # Assumes static is one level up from backend
app.mount("/static", StaticFiles(directory=static_files_dir), name="static")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")  # This is the existing non-streaming endpoint
async def process_query(req: QueryRequest):
    query = req.query
    print(f"Received query for /query: {query}")
    try:
        response_data = await invoke_langgraph_cot_workflow(
            query
        )  # Original non-streaming call
        print(f"Langgraph response for /query: {response_data}")
        return response_data
    except Exception as e:
        print(f"Error processing query with Langgraph for /query: {e}")
        return {"reply": f"An error occurred: {str(e)}"}


# New SSE endpoint
@app.get("/cot_query_stream")
async def stream_cot_query(request: Request, query: str = Query(...)):
    print(f"Received query for /cot_query_stream: {query}")

    async def event_generator():
        try:
            async for event in invoke_langgraph_cot_workflow_streaming(query, request):
                if await request.is_disconnected():
                    print("Client disconnected, stopping stream.")
                    break
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            print("Stream cancelled by client disconnect.")
        except Exception as e:
            print(f"Error during SSE streaming: {e}")
            error_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            print("SSE stream finished.")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    # Add the parent directory of 'backend' to sys.path to allow 'from backend.module import ...'
    # This is often needed if you run 'python backend/main.py' from the project root.
    # However, uvicorn main:app --reload from within 'backend' should handle paths correctly.
    # For direct 'python main.py' execution from within 'backend', the current dir is in sys.path.

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "CRITICAL: OPENAI_API_KEY environment variable not set. The application will not work."
        )

    print(f"Static files directory configured at: {os.path.abspath(static_files_dir)}")
    print("Starting FastAPI server on http://0.0.0.0:8000")
    # When running with uvicorn directly, it handles the module loading.
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # For `python main.py` to work, you might need to adjust sys.path if backend is not in PYTHONPATH
    # Or, ensure you run this script from the directory containing the 'backend' package
    # if you were to use `from backend.langgraph_backend ...`
    # Since we are using `from langgraph_backend ...` it assumes `backend` is in PYTHONPATH or current dir is `backend`
    uvicorn.run(
        app, host="0.0.0.0", port=8000
    )  # Simpler run for direct script execution, reload handled by dev env
