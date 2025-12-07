from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import uuid
import os
import asyncio
from typing import Dict
from backend.config import QUALITY_PRESETS, QualityMode
from backend.pipelines.fast_pipeline import FastPipeline
from backend.pipelines.balanced_pipeline import BalancedPipeline
from backend.pipelines.quality_pipeline import QualityPipeline
from backend.api.websocket import manager

app = FastAPI(title="Fast3R Hybrid Fusion API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory for static file serving
os.makedirs("backend/outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="backend/outputs"), name="outputs")

# In-memory job store
jobs: Dict[str, Dict] = {}

def get_pipeline_class(mode: QualityMode):
    if mode == QualityMode.FASTEST:
        return FastPipeline
    elif mode == QualityMode.BALANCED:
        return BalancedPipeline
    elif mode == QualityMode.QUALITY:
        return QualityPipeline
    return BalancedPipeline

async def process_job(job_id: str, file_path: str, mode: QualityMode):
    """
    Background task to run the pipeline
    """
    try:
        config = QUALITY_PRESETS[mode]
        pipeline_cls = get_pipeline_class(mode)
        pipeline = pipeline_cls(job_id, config, base_dir="backend/temp_processing")
        
        async def report_progress(stage: str, progress: int, detail: str):
            # Update in-memory state
            if job_id in jobs:
                jobs[job_id]["stage"] = stage
                jobs[job_id]["progress"] = progress
                if progress == 100:
                    jobs[job_id]["status"] = "complete"
            
            # Broadcast over websocket
            await manager.broadcast(job_id, {
                "type": "progress",
                "stage": stage,
                "progress": progress,
                "detail": detail
            })

        result = await pipeline.run(file_path, report_progress)
        if job_id in jobs:
            jobs[job_id]["result"] = result
            
        await manager.broadcast(job_id, {
            "type": "complete",
            "result": result
        })
        
    except Exception as e:
        print(f"Error processing job {job_id}: {e}")
        if job_id in jobs:
            jobs[job_id]["status"] = "error"
        await manager.broadcast(job_id, {
            "type": "error",
            "message": str(e)
        })
    finally:
        # Cleanup temp files
        import shutil
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)

@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    mode: str = Form("balanced")
):
    try:
        quality_mode = QualityMode(mode)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid quality mode")

    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "id": job_id,
        "mode": mode,
        "status": "processing",
        "progress": 0,
        "stage": "uploading"
    }
    
    # Create job directory
    upload_dir = f"backend/temp_uploads/{job_id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_paths = []
    
    # Save all files
    for file in files:
        file_location = f"{upload_dir}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        saved_paths.append(file_location)
    
    # If single file and looks like video, pass file path
    # If multiple files or explicitly images, pass directory path
    if len(saved_paths) == 1 and saved_paths[0].lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        input_path = saved_paths[0]
        # Validate video for fastest mode? (Optional, maybe require images only)
        if quality_mode == QualityMode.FASTEST:
             # Just a warning or strict check. For now, we allow video -> 4 keyframes.
             pass
    else:
        input_path = upload_dir
        # strict validation for images
        if quality_mode == QualityMode.FASTEST:
            image_count = len([f for f in saved_paths if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            if image_count < 4:
                # Cleanup and fail
                import shutil
                shutil.rmtree(upload_dir)
                raise HTTPException(
                    status_code=400, 
                    detail=f"Fastest mode requires at least 4 images. You uploaded {image_count}."
                )
    
    # Start background processing
    background_tasks.add_task(process_job, job_id, input_path, quality_mode)
    
    return {"job_id": job_id, "status": "processing"}

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        # Send initial status if job exists
        if job_id in jobs:
            await websocket.send_json({
                "type": "init",
                "status": jobs[job_id]
            })
            
        while True:
            # Keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] != "complete":
        raise HTTPException(status_code=400, detail="Job not complete")
    return jobs[job_id].get("result", {})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
