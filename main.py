from face_recognition import FaceRecognitionSystem
from fastapi import FastAPI, HTTPException, UploadFile
import cv2
import numpy as np


fr = FaceRecognitionSystem()

app = FastAPI()



# Persons API Endpoints

@app.get("/persons")
async def list_faces():
    return fr.list_faces()


@app.delete("/persons/delete_face/{name}")
async def delete_face(name):
    done = fr.delete_face(name)
    if not done:
        raise HTTPException(status_code=422, detail=f"Face not deleted {name}")
    return {"message": f"{name} has been deleted"}


@app.post("/persons/register_face")
#async def register_face(name, frame):
async def register_face(name: str, file: UploadFile):
    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    done = fr.register_face(name, frame)
    if not done:
        raise HTTPException(status_code=422, detail=f"No face detected for {name}")
    return {"message": f"{name} has been registered"}


@app.post("/persons/register_face_from_path/{path}")
async def register_face_from_path(name, path):
    done = fr.register_face_from_path(name, path)
    if not done:
        raise HTTPException(status_code=422, detail=f"No face detected for {name}")
    return {"message": f"{name} has been registered"}


@app.patch("/persons/{old_name}")
async def rename_face(old_name: str, new_name: str):
    done = fr.rename_face(old_name, new_name)
    if not done:
        raise HTTPException(status_code=404, detail=f"{old_name} not found")
    return {"message": f"Renamed {old_name} to {new_name}"}


# Webcam API Endpoints

@app.post("/webcam/start")
async def start_webcam():
    """Confirms the backend and face recognition model are ready to receive frames."""
    try:
        _ = fr.app  # verify FaceRecognitionSystem is loaded
        return {"status": "ready", "message": "Backend is ready. You can start the webcam."}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Backend not ready: {str(e)}")


# Recognition API Endpoints

@app.post("/recognize")
async def recognize(file: UploadFile):
    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    results = fr.recognize(frame, return_all=True)

    if not results:
        raise HTTPException(status_code=422, detail="No faces detected")

    return [
        {
            "name": name,
            "confidence": round(float(confidence), 4),
            "bbox": bbox.tolist()   # numpy array → plain list for JSON
        }
        for name, confidence, bbox, face in results  # face object is dropped — not JSON serializable
    ]

