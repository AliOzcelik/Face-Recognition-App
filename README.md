# Face Recognition App

A FastAPI web application for face registration and recognition using [InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l` model). Faces are stored in a local SQLite database. A browser-based dashboard provides a live webcam interface.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Install the CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.
> The app auto-detects CUDA — no manual configuration needed.

---

## Running the App

### Start the backend

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive API docs (Swagger UI) at `http://localhost:8000/docs`.

### Open the dashboard

Open `dashboard.html` directly in your browser. No extra server needed.

---

## Using the Dashboard

1. **Start Webcam** — pings the backend to confirm it's ready, then requests camera access. The status line below the buttons shows the result.

2. **Register a face**
   - Click **Capture for Register** to freeze the current frame
   - Enter a name in the input field on the right
   - Click **Register** — the face embedding is saved to the database

3. **Recognize faces**
   - Click **Recognize** — sends the current frame to the backend
   - Bounding boxes with names and confidence scores are drawn on the video feed
   - Results are listed below the video

4. **Manage persons**
   - The **Registered Persons** panel on the right lists all saved names
   - Click **Delete** next to a name to remove them and all their embeddings
   - Click **Refresh** to reload the list

---

## Project Structure

```
face_recognition_app/
├── main.py               # FastAPI endpoints
├── face_recognition.py   # FaceRecognitionSystem class (InsightFace + OpenCV)
├── database_tables.py    # SQLAlchemy models and DB engine
├── dashboard.html        # Browser UI
├── API_DOCS.md           # API endpoint reference
├── requirements.txt      # Python dependencies
└── faces.db              # SQLite database (auto-created on first run)
```

---

## API Endpoints (Summary)

| Method | Path | Description |
|---|---|---|
| `POST` | `/webcam/start` | Check backend is ready |
| `GET` | `/persons` | List all registered persons |
| `POST` | `/persons/register_face` | Register a face (multipart: `name` + `file`) |
| `DELETE` | `/persons/delete_face/{name}` | Delete a person and their embeddings |
| `PATCH` | `/persons/{old_name}` | Rename a person |
| `POST` | `/recognize` | Recognize faces in an uploaded image |


---

## Model Notes

- Default model: `buffalo_l` — high accuracy, recommended for GPU
- Lighter alternative: initialize with `FaceRecognitionSystem(model_name='buffalo_s')` for CPU-only use
- Recognition threshold: `0.4` cosine distance (lower = stricter matching)
