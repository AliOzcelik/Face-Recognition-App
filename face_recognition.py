import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from sqlalchemy.orm import Session
from database_tables import engine, Persons, FaceEmbeddings, init_db


class FaceRecognitionSystem:

    def __init__(self, model_name='buffalo_l', ctx_id=None, det_thresh=0.5):
        # Auto-detect CUDA availability
        if ctx_id is None:
            if torch.cuda.is_available():
                ctx_id = 0
                print("CUDA detected. Using GPU.")
            else:
                ctx_id = -1
                print("CUDA not available. Using CPU.")


        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id >= 0 else ['CPUExecutionProvider']

        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=det_thresh)

        self.recognition_threshold = 0.4
        init_db()  # creates tables if they don't exist


    @staticmethod
    def _emb_to_blob(embedding: np.ndarray) -> bytes:
        """numpy float32 array  →  raw bytes for BLOB column."""
        return embedding.astype(np.float32).tobytes()



    @staticmethod
    def _blob_to_emb(blob: bytes) -> np.ndarray:
        """Raw bytes from BLOB column  →  numpy float32 array."""
        return np.frombuffer(blob, dtype=np.float32).copy()



    def _load_all_embeddings(self) -> dict:
        """Return {name: [embedding, ...]} loaded fresh from the DB."""
        with Session(engine) as session:
            rows = (
                session.query(Persons.name, FaceEmbeddings.embedding)
                .join(FaceEmbeddings, FaceEmbeddings.person_id == Persons.person_id)
                .all()
            )
        known_faces = {}
        for name, blob in rows:
            known_faces.setdefault(name, []).append(self._blob_to_emb(blob))
        return known_faces



    def _detect_faces(self, frame: np.ndarray):
        """Works on any numpy BGR frame — from webcam, video, or decoded file bytes."""
        return self.app.get(frame)

    def register_face_from_path(self, name: str, image_path: str) -> bool:
        """Thin wrapper for file-based callers."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.register_face(name, frame)

    def register_face(self, name: str, frame: np.ndarray) -> bool:
        faces = self._detect_faces(frame)

        if len(faces) == 0:
            print(f"No face detected in image for {name}")
            return False

        if len(faces) > 1:
            print(f"Multiple faces detected, using the largest one for {name}")
            faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)

        embedding = faces[0].embedding

        with Session(engine) as session:
            person = session.query(Persons).filter_by(name=name).first()
            if person is None:
                person = Persons(name=name)
                session.add(person)
                session.flush()  # populates person.person_id before we reference it

            person_id = person.person_id
            session.add(FaceEmbeddings(
                person_id=person_id,
                embedding=self._emb_to_blob(embedding),
            ))
            session.commit()

            sample_count = session.query(FaceEmbeddings).filter_by(person_id=person_id).count()

        print(f"Registered {name} (total samples: {sample_count})")
        return True



    def rename_face(self, old_name: str, new_name: str) -> bool:
        with Session(engine) as session:
            person = session.query(Persons).filter_by(name=old_name).first()
            if person is None:
                return False
            person.name = new_name
            session.commit()
        return True


    def delete_face(self, name: str) -> bool:
        """Delete a person and all their embeddings (cascade handles the rest)."""
        with Session(engine) as session:
            person = session.query(Persons).filter_by(name=name).first()
            if person is None:
                print(f"{name} not found in database")
                return False
            session.delete(person)
            session.commit()

        print(f"Deleted {name}")
        return True



    def list_faces(self) -> list[str]:
        """Return all registered person names."""
        with Session(engine) as session:
            rows = session.query(Persons.name).order_by(Persons.name).all()
        return [row.name for row in rows]



    def cosine_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        return 1 - np.dot(emb1, emb2)



    def recognize(self, image: np.ndarray, return_all: bool = False) -> list:
        """return_all: If True, include Unknown faces in results."""
        faces = self.app.get(image)
        known_faces = self._load_all_embeddings()
        results = []

        for face in faces:
            embedding = face.embedding
            bbox = face.bbox.astype(int)
            best_match = None
            best_distance = float('inf')

            for name, embeddings in known_faces.items():
                for known_emb in embeddings:
                    distance = self.cosine_distance(embedding, known_emb)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = name

            if best_distance < self.recognition_threshold:
                results.append((best_match, 1 - best_distance, bbox, face))
            elif return_all:
                results.append(("Unknown", 0, bbox, face))

        return results



    def recognize_and_draw(self, image, return_image: bool = True):
        if isinstance(image, str):
            image = cv2.imread(image)

        image = image.copy()
        results = self.recognize(image, return_all=True)

        for name, confidence, bbox, face in results:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if hasattr(face, 'kps') and face.kps is not None:
                for kp in face.kps:
                    cv2.circle(image, (int(kp[0]), int(kp[1])), 2, (255, 0, 0), -1)

        if return_image:
            return image, results
        return results
    
    def draw(self, results, image, return_image=False):


        for name, confidence, bbox, face in results:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if hasattr(face, 'kps') and face.kps is not None:
                for kp in face.kps:
                    cv2.circle(image, (int(kp[0]), int(kp[1])), 2, (255, 0, 0), -1)

        if return_image:
            return image, results
        return results
