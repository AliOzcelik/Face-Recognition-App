from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, LargeBinary, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship, Session


# -------------------------------------------------------------------
# 1. Engine  — the connection to the SQLite file
# -------------------------------------------------------------------
# "sqlite:///faces.db" creates faces.db in the current working directory.
# check_same_thread=False is required when sharing the engine with FastAPI.
engine = create_engine(
    "sqlite:///faces.db",
    connect_args={"check_same_thread": False},
    echo=False,  # set True to print every SQL query (good for debugging)
)


# -------------------------------------------------------------------
# 2. Base  — every ORM model class inherits from this
# -------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# -------------------------------------------------------------------
# 3. ORM Models  (one class = one table)
# -------------------------------------------------------------------

class Person(Base):
    __tablename__ = "persons"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    name       = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # "cascade" here mirrors ON DELETE CASCADE in raw SQL:
    # deleting a Person automatically deletes all their FaceEmbedding rows.
    embeddings = relationship("FaceEmbedding", back_populates="person", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Person id={self.id} name={self.name!r}>"


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    # ondelete="CASCADE" tells the DB engine to enforce deletion at the SQL level too
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # stores a numpy array as raw bytes

    person = relationship("Person", back_populates="embeddings")

    def __repr__(self):
        return f"<FaceEmbedding id={self.id} person_id={self.person_id}>"


# -------------------------------------------------------------------
# 4. init_db()  — call once at app startup to create the tables
# -------------------------------------------------------------------
def init_db():
    Base.metadata.create_all(engine)


# -------------------------------------------------------------------
# 5. Usage examples
# -------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    init_db()  # creates faces.db with both tables if they don't exist

    with Session(engine) as session:

        # INSERT a person + one embedding
        person = Person(name="Alice")
        session.add(person)
        session.flush()          # populates person.id so the ForeignKey below is valid

        fake_embedding = np.random.rand(512).astype(np.float32)
        emb = FaceEmbedding(
            person_id=person.id,
            embedding=fake_embedding.tobytes(),  # numpy array → raw bytes → stored as BLOB
        )
        session.add(emb)
        session.commit()
        print("Inserted:", person)

        # SELECT all persons
        all_persons = session.query(Person).all()
        print("All persons:", all_persons)

        # READ embeddings back via the relationship
        alice = session.query(Person).filter_by(name="Alice").first()
        for row in alice.embeddings:
            array = np.frombuffer(row.embedding, dtype=np.float32)  # bytes → numpy array
            print(f"  embedding shape: {array.shape}")

        # DELETE — CASCADE removes FaceEmbedding rows automatically
        session.delete(alice)
        session.commit()
        print("Deleted Alice and all her embeddings.")
