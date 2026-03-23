from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Float, TIMESTAMP, func, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase, relationship, Session
from datetime import datetime


engine = create_engine(
    "sqlite:///faces.db",
    connect_args={"check_same_thread": False},
    echo=False,  # set True to print every SQL query (good for debugging)
)

#Base = declarative_base()
class Base(DeclarativeBase):
    pass


class Persons(Base):
    __tablename__ = 'persons'
    person_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # "cascade" here mirrors ON DELETE CASCADE in raw SQL:
    # deleting a Person automatically deletes all their FaceEmbedding rows.
    embeddings = relationship("FaceEmbeddings", back_populates="person", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Person id={self.id} name={self.name!r}>"


class FaceEmbeddings(Base):
    __tablename__ = "face_embeddings"
    face_embedding_id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("persons.person_id", ondelete="CASCADE"), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # stores a numpy array as raw bytes

    person = relationship("Persons", back_populates="embeddings")

    def __repr__(self):
        return f"<FaceEmbedding id={self.id} person_id={self.person_id}>"
    

def init_db():
    Base.metadata.create_all(engine)