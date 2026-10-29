import os
from pathlib import Path
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("ADPDS_DB_PATH", str(BASE_DIR / "adpds.db")))
engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class AnalysisRecord(Base):
    __tablename__ = "analysis_records"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    filename = Column(String(255), nullable=False)
    decision = Column(String(40), nullable=False)
    lane_status = Column(String(100), nullable=False)
    object_count = Column(Integer, nullable=False, default=0)
    summary_json = Column(Text, nullable=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
