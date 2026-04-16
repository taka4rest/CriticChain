from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os

# Database Setup
DB_PATH = "sqlite:///reviews.db"
engine = create_engine(DB_PATH, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models
class ReviewSession(Base):
    __tablename__ = "review_sessions"

    id = Column(String, primary_key=True, index=True) # Changed to UUID String
    user_id = Column(String, nullable=False, default="guest_user") # Submitter
    reviewer_id = Column(String, nullable=False, default="ai_agent") # Reviewer (AI or Human)
    task_type = Column(String, nullable=False) # 'prompt' or 'code'
    submission_content = Column(Text, nullable=False)
    model_answer = Column(Text, nullable=True)
    status = Column(String, default="in_progress") # 'in_progress', 'completed', 'failed'
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("ReviewMessage", back_populates="session", cascade="all, delete-orphan")
    result = relationship("ReviewResult", back_populates="session", uselist=False, cascade="all, delete-orphan")
    step_logs = relationship("ReviewStepLog", back_populates="session", cascade="all, delete-orphan")

class ReviewMessage(Base):
    __tablename__ = "review_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("review_sessions.id"), nullable=False) # Changed FK to String
    role = Column(String, nullable=False) # 'user' or 'ai'
    content = Column(Text, nullable=False)
    step_name = Column(String, nullable=True) # e.g., 'analysis', 'refinement'
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ReviewSession", back_populates="messages")

class ReviewResult(Base):
    __tablename__ = "review_results"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("review_sessions.id"), nullable=False) # Changed FK to String
    final_comment = Column(Text, nullable=False)
    score = Column(Integer, nullable=True) # 0-100
    evaluation_result = Column(Text, nullable=True) # JSON string
    consistency_result = Column(Text, nullable=True) # JSON string
    is_manual_override = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ReviewSession", back_populates="result")

class ReviewStepLog(Base):
    __tablename__ = "review_step_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("review_sessions.id"), nullable=False) # Changed FK to String
    step_name = Column(String, nullable=False) # 'lint', 'research', 'analyze', 'fact_check', 'draft', 'critique', 'refine'
    input_data = Column(Text, nullable=True) # JSON string
    output_data = Column(Text, nullable=True) # JSON string or text
    status = Column(String, nullable=True) # 'success', 'error'
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ReviewSession", back_populates="step_logs")

# Init DB
def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized successfully.")
