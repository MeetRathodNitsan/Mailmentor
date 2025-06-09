DATABASE_URL = "postgresql://postgres:admin@localhost:5432/mailmentor"
OPENAI_API_KEY = "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"
OPENAI_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
# Add or update these settings
USE_OPENAI_API = False  # Set to False to use local models
LOCAL_MODELS = {
    "summarization": "llama3.1",
    "response": "llama3.1"
}

# Model Parameters
MODEL_PARAMS = {
    "max_length": 150,
    "min_length": 50,
    "temperature": 0.7,
    "top_p": 0.9
}

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create database engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()