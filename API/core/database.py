from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# Read variables from .env
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")

# Encode password safely (handles @, #, !, etc.)
safe_password = quote_plus(POSTGRES_PASSWORD or "")

# Build the full database URL
DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{safe_password}@"
    f"{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}?client_encoding=utf8"
)

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Test connection (only runs when executing this file directly)
if __name__ == "__main__":
    print("🔍 Checking database connection...")
    print("DATABASE_URL:", DATABASE_URL)
    print("User:", POSTGRES_USER)
    print("DB:", POSTGRES_DB)
    print("Server:", POSTGRES_SERVER)
    print("Port:", POSTGRES_PORT)

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            print("Database connection successful!")
    except Exception as e:
        print("Database connection failed!")
        print("Error:", e)
