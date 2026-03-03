import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
# override=True ensures it picks up live changes while Streamlit is running
load_dotenv(override=True)

import urllib.parse

def get_engine():
    """Create and return a SQLAlchemy engine instance."""
    # Load environment variables carefully each time to avoid Streamlit caching
    load_dotenv(override=True)
    
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASS = os.getenv("DB_PASS", "yourpassword") # User should update this!
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "cyberfin_db")
    
    # PostgreSQL connection string for SQLAlchemy
    # Quote the password to handle special characters like '@'
    encoded_pass = urllib.parse.quote_plus(DB_PASS)
    DB_URL = f"postgresql://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    return create_engine(DB_URL)

def test_connection():
    """Test if the database connection is successful."""
    try:
        # Load environment variables carefully each time to avoid Streamlit caching
        load_dotenv(override=True)
        
        DB_USER = os.getenv("DB_USER", "postgres")
        DB_PASS = os.getenv("DB_PASS", "Mskumar@05") # User should update this!
        DB_HOST = os.getenv("DB_HOST", "localhost")
        DB_PORT = os.getenv("DB_PORT", "5432")
        DB_NAME = os.getenv("DB_NAME", "cyberfin_db")
        
        # PostgreSQL connection string for SQLAlchemy
        # Quote the password to handle special characters like '@'
        encoded_pass = urllib.parse.quote_plus(DB_PASS)
        DB_URL = f"postgresql://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            print("✅ Successfully connected to PostgreSQL database!")
            return True
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return False

if __name__ == "__main__":
    test_connection()
