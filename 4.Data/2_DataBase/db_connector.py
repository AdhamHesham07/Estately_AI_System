import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

class DatabaseConnector:
    """
    Professional SQL Server Connector using SQLAlchemy.
    Handles connection pooling and safe query execution.
    """
    
    # 1. Configuration (Local SQLite)
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(CUR_DIR, "RealEstate.db")
    CONNECTION_URL = f"sqlite:///{DB_PATH}"

    _engine = None
    _SessionLocal = None

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            try:
                cls._engine = create_engine(
                    cls.CONNECTION_URL,
                    echo=False, # Set to True to see SQL logs
                    pool_pre_ping=True # Ensures connections are alive
                )
                print(f"--- [DB] Engine initialized for {cls.DB_PATH} ---")
            except Exception as e:
                print(f"!!! [DB_ERROR] Failed to create engine: {e}")
                raise
        return cls._engine

    @classmethod
    def get_session(cls):
        if cls._SessionLocal is None:
            engine = cls.get_engine()
            cls._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return cls._SessionLocal()

    @staticmethod
    def test_connection():
        """Verifies if the connection is working."""
        try:
            session = DatabaseConnector.get_session()
            result = session.execute(text("SELECT 1")).fetchone()
            print(f"--- [DB] Connection SUCCESS! ---")
            session.close()
            return True
        except Exception as e:
            print(f"!!! [DB_ERROR] Connection FAILED: {e}")
            return False

if __name__ == "__main__":
    # Diagnostic test
    DatabaseConnector.test_connection()
