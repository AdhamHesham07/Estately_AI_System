import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    SQLite connector for the local RealEstate.db database.
    """

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    _engine = None
    _SessionLocal = None

    @classmethod
    def _build_connection_url(cls) -> str:
        db_path = os.path.join(cls.CUR_DIR, "RealEstate.db")
        logger.info(f"DB Mode: SQLite | Path: {db_path}")
        return f"sqlite:///{db_path}"

    @classmethod
    def get_engine(cls):
        """Returns the SQLAlchemy engine, creating it once."""
        if cls._engine is None:
            try:
                connection_url = cls._build_connection_url()
                cls._engine = create_engine(connection_url, echo=False, pool_pre_ping=True)
                logger.info("DB engine initialized successfully.")
            except Exception as e:
                logger.critical(f"Failed to create DB engine: {e}", exc_info=True)
                raise
        return cls._engine

    @classmethod
    def get_session(cls):
        if cls._SessionLocal is None:
            engine = cls.get_engine()
            cls._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return cls._SessionLocal()

    @classmethod
    def reset(cls):
        """Disposes the engine and resets the singleton."""
        if cls._engine is not None:
            cls._engine.dispose()
        cls._engine = None
        cls._SessionLocal = None

    @staticmethod
    def test_connection():
        """Verifies if the connection is working."""
        try:
            session = DatabaseConnector.get_session()
            result = session.execute(text("SELECT 1")).fetchone()
            logger.info(f"DB connection test SUCCESS. Result: {result[0]}")
            session.close()
            return True
        except Exception as e:
            logger.error(f"DB connection test FAILED: {e}")
            return False


if __name__ == "__main__":
    DatabaseConnector.test_connection()
