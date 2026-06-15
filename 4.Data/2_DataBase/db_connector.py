import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    Dual-mode SQL Connector (SQLite for development / SQL Server for production).
    Controlled entirely by environment variables — no code changes needed to switch.

    To use SQL Server (SSMS), add these to your .env file:
        DB_MODE=sqlserver
        DB_SERVER=YOUR_SERVER_NAME          # e.g. localhost or DESKTOP-ABC\\SQLEXPRESS
        DB_NAME=EstatelyDB                  # your database name
        DB_DRIVER=ODBC+Driver+17+for+SQL+Server
        DB_TRUSTED=yes                      # 'yes' = Windows Auth, 'no' = use DB_USER/DB_PASS
        DB_USER=                            # only needed if DB_TRUSTED=no
        DB_PASS=                            # only needed if DB_TRUSTED=no
    """

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    _engine = None
    _SessionLocal = None

    @classmethod
    def _build_connection_url(cls) -> str:
        """
        Builds the correct SQLAlchemy connection URL based on DB_MODE env variable.
        Defaults to SQLite if DB_MODE is not set or set to 'sqlite'.
        """
        db_mode = os.getenv("DB_MODE", "sqlite").strip().lower()

        if db_mode == "sqlserver":
            server   = os.getenv("DB_SERVER", "localhost")
            database = os.getenv("DB_NAME", "EstatelyDB")
            driver   = os.getenv("DB_DRIVER", "ODBC+Driver+17+for+SQL+Server")
            trusted  = os.getenv("DB_TRUSTED", "yes").strip().lower()

            if trusted == "yes":
                # Windows Authentication — no username/password needed
                connection_url = (
                    f"mssql+pyodbc://{server}/{database}"
                    f"?driver={driver}&trusted_connection=yes"
                )
            else:
                # SQL Server Authentication
                db_user = os.getenv("DB_USER", "")
                db_pass = os.getenv("DB_PASS", "")
                connection_url = (
                    f"mssql+pyodbc://{db_user}:{db_pass}@{server}/{database}"
                    f"?driver={driver}"
                )

            logger.info(f"DB Mode: SQL Server | Server: {server} | DB: {database}")
            return connection_url

        else:
            # Default: SQLite (local development)
            db_path = os.path.join(cls.CUR_DIR, "RealEstate.db")
            logger.info(f"DB Mode: SQLite | Path: {db_path}")
            return f"sqlite:///{db_path}"

    @classmethod
    def get_engine(cls):
        """Returns the SQLAlchemy engine, creating it once (singleton pattern)."""
        if cls._engine is None:
            try:
                connection_url = cls._build_connection_url()
                # SQL Server benefits from a larger pool; SQLite only supports 1
                db_mode = os.getenv("DB_MODE", "sqlite").strip().lower()
                pool_kwargs = (
                    {"pool_size": 5, "max_overflow": 10, "pool_pre_ping": True}
                    if db_mode == "sqlserver"
                    else {"pool_pre_ping": True}
                )
                cls._engine = create_engine(connection_url, echo=False, **pool_kwargs)
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
        """Disposes the engine and resets the singleton — useful when switching DB_MODE at runtime."""
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
