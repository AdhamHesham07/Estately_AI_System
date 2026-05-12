import os
import pyodbc
from dotenv import load_dotenv

load_dotenv(".env")
server = os.getenv("DB_SERVER")
database = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASS")
driver = "{" + os.getenv("DB_DRIVER", "SQL Server") + "}"

try:
    conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={user};PWD={password};TrustServerCertificate=yes;"
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM TblProperties")
    print(f"Total: {cursor.fetchone()[0]}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
