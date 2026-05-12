import os
import pyodbc
from dotenv import load_dotenv

load_dotenv(".env")
server = os.getenv("DB_SERVER")
database = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASS")
driver = "{ODBC Driver 17 for SQL Server}"

try:
    conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={user};PWD={password};TrustServerCertificate=yes;"
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COLUMN_NAME, IS_NULLABLE, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'TblProperties'")
    cols = cursor.fetchall()
    for c in cols:
        print(f"Column: {c[0]} | Nullable: {c[1]} | Type: {c[2]}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
