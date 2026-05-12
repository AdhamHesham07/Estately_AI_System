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
    
    print("--- [TEST] Inserting 1 Record ---")
    insert_sql = """
    INSERT INTO TblProperties (
        PropertyTypeID, ZoneID, Price, BedsNo, BathsNo, Area, 
        Latitude, Longitude, Description, ListingDate, IsDeleted
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
    """
    
    # Dummy data
    data = (1, 1, 1500000.0, 3, 2, 120.0, 30.0, 31.0, "Test Mig", "2026-04-25")
    
    cursor.execute(insert_sql, data)
    conn.commit()
    
    print("--- [SUCCESS] Test record inserted! ---")
    conn.close()
except Exception as e:
    print(f"!!! [FAILED] Test insert failed: {e}")
