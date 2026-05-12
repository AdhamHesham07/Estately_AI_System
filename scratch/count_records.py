import os
import pandas as pd
import pyodbc
from dotenv import load_dotenv

load_dotenv(".env")
server = os.getenv("DB_SERVER")
database = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASS")
driver = "{" + os.getenv("DB_DRIVER", "SQL Server") + "}"

try:
    # Check Database
    conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={user};PWD={password};TrustServerCertificate=yes;"
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM TblProperties WHERE IsDeleted = 0 OR IsDeleted IS NULL")
    db_count = cursor.fetchone()[0]
    conn.close()
    
    # Check CSV
    csv_path = r"c:\Users\Adham\Desktop\AI_System\4.Data\propertyfinder.csv"
    csv_count = len(pd.read_csv(csv_path, usecols=['listing_id']))
    
    print(f"Database (TblProperties): {db_count} active records")
    print(f"CSV File (propertyfinder.csv): {csv_count} records")
    
except Exception as e:
    print(f"Error: {e}")
