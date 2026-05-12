import os
import pandas as pd
import pyodbc
from tqdm import tqdm
from dotenv import load_dotenv
from rapidfuzz import process, utils

load_dotenv(".env")
server = os.getenv("DB_SERVER")
database = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASS")
driver = "{ODBC Driver 17 for SQL Server}" # Upgraded from legacy driver

def migrate():
    print("--- [MIGRATION] Initializing Pipeline ---")
    
    # 1. Connect and Load Lookups
    conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={user};PWD={password};TrustServerCertificate=yes;"
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.fast_executemany = True
    
    types_df = pd.read_sql("SELECT PropertyTypeID, TypeName FROM LkpPropertyTypes", conn)
    type_names = types_df['TypeName'].tolist()
    type_map = dict(zip(types_df['TypeName'], types_df['PropertyTypeID']))
    
    zones_df = pd.read_sql("SELECT ZoneID, ZoneName FROM TblZones", conn)
    zone_names = zones_df['ZoneName'].tolist()
    zone_map = dict(zip(zones_df['ZoneName'], zones_df['ZoneID']))
    
    # 2. Load Source CSV
    print("--- [MIGRATION] Loading CSV Source (39k records) ---")
    csv_path = r"c:\Users\Adham\Desktop\AI_System\4.Data\propertyfinder.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    
    # 3. Clean and Transform
    print("--- [MIGRATION] Transforming Schema... ---")
    
    def get_type_id(row):
        name = str(row['property_type'])
        res = process.extractOne(name, type_names, score_cutoff=70)
        return type_map[res[0]] if res else 1
        
    def get_zone_id(row):
        name = str(row['town'])
        res = process.extractOne(name, zone_names, score_cutoff=60)
        return zone_map[res[0]] if res else None

    # Pre-calculate IDs to avoid per-row overhead in insert loop
    print("Mapping Locations (this may take a minute)...")
    df['mapped_type_id'] = df.apply(get_type_id, axis=1)
    df['mapped_zone_id'] = df.apply(get_zone_id, axis=1)
    
    # Drop records with no zone mapping
    df = df.dropna(subset=['mapped_zone_id'])
    
    # 4. Bulk Insert
    print(f"--- [MIGRATION] Inserting {len(df)} records into SQL Server... ---")
    
    insert_sql = """
    INSERT INTO TblProperties (
        PropertyTypeID, ZoneID, Price, BedsNo, BathsNo, Area, 
        Latitude, Longitude, Description, ListingDate, IsDeleted,
        PropertyCode, Address
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
    """
    
    batch_size = 500
    total_inserted = 0
    
    for i in tqdm(range(0, len(df), batch_size)):
        chunk = df.iloc[i : i + batch_size]
        data = []
        for _, row in chunk.iterrows():
            # Robustly extract numbers from strings like '7+' or 'Studio'
            def clean_num(val):
                if pd.isna(val): return 0
                import re
                nums = re.findall(r'\d+', str(val))
                return int(nums[0]) if nums else 0

            data.append((
                int(row['mapped_type_id']),
                int(row['mapped_zone_id']),
                float(row['price_egp']),
                clean_num(row['bedrooms']),
                clean_num(row['bathrooms']),
                float(row['area_value']),
                float(row['lat']),
                float(row['lon']),
                str(row['description'])[:90] if not pd.isna(row['description']) else "",
                str(row['listed_date']) if not pd.isna(row['listed_date']) else None,
                str(row['listing_id']), # PropertyCode
                str(row['town']) # Address
            ))
        
        try:
            cursor.executemany(insert_sql, data)
            conn.commit()
            total_inserted += len(data)
        except Exception as e:
            # If batch fails, try inserting row by row to skip only bad records
            print(f"Batch {i} failed, switching to row-by-row fallback...")
            for row_data in data:
                try:
                    cursor.execute(insert_sql, row_data)
                    conn.commit()
                    total_inserted += 1
                except:
                    conn.rollback()
                    continue
            
    conn.close()
    print(f"--- [MIGRATION COMPLETE] Total Records Added: {total_inserted} ---")

if __name__ == "__main__":
    migrate()
