import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import process, utils

load_dotenv(".env")
server = os.getenv("DB_SERVER")
database = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASS")
driver = "{" + os.getenv("DB_DRIVER", "SQL Server") + "}"

def get_mapping_logic():
    conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={user};PWD={password};TrustServerCertificate=yes;"
    conn = pyodbc.connect(conn_str)
    
    # 1. Property Types
    types_df = pd.read_sql("SELECT PropertyTypeID, TypeName FROM LkpPropertyTypes", conn)
    type_names = types_df['TypeName'].tolist()
    type_map = dict(zip(types_df['TypeName'], types_df['PropertyTypeID']))
    
    # 2. Zones
    zones_df = pd.read_sql("SELECT ZoneID, ZoneName FROM TblZones", conn)
    zone_names = zones_df['ZoneName'].tolist()
    zone_map = dict(zip(zones_df['ZoneName'], zones_df['ZoneID']))
    
    conn.close()
    
    def map_type(name):
        res = process.extractOne(name, type_names, score_cutoff=70)
        return type_map[res[0]] if res else 1 # Default to 1 (Apartment)
        
    def map_zone(name):
        res = process.extractOne(name, zone_names, score_cutoff=60)
        return zone_map[res[0]] if res else None
        
    return map_type, map_zone

if __name__ == "__main__":
    mt, mz = get_mapping_logic()
    print(f"Mapping 'Apartment': {mt('Apartment')}")
    print(f"Mapping 'New Cairo City': {mz('New Cairo City')}")
    print(f"Mapping 'Nasr City': {mz('Nasr City')}")
