import os
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

# Configuration
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CUR_DIR, "RealEstate.db")
CSV_PATH = os.path.join(os.path.dirname(CUR_DIR), "propertyfinder.csv")

def rebuild():
    print(f"--- [REBUILD] Starting Fresh Database Generation: {DB_PATH} ---")
    
    # 1. Initialize Connection
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Existing database file removed.")
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 2. Schema Creation
    print("Creating schema...")
    cursor.executescript("""
        CREATE TABLE TblCities (CityID INTEGER PRIMARY KEY, CityName TEXT);
        CREATE TABLE TblZones (ZoneID INTEGER PRIMARY KEY, ZoneName TEXT, CityID INTEGER);
        CREATE TABLE LkpPropertyTypes (PropertyTypeID INTEGER PRIMARY KEY, TypeName TEXT);
        CREATE TABLE TblPropertyFeatures (FeatureID INTEGER PRIMARY KEY, FeatureName TEXT);
        
        CREATE TABLE TblProperties (
            PropertyID INTEGER PRIMARY KEY AUTOINCREMENT,
            PropertyTypeID INTEGER,
            ZoneID INTEGER,
            Category TEXT,
            OfferingType TEXT,
            PricePeriod TEXT,
            Price REAL,
            BedsNo INTEGER,
            BathsNo INTEGER,
            Area REAL,
            Latitude REAL,
            Longitude REAL,
            Description TEXT,
            Title TEXT,
            District TEXT,
            Subdistrict TEXT,
            ListingDate TEXT,
            IsDeleted INTEGER DEFAULT 0,
            PropertyCode TEXT,
            Address TEXT
        );

        CREATE TABLE TblPropertyFeaturesMapping (
            PropertyID INTEGER,
            FeatureID INTEGER,
            PRIMARY KEY (PropertyID, FeatureID)
        );
    """)

    # 3. Load Source
    print(f"Loading raw CSV: {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    
    # 4. Populate Lookups
    print("Extracting Lookups...")
    
    # Property Types
    unique_types = df['property_type'].dropna().unique()
    type_data = [(i+1, str(t)) for i, t in enumerate(unique_types)]
    cursor.executemany("INSERT INTO LkpPropertyTypes (PropertyTypeID, TypeName) VALUES (?, ?)", type_data)
    type_map = {t: i+1 for i, t in enumerate(unique_types)}
    
    # Cities
    unique_cities = df['city'].dropna().unique()
    city_data = [(i+1, str(c)) for i, c in enumerate(unique_cities)]
    cursor.executemany("INSERT INTO TblCities (CityID, CityName) VALUES (?, ?)", city_data)
    city_map = {c: i+1 for i, c in enumerate(unique_cities)}
    
    # Zones (Towns)
    zone_groups = df[['city', 'town']].dropna().drop_duplicates()
    zone_data = []
    zone_map = {}
    for i, (_, row) in enumerate(zone_groups.iterrows()):
        zone_id = i + 1
        zone_data.append((zone_id, str(row['town']), city_map.get(row['city'])))
        zone_map[(str(row['city']), str(row['town']))] = zone_id
    cursor.executemany("INSERT INTO TblZones (ZoneID, ZoneName, CityID) VALUES (?, ?, ?)", zone_data)
    
    # Features (Amenities)
    all_amenities = set()
    df['amenities'].dropna().apply(lambda x: all_amenities.update([a.strip() for a in str(x).split(',')]))
    feature_data = [(i+1, str(f)) for i, f in enumerate(sorted(all_amenities))]
    cursor.executemany("INSERT INTO TblPropertyFeatures (FeatureID, FeatureName) VALUES (?, ?)", feature_data)
    feature_map = {f: i+1 for i, f in enumerate(sorted(all_amenities))}
    
    conn.commit()

    # 5. Migrate Properties
    print(f"Migrating {len(df)} properties...")
    
    def clean_num(val):
        if pd.isna(val): return 0
        nums = re.findall(r'\d+', str(val))
        return int(nums[0]) if nums else 0

    insert_sql = """
    INSERT INTO TblProperties (
        PropertyTypeID, ZoneID, Category, OfferingType, PricePeriod, Price, BedsNo, BathsNo, Area, 
        Latitude, Longitude, Description, Title, District, Subdistrict, ListingDate, PropertyCode, Address
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    # Prepare batch data
    batch_data = []
    amenity_mapping_data = []
    
    failed_rows = 0
    failed_examples = []
    # We use iterrows for mapping, but we'll use executemany for speed
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            p_type_id = type_map.get(row['property_type'], 1)
            z_id = zone_map.get((str(row['city']), str(row['town'])), 1)
            
            # Main Property Record
            prop_record = (
                p_type_id,
                z_id,
                str(row['category']).strip().lower() if 'category' in row and not pd.isna(row['category']) else "buy",
                str(row['offering_type']).strip() if 'offering_type' in row and not pd.isna(row['offering_type']) else "",
                str(row['price_period']).strip().lower() if 'price_period' in row and not pd.isna(row['price_period']) else "",
                float(row['price_egp']) if not pd.isna(row['price_egp']) else 0.0,
                clean_num(row['bedrooms']),
                clean_num(row['bathrooms']),
                float(row['area_value']) if not pd.isna(row['area_value']) else 0.0,
                float(row['lat']) if not pd.isna(row['lat']) else 0.0,
                float(row['lon']) if not pd.isna(row['lon']) else 0.0,
                str(row['description'])[:500] if not pd.isna(row['description']) else "",
                str(row['title'])[:200] if not pd.isna(row['title']) else "",
                str(row['district']) if not pd.isna(row['district']) else "",
                str(row['subdistrict']) if not pd.isna(row['subdistrict']) else "",
                str(row['listed_date']) if not pd.isna(row['listed_date']) else None,
                str(row['listing_id']),
                str(row['town'])
            )
            
            cursor.execute(insert_sql, prop_record)
            last_id = cursor.lastrowid
            
            # Amenity Mappings
            if not pd.isna(row['amenities']):
                current_amenities = [a.strip() for a in str(row['amenities']).split(',')]
                for a_name in current_amenities:
                    f_id = feature_map.get(a_name)
                    if f_id:
                        amenity_mapping_data.append((last_id, f_id))
                        
            # Commit every 1000 records to keep memory low
            if idx % 1000 == 0:
                conn.commit()
                
        except Exception as e:
            failed_rows += 1
            if len(failed_examples) < 5:
                failed_examples.append((idx, str(e)))
            continue

    # 6. Bulk Insert Amenity Mappings
    print(f"Inserting {len(amenity_mapping_data)} amenity mappings...")
    cursor.executemany("INSERT INTO TblPropertyFeaturesMapping (PropertyID, FeatureID) VALUES (?, ?)", amenity_mapping_data)
    
    conn.commit()
    conn.close()
    if failed_rows:
        print(f"!!! [REBUILD_WARN] Failed to migrate {failed_rows} rows.")
        for row_idx, err in failed_examples:
            print(f"   - row {row_idx}: {err}")
    print("--- [REBUILD COMPLETE] Database generated from scratch! ---")

if __name__ == "__main__":
    rebuild()
