import sqlite3

DB = r"c:\Users\Adham\Desktop\AI_System\4.Data\2_DataBase\RealEstate.db"
con = sqlite3.connect(DB)
cur = con.cursor()
for col in ("town", "district", "subdistrict"):
    try:
        q = f"""
        SELECT DISTINCT {col} FROM properties
        WHERE lower(CAST({col} AS TEXT)) LIKE '%5th%'
           OR lower(CAST({col} AS TEXT)) LIKE '%fifth%'
        LIMIT 40
        """
        rows = cur.execute(q).fetchall()
        print(f"=== {col} ({len(rows)} samples) ===")
        for (val,) in rows[:20]:
            print(" ", val)
    except Exception as e:
        print(col, e)
con.close()
