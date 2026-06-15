import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Ensure relative imports work
sys.path.append(os.path.dirname(__file__))
from db_connector import DatabaseConnector

logger = logging.getLogger(__name__)

class DBAdapter:
    """
    Raw Data Adapter Layer.
    Fetches raw strings (Apartment, Cairo, etc.) from the database to feed 
    into the existing Recommender preprocessing pipeline.
    """
    
    # Required columns for 3_preprocessing.py
    REQUIRED_COLS = [
        'listing_id', 'title', 'category', 'property_type', 'offering_type', 'completion_status',
        'price_egp', 'price_period', 'city', 'town', 'district', 'subdistrict',
        'lat', 'lon', 'bedrooms', 'bathrooms', 'area_value', 'furnished',
        'listing_level', 'is_premium', 'is_featured', 'has_view_360',
        'payment_method', 'amenities', 'description', 'listed_date', 'scraped_at'
    ]

    @staticmethod
    def fetch_properties() -> pd.DataFrame:
        """
        Fetches RAW strings from TblProperties and joins with feature names.
        """
        engine = DatabaseConnector.get_engine()
        
        try:
            logger.info("Fetching RAW data from database...")

            # Detect whether the DB has rent/sale fields (backward-compatible with older DB builds).
            try:
                cols_df = pd.read_sql_query("PRAGMA table_info(TblProperties)", engine)
                prop_cols = set(cols_df["name"].astype(str).tolist())
            except Exception:
                prop_cols = set()

            category_expr = "p.Category" if "Category" in prop_cols else "'buy'"
            offering_expr = "p.OfferingType" if "OfferingType" in prop_cols else "'sale'"
            period_expr = "p.PricePeriod" if "PricePeriod" in prop_cols else "'sell'"
            
            # 1. Fetch Main Property Data (RAW STRINGS)
            query = """
            SELECT 
                p.PropertyID as listing_id,
                {category_expr} as category,
                lt.TypeName as property_type,
                {offering_expr} as offering_type,
                'completed' as completion_status,
                p.Price as price_egp,
                {period_expr} as price_period,
                c.CityName as city,
                z.ZoneName as town,
                p.Latitude as lat,
                p.Longitude as lon,
                p.BedsNo as bedrooms,
                p.BathsNo as bathrooms,
                p.Area as area_value,
                p.Title as title,
                p.District as district,
                p.Subdistrict as subdistrict,
                'unfurnished' as furnished,
                'standard' as listing_level,
                0 as is_premium,
                0 as is_featured,
                0 as has_view_360,
                'cash' as payment_method,
                p.Description as description,
                p.ListingDate as listed_date
            FROM TblProperties p
            LEFT JOIN LkpPropertyTypes lt ON p.PropertyTypeID = lt.PropertyTypeID
            LEFT JOIN TblZones z ON p.ZoneID = z.ZoneID
            LEFT JOIN TblCities c ON z.CityID = c.CityID
            WHERE p.IsDeleted = 0 OR p.IsDeleted IS NULL
            """
            prop_df = pd.read_sql_query(
                query.format(category_expr=category_expr, offering_expr=offering_expr, period_expr=period_expr),
                engine
            )
            
            # 2. Fetch Feature Names (Amenities)
            feat_query = """
            SELECT 
                m.PropertyID,
                f.FeatureName
            FROM TblPropertyFeaturesMapping m
            JOIN TblPropertyFeatures f ON m.FeatureID = f.FeatureID
            """
            feat_df = pd.read_sql_query(feat_query, engine)
            
            # 3. Aggregate Features in Python (Safer than STRING_AGG for old SQL versions)
            logger.info("Aggregating amenities...")
            amenities_map = feat_df.groupby('PropertyID')['FeatureName'].apply(lambda x: ', '.join(x)).to_dict()
            
            prop_df['amenities'] = prop_df['listing_id'].map(amenities_map).fillna('')
            prop_df['scraped_at'] = datetime.now().isoformat()
            
            # 4. Final Alignment with Preprocessing expectations
            for col in DBAdapter.REQUIRED_COLS:
                if col not in prop_df.columns:
                    prop_df[col] = ''
            
            result_df = prop_df[DBAdapter.REQUIRED_COLS]
            
            logger.info(f"Successfully fetched {len(result_df)} RAW records.")
            return result_df

        except Exception as e:
            logger.error(f"Failed to fetch raw data: {e}")
            return pd.DataFrame(columns=DBAdapter.REQUIRED_COLS)

if __name__ == "__main__":
    df = DBAdapter.fetch_properties()
    print("\nSample RAW Data (Top 2):")
    print(df[['listing_id', 'property_type', 'city', 'amenities']].head(2))
