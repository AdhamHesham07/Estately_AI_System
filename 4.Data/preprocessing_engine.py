import os
import sys
import importlib
import pandas as pd
import numpy as np

# Cross-module path handling
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "2.Recommender"))
sys.path.append(os.path.join(BASE_DIR, "4.Data", "2_DataBase"))

from db_adapter import DBAdapter
mod1 = importlib.import_module("1_config_and_cache")
CONFIG = mod1.CONFIG

# =========================================================
# DATA PREPROCESSING PIPELINE
# =========================================================
def preprocess():
    """
    Loads raw data from the SQL Database, cleans columns, handles missing values, 
    and engineers features necessary for the model.
    """
    df = DBAdapter.fetch_properties()
    required_columns = ['listing_id', 'price_egp', 'lat', 'lon', 'area_value', 'category', 'property_type']
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns from DB adapter: {missing_required}")

    numeric_columns = ['price_egp', 'area_value', 'lat', 'lon', 'bedrooms', 'bathrooms', 'images_count']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that are missing critical geographic or identity information
    df = df.dropna(subset=['listing_id', 'price_egp', 'lat', 'lon', 'area_value'])

    # Filter out non-property listings (e.g., 'project_recommendation')
    if 'listing_type' in df.columns:
        df = df[df['listing_type'] == 'property']

    # --- Intelligent Categorical Imputation ---
    # Rentals are inherently completed properties and almost exclusively paid in cash/annual checks
    if 'category' in df.columns:
        is_rent = df['category'] == 'rent'
        if 'completion_status' in df.columns:
            df.loc[is_rent & df['completion_status'].isna(), 'completion_status'] = 'completed'
        if 'payment_method' in df.columns:
            df.loc[is_rent & df['payment_method'].isna(), 'payment_method'] = 'cash'

    # --- Grouped Outlier Handling ---
    def remove_grouped_outliers(data, group_cols, target_col):
        missing_cols = [c for c in (group_cols + [target_col]) if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for outlier removal: {missing_cols}")
        grouped = data.groupby(group_cols)[target_col]
        q1 = grouped.transform(lambda x: x.quantile(0.25))
        q3 = grouped.transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        # Keep rows that are <= upper bound OR where upper bound is NaN (small groups)
        return data[(data[target_col] <= upper_bound) | (upper_bound.isna())]

    # Remove price outliers WITHIN the same category (Buy vs Rent) and property type
    df = remove_grouped_outliers(df, ['category', 'property_type'], 'price_egp')

    # Remove area outliers WITHIN the same property type
    df = remove_grouped_outliers(df, ['property_type'], 'area_value')

    # --- Smarter Numerical Imputation ---
    # Fill missing bedroom/bathroom values with the median value of properties of the same type
    for col in ['bedrooms', 'bathrooms']:
        grouped_median = df.groupby('property_type')[col].transform('median')
        df[col] = df[col].fillna(grouped_median)
        df[col] = df[col].fillna(df[col].median()).astype(int) # Fallback if entirely empty group

    # --- Feature Engineering ---
    # 1. Temporal: Days on Market
    if 'listed_date' in df.columns and 'scraped_at' in df.columns:
        df['listed_date_dt'] = pd.to_datetime(df['listed_date'], errors='coerce', utc=True)
        df['scraped_at_dt'] = pd.to_datetime(df['scraped_at'], errors='coerce', utc=True)
        df['days_on_market'] = (df['scraped_at_dt'] - df['listed_date_dt']).dt.days
        df['days_on_market'] = df['days_on_market'].clip(lower=0).fillna(0) # Prevent negatives
        df.drop(columns=['listed_date_dt', 'scraped_at_dt'], inplace=True)
    else:
        df['days_on_market'] = 0

    # 2. Listing Quality Score (Aggregated trust and visual richness)
    quality_score = np.zeros(len(df))
    if 'is_verified' in df.columns:
        quality_score += df['is_verified'].fillna(False).astype(int) * 0.3
    if 'is_premium' in df.columns:
        quality_score += df['is_premium'].fillna(False).astype(int) * 0.2
    if 'has_view_360' in df.columns:
        quality_score += df['has_view_360'].fillna(False).astype(int) * 0.2
    if 'images_count' in df.columns:
        img_score = np.clip(df['images_count'].fillna(0) / 20.0, 0, 1) * 0.3 # Max at 20 images
        quality_score += img_score
    df['listing_quality_score'] = quality_score

    # 3. Text and Economics Proxies
    df['desc_log'] = np.log1p(df['description'].fillna('').astype(str).apply(len))
    df['price_sqft'] = df['price_egp'] / df['area_value'].replace(0, np.nan).fillna(df['area_value'].median()).clip(lower=1)

    # Clean target categorical columns for the Analyzer/Recommender downstream
    for col in ['category', 'completion_status', 'furnished', 'payment_method']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str).str.lower()

    # --- Amenities Processing ---
    amenity_cols = []
    for amenity_name in CONFIG["top_amenities"]:
        safe_col_name = f"amen__{amenity_name.lower().replace(' ', '_')}"
        df[safe_col_name] = df['amenities'].fillna('').astype(str).apply(
            lambda raw_str: int(amenity_name.lower() in raw_str.lower())
        )
        amenity_cols.append(safe_col_name)

    # Safely cast listing_id as string to prevent ValueError crashes on alphanumeric IDs
    df['listing_id'] = df['listing_id'].astype(str)  

    return df.reset_index(drop=True), amenity_cols
