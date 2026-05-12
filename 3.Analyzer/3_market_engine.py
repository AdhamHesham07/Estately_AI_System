import numpy as np
import pandas as pd
import importlib
from scipy import stats
import math

# Dynamically load the Ingestion Adapter to pull the clean data contract
ingestion_module = importlib.import_module("2_ingestion_adapter")
get_clean_data_contract = ingestion_module.get_ingested_catalog

config_module = importlib.import_module("1_config")
ANALYZER_CONFIG = config_module.ANALYZER_CONFIG

# =========================================================
# LAYER 1: GENERAL MARKET ENGINE
# =========================================================

class FairPriceEstimator:
    """
    1.4 Fair Price Estimator
    A statistically rigorous engine that determines if a property's asking price 
    is mathematically competitive within its hyper-local micro-market.
    Includes a Confidence Score (0-1) to warn users if data is too sparse.
    """
    @staticmethod
    def estimate(category_intent: str, town_name: str, district_name: str, property_type: str, number_of_bedrooms: int, asking_price: float, property_area_sqm: float, is_furnished: bool = False, source_dataframe: pd.DataFrame = None) -> dict:
        
        # Pull the clean global dataset unless a specific subset was passed in
        analytical_dataframe = source_dataframe if source_dataframe is not None else get_clean_data_contract()
        minimum_sample_size_threshold = ANALYZER_CONFIG["thresholds"]["min_sample_size"]
        
        # Guardrails against mathematical impossibility (division by zero)
        if property_area_sqm is None or float(property_area_sqm) <= 0:
            return {"status": "error", "message": "Invalid area. 'property_area_sqm' must be greater than zero."}
        if asking_price is None or float(asking_price) <= 0:
            return {"status": "error", "message": "Invalid asking price. 'asking_price' must be greater than zero."}

        # Normalize all string inputs for reliable pandas masking
        normalized_category = str(category_intent).strip().lower()
        normalized_town = str(town_name).strip().lower()
        normalized_district = str(district_name).strip().lower() if district_name else ""
        normalized_property_type = str(property_type).strip().lower()
        
        target_bedrooms = int(number_of_bedrooms) if number_of_bedrooms is not None else 0
        target_area_sqm = float(property_area_sqm)
        target_asking_price = float(asking_price)
        
        # 1) Build hierarchical segment masks to drill down from the City level to the District level
        base_market_mask = (
            (analytical_dataframe['category'].astype(str).str.lower() == normalized_category) &
            (analytical_dataframe['town'].astype(str).str.lower() == normalized_town) &
            (analytical_dataframe['property_type'].astype(str).str.lower() == normalized_property_type)
        )
        
        if normalized_district:
            district_level_mask = base_market_mask & (analytical_dataframe['district'].astype(str).str.lower() == normalized_district)
        else:
            district_level_mask = base_market_mask
        
        # 2) Strict Size Bucketing (+/- 20% area), with cascading pragmatic fallbacks to prevent returning "No Data"
        area_floor, area_ceiling = target_area_sqm * 0.8, target_area_sqm * 1.2
        
        # Attempt 1: Perfect Match (Same district, same beds, similar area)
        market_segment = analytical_dataframe[district_level_mask & (analytical_dataframe['bedrooms'] == target_bedrooms)]
        market_segment = market_segment[(market_segment['area_value'] >= area_floor) & (market_segment['area_value'] <= area_ceiling)]
        
        # Fallback A: Same district, same beds, relax area constraints entirely
        if len(market_segment) < 5:
            market_segment = analytical_dataframe[district_level_mask & (analytical_dataframe['bedrooms'] == target_bedrooms)]
            
        # Fallback B: Zoom out to the Town level, same property type, same beds
        if len(market_segment) < 5:
            market_segment = analytical_dataframe[base_market_mask & (analytical_dataframe['bedrooms'] == target_bedrooms)]
            
        # Fallback C: Town level, but allow +/- 1 bedroom difference (e.g. comparing a large 2-bed to a small 3-bed)
        if len(market_segment) < 5:
            market_segment = analytical_dataframe[base_market_mask & (analytical_dataframe['bedrooms'].between(max(0, target_bedrooms - 1), target_bedrooms + 1))]

        # 3) Apples-to-Apples Furnished Filter (A furnished apartment costs drastically more than an unfurnished one)
        furnished_binary_target = 1 if is_furnished else 0
        if 'furnished_bin' in market_segment.columns:
            market_segment = market_segment[market_segment['furnished_bin'] == furnished_binary_target]
        
        if len(market_segment) < 3:
            return {"status": "error", "message": "Extremely low data. Market too niche to analyze."}
            
        market_segment = market_segment.copy()
        market_segment['price_egp'] = pd.to_numeric(market_segment['price_egp'], errors='coerce')
        market_segment['area_value'] = pd.to_numeric(market_segment['area_value'], errors='coerce')
        market_segment = market_segment[(market_segment['price_egp'] > 0) & (market_segment['area_value'] > 0)]
        
        # Recompute the Price-Per-Square-Foot entirely from scratch to guarantee analytical integrity
        market_segment['unit_price_per_sqm'] = market_segment['price_egp'] / market_segment['area_value']
        market_segment = market_segment[np.isfinite(market_segment['unit_price_per_sqm']) & (market_segment['unit_price_per_sqm'] > 0)]
        
        if len(market_segment) < 3:
            return {"status": "error", "message": "Insufficient clean pricing signal for this market segment."}

        # Mathematical Trimming: Remove the top and bottom 5% implausible tails before applying robust IQR
        percentile_5th = market_segment['unit_price_per_sqm'].quantile(0.05)
        percentile_95th = market_segment['unit_price_per_sqm'].quantile(0.95)
        market_segment = market_segment[(market_segment['unit_price_per_sqm'] >= percentile_5th) & (market_segment['unit_price_per_sqm'] <= percentile_95th)]
        
        extracted_unit_prices = market_segment['unit_price_per_sqm'].values
        if len(extracted_unit_prices) < 3:
            return {"status": "error", "message": "Insufficient clean pricing signal for this market segment."}
        
        # 4) IQR Outlier Removal (Interquartile Range is vastly superior to Z-scores for skewed real estate data)
        quartile_1, quartile_3 = np.percentile(extracted_unit_prices, 25), np.percentile(extracted_unit_prices, 75)
        interquartile_range = quartile_3 - quartile_1
        iqr_lower_bound = quartile_1 - (1.5 * interquartile_range)
        iqr_upper_bound = quartile_3 + (1.5 * interquartile_range)
        
        extracted_unit_prices = extracted_unit_prices[(extracted_unit_prices >= iqr_lower_bound) & (extracted_unit_prices <= iqr_upper_bound)]
        
        if len(extracted_unit_prices) < 3:
            extracted_unit_prices = market_segment['unit_price_per_sqm'].values # Fallback if IQR clips too aggressively
            
        # 5) Calculate the core statistical anchors
        segment_median_unit_price = np.median(extracted_unit_prices)
        segment_q1_unit_price = np.percentile(extracted_unit_prices, 25)
        segment_q3_unit_price = np.percentile(extracted_unit_prices, 75)
        
        # The ultimate "Fair Value" prediction (Median Unit Price * Target Area)
        predicted_total_fair_price = segment_median_unit_price * target_area_sqm
        
        # Provide the raw median of the segment as context
        segment_raw_median_price = float(pd.to_numeric(market_segment['price_egp'], errors='coerce').median())
        if math.isnan(segment_raw_median_price):
            segment_raw_median_price = float(predicted_total_fair_price)
        
        # 6) Confidence Score Calculation
        # Penalizes the score if the sample size is low, OR if the data variance (standard deviation) is dangerously high
        sample_size_confidence = min(len(extracted_unit_prices) / (minimum_sample_size_threshold * 2), 1.0)
        relative_standard_deviation = np.std(extracted_unit_prices) / (segment_median_unit_price + 1e-9)
        variance_confidence = max(1.0 - (relative_standard_deviation * 2), 0.0) 
        final_confidence_score = (sample_size_confidence * 0.5) + (variance_confidence * 0.5)
        
        # 7) Verdict Generation
        target_asking_unit_price = target_asking_price / target_area_sqm
        statistical_percentile = stats.percentileofscore(extracted_unit_prices, target_asking_unit_price)
        
        if target_asking_unit_price < segment_q1_unit_price: 
            final_verdict = "highly_competitive"
        elif target_asking_unit_price <= segment_median_unit_price: 
            final_verdict = "fair_value"
        elif target_asking_unit_price <= segment_q3_unit_price: 
            final_verdict = "premium_priced"
        else: 
            final_verdict = "overpriced"
            
        return {
            "status": "success",
            "query": {"category": category_intent, "town": town_name, "district": district_name, "property_type": property_type, "bedrooms": target_bedrooms, "asking_price": target_asking_price, "area": target_area_sqm},
            "market_stats": {
                "comparable_listings_count": int(len(extracted_unit_prices)),
                "market_median_sqft": float(segment_median_unit_price),
                "predicted_fair_price": float(predicted_total_fair_price),
                "median_price": float(segment_raw_median_price),
                "price_percentile": float(round(statistical_percentile, 1)),
                "confidence_score": round(final_confidence_score, 2)
            },
            "verdict": final_verdict
        }

class InvestmentScorer:
    """
    2.2 Investment Scouter
    Actively hunts through the dataset to identify properties priced significantly 
    below the market median, flagging them as potential investment "Gems".
    """
    @staticmethod
    def identify_gems(category_intent: str, town_name: str, property_type: str = None, top_results_limit: int = 5) -> dict:
        analytical_dataframe = get_clean_data_contract()
        
        market_mask = (analytical_dataframe['category'] == category_intent.lower()) & (analytical_dataframe['town'].str.lower() == town_name.lower())
        if property_type:
            market_mask = market_mask & (analytical_dataframe['property_type'].str.lower() == property_type.lower())
            
        market_segment = analytical_dataframe[market_mask].copy()
        
        if len(market_segment) < 10:
            return {"status": "error", "message": "Insufficient data to scout investment gems."}
            
        # Group by Bedroom count to handle area variance safely (a 5-bed will naturally cost more than a 1-bed)
        bedroom_median_price_map = market_segment.groupby('bedrooms')['price_egp'].median().to_dict()
        market_segment['median_price_for_bedrooms'] = market_segment['bedrooms'].map(bedroom_median_price_map)
        
        # Mathematical Undervaluation = (Market Median - Actual Price) / Market Median
        market_segment['undervalue_percentage'] = (market_segment['median_price_for_bedrooms'] - market_segment['price_egp']) / market_segment['median_price_for_bedrooms']
        
        # Filter for properties that are at least 15% cheaper than the median for their size
        investment_gems = market_segment[market_segment['undervalue_percentage'] > 0.15].sort_values(by='undervalue_percentage', ascending=False)
        
        return {
            "status": "success",
            "market_median": market_segment['price_egp'].median(),
            "gems": investment_gems.head(top_results_limit)[['listing_id', 'price_egp', 'bedrooms', 'undervalue_percentage']].to_dict('records')
        }

class MarketPulse:
    """
    1.1 Market Pulse Dashboard
    Generates a high-level macroscopic snapshot of the overall real estate market, 
    calculating total volumes, medians, and liquidity velocity.
    """
    @staticmethod
    def get_snapshot() -> dict:
        global_dataframe = get_clean_data_contract()
        
        buyers_market_dataframe = global_dataframe[global_dataframe['category'] == 'buy']
        renters_market_dataframe = global_dataframe[global_dataframe['category'] == 'rent']
        
        # Liquidity Velocity Metric (How fast are properties moving off the market?)
        median_buy_days_on_market = buyers_market_dataframe['days_on_market'].median() if 'days_on_market' in buyers_market_dataframe else None
        median_rent_days_on_market = renters_market_dataframe['days_on_market'].median() if 'days_on_market' in renters_market_dataframe else None
        
        # Macroscopic pricing medians
        global_buy_median = buyers_market_dataframe['price_egp'].median() if len(buyers_market_dataframe) else None
        global_rent_median = renters_market_dataframe['price_egp'].median() if len(renters_market_dataframe) else None
        global_buy_unit_price = buyers_market_dataframe['price_sqft'].median() if len(buyers_market_dataframe) and 'price_sqft' in buyers_market_dataframe else None
        global_rent_unit_price = renters_market_dataframe['price_sqft'].median() if len(renters_market_dataframe) and 'price_sqft' in renters_market_dataframe else None
        
        # Quality Segment Analysis (What percentage of the market meets luxury standards?)
        if 'listing_quality_score' in global_dataframe:
            quality_score_column = global_dataframe['listing_quality_score']
            premium_market_percentage = (len(global_dataframe[quality_score_column > 0.7]) / len(global_dataframe)) * 100 if len(global_dataframe) else 0
        else:
            premium_market_percentage = 0
        
        # Analyze financing distribution for BUYERS only (Renters are inherently cash)
        buyer_financing_distribution = buyers_market_dataframe['payment_method'].value_counts(normalize=True).to_dict() if 'payment_method' in buyers_market_dataframe else {}
        
        # Identify the hottest geographical hubs
        top_active_cities = global_dataframe['city'].value_counts().head(3).to_dict()
        
        return {
            "market_volume": {
                "total_active_listings": len(global_dataframe),
                "buy_count": len(buyers_market_dataframe),
                "rent_count": len(renters_market_dataframe)
            },
            "price_economics": {
                "median_buy_price": float(global_buy_median) if global_buy_median is not None and not pd.isna(global_buy_median) else None,
                "median_rent_price": float(global_rent_median) if global_rent_median is not None and not pd.isna(global_rent_median) else None,
                "avg_buy_price_sqm": float(global_buy_unit_price) if global_buy_unit_price is not None and not pd.isna(global_buy_unit_price) else None,
                "avg_rent_price_sqm": float(global_rent_unit_price) if global_rent_unit_price is not None and not pd.isna(global_rent_unit_price) else None
            },
            "market_velocity": {
                "median_days_on_market_buy": float(median_buy_days_on_market) if median_buy_days_on_market is not None else None,
                "median_days_on_market_rent": float(median_rent_days_on_market) if median_rent_days_on_market is not None else None
            },
            "market_health": {
                "premium_verified_listings_pct": round(premium_market_percentage, 1),
                "buyer_financing_mix": {str(method): round(percentage*100, 1) for method, percentage in buyer_financing_distribution.items()}
            },
            "top_hubs": top_active_cities
        }

class AreaComparator:
    """
    1.3 Area Comparator
    A statistical module that compares two rival towns/areas side-by-side, 
    ensuring they are compared within the strict boundaries of a single category and property type.
    """
    @staticmethod
    def compare(category_intent: str, town_a_name: str, town_b_name: str, property_type: str = None) -> dict:
        analytical_dataframe = get_clean_data_contract()
        
        # Establish the base market slice
        base_market_mask = analytical_dataframe['category'] == category_intent.lower()
        if property_type:
            base_market_mask = base_market_mask & (analytical_dataframe['property_type'].str.lower() == property_type.lower())
            
        market_segment = analytical_dataframe[base_market_mask]
        
        # Extract the two competing datasets
        town_a_dataset = market_segment[market_segment['town'].str.lower() == town_a_name.lower()]
        town_b_dataset = market_segment[market_segment['town'].str.lower() == town_b_name.lower()]
        
        def _compute_regional_statistics(data_subset):
            if len(data_subset) == 0: return None
            return {
                "median_price": float(data_subset['price_egp'].median()),
                "avg_price_sqm": float(data_subset['price_sqft'].median()) if 'price_sqft' in data_subset else None,
                "listing_count": len(data_subset),
                "avg_bedrooms": float(data_subset['bedrooms'].mean()),
                "median_days_on_market": float(data_subset['days_on_market'].median()) if 'days_on_market' in data_subset else None,
                "off_plan_pct": float((data_subset['completion_status'] == 'off_plan').mean() * 100) if 'completion_status' in data_subset else 0.0
            }
            
        town_a_statistics = _compute_regional_statistics(town_a_dataset)
        town_b_statistics = _compute_regional_statistics(town_b_dataset)
        
        if not town_a_statistics or not town_b_statistics:
            return {"status": "error", "message": "Insufficient data to mathematically compare these two areas for the given category/type."}
            
        return {
            "status": "success",
            "query": {"category": category_intent, "town_a": town_a_name, "town_b": town_b_name, "property_type": property_type},
            "comparison": {
                "median_price": {
                    "a": town_a_statistics["median_price"], 
                    "b": town_b_statistics["median_price"], 
                    "diff_pct": round((town_a_statistics["median_price"] - town_b_statistics["median_price"]) / town_b_statistics["median_price"] * 100, 1)
                },
                "listing_count": {
                    "a": town_a_statistics["listing_count"], 
                    "b": town_b_statistics["listing_count"]
                },
                "median_days_on_market": {
                    "a": town_a_statistics["median_days_on_market"], 
                    "b": town_b_statistics["median_days_on_market"]
                },
                "off_plan_pct": {
                    "a": round(town_a_statistics["off_plan_pct"], 1), 
                    "b": round(town_b_statistics["off_plan_pct"], 1)
                }
            }
        }

if __name__ == "__main__":
    # Test the estimators when the module is executed directly
    print("--- Testing Fair Price Estimator ---")
    estimator_result = FairPriceEstimator.estimate(
        category_intent="buy",
        town_name="New Cairo City",
        district_name="",
        property_type="Apartment",
        number_of_bedrooms=3,
        asking_price=5200000,
        property_area_sqm=150
    )
    print(estimator_result)
    
    print("\n--- Testing Market Pulse ---")
    pulse_snapshot = MarketPulse.get_snapshot()
    import json
    print(json.dumps(pulse_snapshot, indent=2))
    
    print("\n--- Testing Area Comparator ---")
    comparison_result = AreaComparator.compare("buy", "New Cairo City", "Sheikh Zayed City")
    print(json.dumps(comparison_result, indent=2))
