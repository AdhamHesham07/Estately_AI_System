import numpy as np
import pandas as pd
import importlib
from tqdm import tqdm
import time

# Dynamic Imports to load configuration and core modules
config_module = importlib.import_module("1_config")
market_engine_module = importlib.import_module("3_market_engine")
ingestion_adapter_module = importlib.import_module("2_ingestion_adapter")

ANALYZER_CONFIG = config_module.ANALYZER_CONFIG
FairPriceEstimator = market_engine_module.FairPriceEstimator
get_global_clean_data = ingestion_adapter_module.get_ingested_catalog

def run_backtest(number_of_samples=100):
    """
    Backtests the FairPriceEstimator using strict Leave-One-Out (LOO) validation.
    This simulates how accurate the estimator is on properties it has never seen before.
    It measures the MdAPE (Median Absolute Percentage Error) to determine statistical reliability.
    """
    global_database_dataframe = get_global_clean_data()
    pipeline_logger = ingestion_adapter_module.ingestion_logger
    
    pipeline_logger.info(f"--- Starting Analyzer Backtest ({number_of_samples} samples) ---")
    
    # Select samples with enough neighbors to be statistically fair
    # (Only test properties located in Towns with at least 50 historical listings)
    town_listing_counts = global_database_dataframe['town'].value_counts()
    statistically_viable_towns = town_listing_counts[town_listing_counts >= 50].index.tolist()
    
    viable_test_pool = global_database_dataframe[global_database_dataframe['town'].isin(statistically_viable_towns)]
    
    # Safely clamp the sample size if the database is unexpectedly small
    if len(viable_test_pool) < number_of_samples:
        number_of_samples = len(viable_test_pool)
        
    randomized_test_samples = viable_test_pool.sample(number_of_samples, random_state=42)
    
    absolute_percentage_errors = []
    generated_confidence_scores = []
    successful_estimation_count = 0
    
    start_timestamp = time.time()
    
    for _, test_row in tqdm(randomized_test_samples.iterrows(), total=len(randomized_test_samples)):
        # Leave-One-Out: Remove the exact property we are trying to predict from the dataset
        leave_one_out_dataframe = global_database_dataframe[global_database_dataframe['listing_id'] != test_row['listing_id']]
        
        # Attempt to estimate the fair price for this property as if it were a brand new listing
        estimation_result = FairPriceEstimator.estimate(
            category_intent=test_row['category'],
            town_name=test_row['town'],
            district_name=test_row.get('district', None),
            property_type=test_row['property_type'],
            number_of_bedrooms=test_row['bedrooms'],
            asking_price=test_row['price_egp'],
            property_area_sqm=test_row['area_value'],
            is_furnished=(str(test_row.get('furnished', '')).lower() == 'furnished'),
            source_dataframe=leave_one_out_dataframe
        )
        
        if estimation_result['status'] == 'success':
            successful_estimation_count += 1
            actual_ground_truth_price = test_row['price_egp']
            predicted_fair_price = estimation_result['market_stats']['predicted_fair_price']
            
            # Calculate Absolute Percentage Error for this specific prediction
            absolute_percentage_error = abs(actual_ground_truth_price - predicted_fair_price) / (actual_ground_truth_price + 1e-9)
            
            absolute_percentage_errors.append(absolute_percentage_error)
            generated_confidence_scores.append(estimation_result['market_stats']['confidence_score'])
            
    total_duration_seconds = time.time() - start_timestamp
    
    # We use Median APE instead of Mean APE because real estate pricing often has 
    # massive luxury outliers that skew arithmetic averages
    median_absolute_percentage_error = np.median(absolute_percentage_errors) if absolute_percentage_errors else 1.0 
    average_confidence_score = np.mean(generated_confidence_scores) if generated_confidence_scores else 0.0
    segment_coverage_percentage = (successful_estimation_count / number_of_samples) * 100
    
    print("\n" + "="*50)
    print("      ANALYZER QUALITY SCOREBOARD (v1.1)")
    print("="*50)
    print(f"Price Precision (MdAPE):      {median_absolute_percentage_error*100:.2f}%  (Target: <15%)")
    print(f"Estimation Confidence:        {average_confidence_score*100:.1f}%")
    print(f"Market Segment Coverage:      {segment_coverage_percentage:.1f}%")
    print(f"Avg Latency per Estimate:     {total_duration_seconds/number_of_samples*1000:.2f} ms")
    print("="*50)
    
    # Verify if the module meets production standards
    if median_absolute_percentage_error < ANALYZER_CONFIG["thresholds"]["mape_target"]:
        print("VERDICT: [PASS] Analyzer is statistically reliable.")
    else:
        print("VERDICT: [FAIL] Analyzer error exceeds target threshold.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_backtest(100)
