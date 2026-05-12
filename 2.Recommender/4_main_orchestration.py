import os
import logging
import importlib
import numpy as np
import pandas as pd
import faiss
import xgboost as xgb
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# System path configurations
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "4.Data"))

# ---------------------------------------------------------
# DYNAMIC IMPORTS
# ---------------------------------------------------------
config_module = importlib.import_module("1_config_and_cache")
features_module = importlib.import_module("2_math_and_features")
preprocessing_module = importlib.import_module("preprocessing_engine")
teacher_module = importlib.import_module("3_teacher")

CONFIG = config_module.CONFIG
save_cache = config_module.save_cache
load_cache = config_module.load_cache

FeatureStore = features_module.FeatureStore
FeatureEngine = features_module.FeatureEngine
euclidean_distance_vectorized = features_module.euclidean_distance_vectorized
normalize_query = features_module.normalize_query
get_type_sim = features_module.get_type_sim
calculate_jaccard = features_module.calculate_jaccard

preprocess_data = preprocessing_module.preprocess
get_synthetic_persona_labels = teacher_module.get_spm_labels

# Initialize logging to track the progress of the massive multi-stage pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pipeline_logger = logging.getLogger(__name__)

# =========================================================
# HELPER: STRATEGIC EXPLORATION SAMPLER
# =========================================================
def get_exploration_candidates(query_row_dictionary, database_pool, number_of_samples=50):
    """
    Retrieves semi-relevant exploratory candidates from the exact same city as the query.
    This injects "discovery" properties into the training pipeline, expanding the model's 
    catalog coverage without introducing overwhelmingly irrelevant noise (like a different country).
    """
    city_filtered_pool = database_pool[database_pool['city'] == query_row_dictionary['city']]
    if len(city_filtered_pool) < number_of_samples:
        return database_pool.sample(n=number_of_samples, replace=True)
    return city_filtered_pool.sample(n=number_of_samples)

# =========================================================
# MAIN PIPELINE ORCHESTRATION (v9.0 Strategic Exploration)
# =========================================================
def execute_main_training_pipeline():
    """
    The master orchestrator that builds the Recommender model from scratch.
    It follows a strict 5-stage progression:
    1. Text Encoding (Sentence Transformers)
    2. Vector Indexing (FAISS)
    3. Synthetic Label Generation (Persona Teacher)
    4. Gradient Boosted Ranking (XGBoost)
    5. Final System Validation & Evaluation Metrics
    """
    np.random.seed(42) # Lock seed for deterministic, repeatable training
    os.makedirs(CONFIG["paths"]["cache"], exist_ok=True)

    # ---------------------------------------------------------
    # Stage 1: Data Preparation & Neural Encoding
    # ---------------------------------------------------------
    stage_1_cache = load_cache("stage1_data.joblib")
    if stage_1_cache:
        pipeline_logger.info("Stage 1 Checkpoint detected. Skipping heavy neural text encoding...")
        dataframe_train, dataframe_test, embedding_matrix_train, embedding_matrix_test, tracked_amenity_cols = stage_1_cache
        
        # CRITICAL: Reset pandas indexes to strictly align with the 0-indexed numpy arrays of embeddings
        dataframe_train = dataframe_train.reset_index(drop=True)
        dataframe_test = dataframe_test.reset_index(drop=True)
    else:
        pipeline_logger.info("Stage 1: Preprocessing Data and Generating Neural Embeddings...")
        raw_dataframe, tracked_amenity_cols = preprocess_data()
        dataframe_train, dataframe_test = train_test_split(raw_dataframe, test_size=0.15, random_state=42)
        
        dataframe_train = dataframe_train.reset_index(drop=True)
        dataframe_test = dataframe_test.reset_index(drop=True)
        
        # Load the heavy NLP model to convert text descriptions into high-dimensional space
        nlp_model = SentenceTransformer(CONFIG["model_id"])
        training_text_corpus = (dataframe_train['title'] + " " + dataframe_train['description']).tolist()
        testing_text_corpus = (dataframe_test['title'] + " " + dataframe_test['description']).tolist()
        
        embedding_matrix_train = nlp_model.encode(training_text_corpus, normalize_embeddings=True, show_progress_bar=True)
        embedding_matrix_test = nlp_model.encode(testing_text_corpus, normalize_embeddings=True, show_progress_bar=True)
        
        save_cache((dataframe_train, dataframe_test, embedding_matrix_train, embedding_matrix_test, tracked_amenity_cols), "stage1_data.joblib")

    # ---------------------------------------------------------
    # Stage 2: FAISS Vector Indexing
    # ---------------------------------------------------------
    faiss_index_save_path = os.path.join(CONFIG["paths"]["artifacts"], "faiss_index.bin")
    if os.path.exists(faiss_index_save_path):
        pipeline_logger.info("Stage 2: Loading existing pre-compiled FAISS vector index...")
        faiss_search_index = faiss.read_index(faiss_index_save_path)
        fast_feature_store = FeatureStore(dataframe_train, tracked_amenity_cols)
    else:
        pipeline_logger.info("Stage 2: Building and compiling FAISS Inner Product search index...")
        # Use Inner Product (IP) because embeddings were previously normalized to L2 lengths
        faiss_search_index = faiss.IndexFlatIP(embedding_matrix_train.shape[1])
        faiss_search_index = faiss.IndexIDMap2(faiss_search_index)
        faiss_search_index.add_with_ids(embedding_matrix_train.astype('float32'), np.arange(len(dataframe_train)))
        faiss.write_index(faiss_search_index, faiss_index_save_path)
        fast_feature_store = FeatureStore(dataframe_train, tracked_amenity_cols)

    # ---------------------------------------------------------
    # Stage 3: Training Vector Generation (The Strategic Funnel)
    # ---------------------------------------------------------
    stage_3_cache = load_cache("stage3_vectors.joblib")
    if stage_3_cache:
        pipeline_logger.info("Stage 3 Checkpoint detected. Skipping synthetic feature generation...")
        feature_matrix_x, target_labels_y, query_group_ids = stage_3_cache
    else:
        pipeline_logger.info("Stage 3: Generating Training Data (Semantic retrieval + Strategic exploration)...")
        feature_matrix_x, target_labels_y, query_group_ids = [], [], []
        
        if len(dataframe_train) == 0:
            raise ValueError("Training dataframe is entirely empty after preprocessing drops.")
            
        # Sample queries from the dataset to act as synthetic human users
        number_of_queries_to_simulate = min(400, len(dataframe_train))
        sampled_query_indices = np.random.choice(len(dataframe_train), number_of_queries_to_simulate, replace=False)
        listing_id_to_dataframe_index = pd.Series(dataframe_train.index.values, index=dataframe_train['listing_id']).to_dict()
        
        for current_query_group_id, global_query_index in enumerate(tqdm(sampled_query_indices)):
            active_query_row = dataframe_train.iloc[global_query_index]
            
            # --- Candidate Funnel Assembly ---
            # Part A: Retrieve mathematically similar semantic candidates via FAISS (Top 300)
            normalized_query_embedding = normalize_query(embedding_matrix_train[global_query_index])
            _, returned_faiss_indices = faiss_search_index.search(normalized_query_embedding, 300)
            
            # Filter out invalid IDs and prevent the query property from recommending itself
            valid_faiss_indices = [int(i) for i in returned_faiss_indices[0] if i >= 0 and i != global_query_index]
            semantic_candidate_dataframe = dataframe_train.iloc[valid_faiss_indices]
            
            # Part B: Retrieve random but strategically plausible exploration candidates (Top 50)
            exploration_candidate_dataframe = get_exploration_candidates(active_query_row, dataframe_train, number_of_samples=50)
            
            # Combine the pools and deduplicate
            merged_candidates_dataframe = pd.concat([semantic_candidate_dataframe, exploration_candidate_dataframe]).drop_duplicates(subset=['listing_id'])
            
            # Map candidate pandas indices back to their exact rows in the numpy embedding matrix
            candidate_numpy_positions = [listing_id_to_dataframe_index[listing_id] for listing_id in merged_candidates_dataframe['listing_id'] if listing_id in listing_id_to_dataframe_index]
            if not candidate_numpy_positions:
                continue
            aligned_candidate_embeddings = embedding_matrix_train[candidate_numpy_positions]

            # Ask the Feature Engine to build the massive engineering stack
            engineered_features_batch, amenity_jaccard_array = FeatureEngine.build_feature_stack(
                active_query_row, merged_candidates_dataframe, aligned_candidate_embeddings, embedding_matrix_train[global_query_index], fast_feature_store
            )

            # Ask the Teacher Persona to score the batch
            synthetic_labels_batch = get_synthetic_persona_labels(active_query_row, merged_candidates_dataframe, amenity_jaccard_array)

            if len(synthetic_labels_batch) == 0: 
                continue
                
            # Append everything to the master lists
            feature_matrix_x.extend(engineered_features_batch)
            target_labels_y.extend(synthetic_labels_batch)
            query_group_ids.extend([current_query_group_id] * len(synthetic_labels_batch))

        # Convert the gigantic lists into fast numpy arrays and cache them
        feature_matrix_x = np.array(feature_matrix_x)
        target_labels_y = np.array(target_labels_y)
        query_group_ids = np.array(query_group_ids)
        save_cache((feature_matrix_x, target_labels_y, query_group_ids), "stage3_vectors.joblib")

    # ---------------------------------------------------------
    # Stage 4: Training the XGBoost Ranker
    # ---------------------------------------------------------
    ranker_save_path = os.path.join(CONFIG["paths"]["artifacts"], "xgb_ranker.json")
    if os.path.exists(ranker_save_path):
        pipeline_logger.info("Stage 4: Loading previously trained XGBoost Ranker...")
        xgboost_ranker_model = xgb.XGBRanker()
        xgboost_ranker_model.load_model(ranker_save_path)
    else:
        pipeline_logger.info("Stage 4: Training and compiling the XGBoost Ranker from scratch...")
        unique_query_ids = np.unique(query_group_ids)
        
        # Split strictly by Query Groups, not randomly, to prevent data leakage across train/val splits
        train_queries, validation_queries = train_test_split(unique_query_ids, test_size=0.15, random_state=42)
        train_mask = np.isin(query_group_ids, train_queries)
        validation_mask = np.isin(query_group_ids, validation_queries)
        
        feature_x_train, target_y_train, group_id_train = feature_matrix_x[train_mask], target_labels_y[train_mask], query_group_ids[train_mask]
        feature_x_val, target_y_val, group_id_val = feature_matrix_x[validation_mask], target_labels_y[validation_mask], query_group_ids[validation_mask]
        
        # XGBoost requires groups to be strictly ordered contiguously
        order_train, order_val = np.argsort(group_id_train), np.argsort(group_id_val)
        feature_x_train, target_y_train, group_id_train = feature_x_train[order_train], target_y_train[order_train], group_id_train[order_train]
        feature_x_val, target_y_val, group_id_val = feature_x_val[order_val], target_y_val[order_val], group_id_val[order_val]

        xgboost_ranker_model = xgb.XGBRanker(
            objective='rank:ndcg', 
            n_estimators=500, 
            learning_rate=0.05, 
            early_stopping_rounds=20, 
            tree_method="hist",
            eval_metric="ndcg"
        )
        
        xgboost_ranker_model.fit(
            feature_x_train, target_y_train, 
            group=np.unique(group_id_train, return_counts=True)[1], 
            eval_set=[(feature_x_val, target_y_val)], 
            eval_group=[np.unique(group_id_val, return_counts=True)[1]], 
            verbose=10
        )
        xgboost_ranker_model.save_model(ranker_save_path)

    # ---------------------------------------------------------
    # Stage 5: Final System Evaluation
    # ---------------------------------------------------------
    pipeline_logger.info("Stage 5: Final Balanced System Evaluation Phase...")
    ndcg_metric_list, mrr_metric_list, diversity_metric_list = [], [], []
    globally_recommended_unique_set = set()

    if len(dataframe_test) == 0:
        raise ValueError("Test dataframe is entirely empty after the initial split.")
        
    evaluation_query_count = min(100, len(dataframe_test))
    listing_id_to_dataframe_index = pd.Series(dataframe_train.index.values, index=dataframe_train['listing_id']).to_dict()
    
    # Run a complete forward-pass inference on the blind holdout set
    for test_query_index in np.random.choice(len(dataframe_test), evaluation_query_count, replace=False):
        test_query_row = dataframe_test.iloc[test_query_index]
        
        # Step A: FAISS Semantic Retrieval
        normalized_test_query_vector = normalize_query(embedding_matrix_test[test_query_index])
        _, returned_faiss_indices = faiss_search_index.search(normalized_test_query_vector, 300)
        semantic_indices = [int(x) for x in returned_faiss_indices[0] if x >= 0]
        
        # Step B: Strategic Exploration Retrieval
        exploration_candidate_dataframe = get_exploration_candidates(test_query_row, dataframe_train, number_of_samples=50)
        
        merged_candidates_dataframe = pd.concat([dataframe_train.iloc[semantic_indices], exploration_candidate_dataframe]).drop_duplicates(subset=['listing_id'])
        
        candidate_numpy_positions = [listing_id_to_dataframe_index[listing_id] for listing_id in merged_candidates_dataframe['listing_id'] if listing_id in listing_id_to_dataframe_index]
        if not candidate_numpy_positions:
            continue
        aligned_candidate_embeddings = embedding_matrix_train[candidate_numpy_positions]

        # Step C: Feature Engineering
        feature_matrix_x_test, amenity_jaccard_array = FeatureEngine.build_feature_stack(
            test_query_row, merged_candidates_dataframe, aligned_candidate_embeddings, embedding_matrix_test[test_query_index], fast_feature_store
        )
        
        # Step D: XGBoost Live Inference
        model_predictions = xgboost_ranker_model.predict(feature_matrix_x_test)
        
        # Step E: Ground Truth Check
        ground_truth_labels = get_synthetic_persona_labels(test_query_row, merged_candidates_dataframe, amenity_jaccard_array)

        # Slice out the Top 10 predicted results
        top_10_indices = np.argsort(model_predictions)[::-1][:10]
        top_10_ground_truth_labels = ground_truth_labels[top_10_indices]
        top_10_dataframe_results = merged_candidates_dataframe.iloc[top_10_indices]

        # Calculate Ranking Quality (nDCG@10)
        discounted_cumulative_gain = np.sum((2**top_10_ground_truth_labels - 1) / np.log2(np.arange(2, len(top_10_ground_truth_labels) + 2)))
        ideal_discounted_cumulative_gain = np.sum((2**np.sort(ground_truth_labels)[::-1][:10] - 1) / np.log2(np.arange(2, len(top_10_ground_truth_labels) + 2)))
        if ideal_discounted_cumulative_gain > 0: 
            ndcg_metric_list.append(discounted_cumulative_gain / ideal_discounted_cumulative_gain)
            
        # Calculate Search Effectiveness (Mean Reciprocal Rank - MRR)
        perfect_matches = np.where(top_10_ground_truth_labels == 4)[0]
        mrr_metric_list.append(1.0 / (perfect_matches[0] + 1) if len(perfect_matches) > 0 else 0.0)
        
        # Calculate Spatial Diversity (Average Euclidean spread between top 10 results)
        if len(top_10_dataframe_results) > 1:
            latitudes, longitudes = top_10_dataframe_results['lat'].values, top_10_dataframe_results['lon'].values
            inter_property_distances = [euclidean_distance_vectorized(latitudes[i], longitudes[i], latitudes[j], longitudes[j]) for i in range(len(latitudes)) for j in range(i+1, len(latitudes))]
            diversity_metric_list.append(np.mean(inter_property_distances))
            
        # Track catalog footprint
        globally_recommended_unique_set.update(top_10_dataframe_results['listing_id'].tolist())

    catalog_coverage_percentage = (len(globally_recommended_unique_set) / len(dataframe_train)) * 100
    print("\n" + "="*50 + "\n      BALANCED RECOMMENDER SCOREBOARD (v9.5)\n" + "="*50)
    print(f"nDCG@10 (Ranking Quality):    {np.mean(ndcg_metric_list):.4f}")
    print(f"MRR (Search Effectiveness):   {np.mean(mrr_metric_list):.4f}")
    print(f"Spatial Spread (degrees):     {np.mean(diversity_metric_list):.4f}")
    print(f"Catalog Coverage (%):         {catalog_coverage_percentage:.2f}%\n" + "="*50)

if __name__ == "__main__":
    execute_main_training_pipeline()
