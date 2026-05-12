# Documentation: Real Estate Recommender Training Script (`model_train.py`)

## 1. Overview
This script implements a **Hybrid Recommendation System** for real estate listings. It combines two different machine learning approaches to provide balanced recommendations:
1.  **Structural Similarity (KNN):** Matches properties based on hard numbers (price, size, location, rooms).
2.  **Semantic Similarity (NLP):** Matches properties based on the "vibe" and specific features mentioned in their titles and descriptions using deep learning embeddings.

The script processes a dataset of ~34,000 listings, trains the models, and pre-computes the top 10 recommendations for every single listing to ensure instant response times in the final application.

---

## 2. Function Interfaces and Logic

### `load_and_prepare_data()`
*   **Interface:** `() -> pd.DataFrame`
*   **Logic:** 
    *   Loads the cleaned numeric data (`propertyfinder_cleaned.csv`).
    *   Loads specific text columns (`description`, `title`, `location_full`) from the raw dataset.
    *   Merges both on `listing_id` to create a unified view.
    *   Handles missing text by filling nulls with placeholders to prevents errors in the NLP model.

### `train_recommender(df)`
*   **Interface:** `(df: pd.DataFrame) -> (NearestNeighbors, pd.DataFrame, RobustScaler)`
*   **Logic:**
    *   **Feature Selection:** Extracts `price_egp`, `area_value`, `bedrooms`, `bathrooms`, `lat`, `lon`, and `amenities_count`.
    *   **Scaling:** Uses `RobustScaler` which is essential for real estate data because price and area often contain outliers (e.g., extremely luxury penthouses) that would otherwise skew a standard scaler.
    *   **Weighting:** Artificially multiplies Price (2.5x) and Area (2.0x) columns. This "stretches" the data space, forcing the KNN model to treat people's budget and space requirements as more important than the number of bathrooms or exact latitude.
    *   **Model:** Fits a `NearestNeighbors` model using Euclidean distance.

### `generate_embeddings(df)`
*   **Interface:** `(df: pd.DataFrame) -> np.ndarray`
*   **Logic:**
    *   Initializes the `paraphrase-multilingual-MiniLM-L12-v2` transformer. This model is chosen because it understands both Arabic and English (common in Egypt listings).
    *   Concatenates titles and descriptions into a single string for each listing.
    *   Converts these strings into 384-dimensional mathematical vectors (embeddings).

### `precompute_recommendations(df, knn, X_scaled, embeddings)`
*   **Interface:** `(df, knn, X_scaled, embeddings) -> pd.DataFrame`
*   **Logic:**
    *   Runs a two-stage retrieval process for every listing:
        1.  **Candidate Retrieval:** Finds the 50 closest properties using the structural KNN model.
        2.  **Semantic Reranking:** For those 50 candidates, it calculates the **Cosine Similarity** between their text embeddings and the query property.
    *   **Hybrid Scoring:** Combines the structural score (70% weight) and the semantic score (30% weight).
    *   Returns the top 10 final winners.

### `main()`
*   **Interface:** `() -> None`
*   **Logic:** The orchestration layer. It calls each function in sequence and saves the results (`.joblib` for models, `.npy` for embeddings, and `.csv` for pre-computed lists) into the `model_artifacts` folder.

---

## 3. Line-by-Line Breakdown

### Imports and Configuration
*   **L1-L11:** Imports essential libraries for data science (Pandas/Numpy), Machine Learning (Scikit-Learn), NLP (Sentence Transformers), and utilities (Joblib for saving, Logging for tracking).
*   **L14-L15:** Sets up professional logging so we can track progress in the terminal.
*   **L18-L20:** Defines absolute paths to data and where the results should be saved.

### Data Loading (`load_and_prepare_data`)
*   **L24-L25:** Reads the CSV files. `usecols` on L25 is a memory optimization—it only loads the text we actually need.
*   **L28:** A "left join" ensures we don't lose listings if some text metadata is missing.
*   **L31-L32:** Critical step for deep learning: `NaN` values will crash the Transformer model, so we fill them with "No description available".

### Structural Training (`train_recommender`)
*   **L41-L44:** Defines the "Vector Space"—the dimensions we use to measure property similarity.
*   **L51-L52:** Calculation of the Scaling. This ensures that "1,000,000 EGP" doesn't mathematically overpower "2 Bedrooms" just because the number is larger.
*   **L56-L64:** This is the "Brain" of the business logic. By setting 'price_egp' to 2.5, we say price is the most important factor in a recommendation.
*   **L70-L71:** We set `n_neighbors=50` to cast a wide net initially, filtering out the thousands of properties that definitely don't match.

### Semantic Generation (`generate_embeddings`)
*   **L77:** Loads the AI model. Multilingual support is key here for Egyptian real estate.
*   **L80:** Combines context. A property with "Sea View" in the title but not in the description will still be caught here.
*   **L84:** `batch_size=32` keeps the GPU/CPU memory usage steady during the heavy math calculations.

### Pre-computation Strategy (`precompute_recommendations`)
*   **L95:** Uses the KNN model to find the neighbors for the *entire* dataset in one highly optimized batch operation.
*   **L99:** Excludes the first neighbor because the closest property to Property A is always Property A itself.
*   **L104:** Corrects the scale of distances. Since KNN distance is "lower is better," we flip it to a score where "1.0 is a perfect match."
*   **L112-L114:** Manual Cosine Similarity calculation. We normalize the vectors and take the dot product, representing how similar the descriptions "sound" to the AI.
*   **L118:** The Final Fusion. This is where the hybrid nature is realized.
*   **L121:** Sorts and slices to get only the best of the best.

### Execution Plan (`main`)
*   **L133-L134:** Defensive programming: creates the output folder if it doesn't exist.
*   **L150-L153:** Saves every component needed for the live AI Chatbot or Website to function without retraining.
*   **L157-L159:** Global error handling ensures the script fails gracefully with a logged message if something goes wrong.
