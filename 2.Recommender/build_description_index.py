import logging
import os
import sys

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "2.Recommender"))

from importlib import import_module

config_cache_module = import_module("1_config_and_cache")

CONFIG = config_cache_module.CONFIG
load_cache = config_cache_module.load_cache
save_cache = config_cache_module.save_cache

DESC_FAISS_INDEX_FILE = os.path.join(CONFIG["paths"]["artifacts"], "description_faiss.index")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_description_index() -> None:
    """
    Builds the optional real-description semantic index outside the live API path.
    Run this after training artifacts exist, or whenever descriptions change.
    """
    stage1_cache = load_cache("stage1_data.joblib")
    if not stage1_cache:
        raise FileNotFoundError("Recommender cache 'stage1_data.joblib' not found. Train Recommender first.")

    training_data, _, _, _, _ = stage1_cache
    training_data = training_data.reset_index(drop=True)

    descriptions = training_data["description"].fillna("").astype(str).tolist()
    listing_ids = training_data["listing_id"].tolist()

    logger.info("Loading SentenceTransformer model: %s", CONFIG["model_id"])
    model = SentenceTransformer(CONFIG["model_id"])

    logger.info("Encoding %s property descriptions. This may take a while on CPU.", len(descriptions))
    embeddings = model.encode(
        descriptions,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    os.makedirs(CONFIG["paths"]["artifacts"], exist_ok=True)
    faiss.write_index(index, DESC_FAISS_INDEX_FILE)
    save_cache(listing_ids, "description_ids.joblib")

    logger.info("Saved description FAISS index to: %s", DESC_FAISS_INDEX_FILE)
    logger.info("Saved description IDs to cache: description_ids.joblib")


if __name__ == "__main__":
    build_description_index()
