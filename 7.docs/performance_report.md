# Estately AI System - Performance Report

This report details the performance benchmarking of the three core modules: Recommender, Analyzer, and Agent. 

The tests were measured across three key vectors:
- **Time (Latency):** The execution speed in seconds.
- **Quality:** Accuracy, confidence, and contextual relevance.
- **Validity:** The adherence to data contracts, strict filtering rules, and successful execution without exceptions.

---

## 1. Recommender Module

The Recommender orchestrates FAISS vector retrieval, XGBoost ranking, and MMR diversity re-ranking.

| Test Name | Latency (sec) | Quality Score | Validity | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Cold_Start_Load** | 7.262 | N/A | ✅ Pass | Initial loading of FAISS, XGBoost, and SentenceTransformers into memory. |
| **Specific_Query_Buy** | 0.567 | 0.565 (Relevance) | ⚠️ Partial | Returned 10 relevant candidates very quickly (< 600ms). Flagged as "vague" internally due to missing optional filters (e.g., amenities), triggering dynamic MMR diversity. |
| **Vague_Query_Handling** | 0.0002 | N/A | ✅ Pass | Successfully and instantly rejected an invalid/incomplete query (missing budget/location constraints) without running heavy inference. |

> [!TIP]
> **Performance Insight:** A 567ms response time for a complex semantic + geospatial + ML ranking pipeline is extremely fast. The 7.2s cold start only happens once when the server boots.

---

## 2. Analyzer Module

The Analyzer handles heavy pandas statistical aggregations against live databases (37,000+ records).

| Test Name | Latency (sec) | Quality Score | Validity | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Fair_Price_Estimator** | 3.928 | 0.62 (Confidence) | ✅ Pass | Ingested data and accurately predicted "fair_value" for a 3-bedroom apartment in New Cairo. The 62% confidence score reflects variance in that specific micro-market. |
| **Market_Pulse** | 0.197 | High | ✅ Pass | Successfully aggregated global metrics across 37,306 active listings in under 200ms. |
| **Area_Comparator** | 0.039 | N/A | ✅ Pass | Statically compared two rival cities (New Cairo vs Sheikh Zayed) almost instantaneously (39ms). |

> [!NOTE]
> **Performance Insight:** `FairPriceEstimator` takes ~3.9 seconds primarily due to the initial SQL ingestion and amenity parsing pipeline being triggered. Future requests in the same session will be much faster.

---

## 3. Agent Module (LangGraph + LLM)

The Agent manages state, routes intents, and communicates with the `llama-3.3-70b-versatile` LLM via the Groq API.

| Test Name | Latency (sec) | Quality Score | Validity | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Build_Graph** | 0.054 | N/A | ✅ Pass | Graph compilation is virtually instant. |
| **Message_1_Response** | 0.738 | "idle" (Intent) | ✅ Pass | User: *"Hi, I am looking for a house."* <br> Agent correctly classified intent as `idle` with 1.0 confidence and skipped heavy tool usage, resulting in a lightning-fast <1s response. |
| **Message_2_Response** | 12.255 | "search" (Intent) | ✅ Pass | User: *"I want a 3 bedroom... max budget 6 million."* <br> Agent successfully triggered the Recommender tool, fetched data, injected it into context, and generated a final response. |

> [!WARNING]
> **Performance Insight:** While the system logic is fast, calling the `70b` model twice in succession (once for routing/tool execution, once for generating the final textual response) inherently takes ~12 seconds. This is standard for large LLMs but is the primary bottleneck for end-user latency.
