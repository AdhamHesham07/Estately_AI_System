# Estately AI System

Estately is an AI-powered Real Estate Assistant and Recommender built with LangGraph, FastAPI, and Gradio. It offers smart property searches, market analytics, and AI-driven recommendations.

## Requirements

- **Python 3.10+** (Tested on Python 3.14)

---

## 🛠️ Installation & Setup (New Machine Guide)

Follow these steps to get the project running on a brand-new machine:

### 1. Clone the Repository
```powershell
git clone <your-repo-url>
cd Estately_AI_System
```

### 2. Set Up a Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
Make sure you are in the project root folder.
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables
1. Copy the provided template to create your `.env` file:
   ```powershell
   cp .env.example .env
   ```
2. Open `.env` and fill in your API keys:
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`
   - `HUGGINGFACEHUB_API_TOKEN`

### 5. Database Setup

The system works out-of-the-box with **SQLite**.
Simply ensure `RealEstate.db` is present in `4.Data\2_DataBase\`.
*(If it's not checked into Git, you will need to copy it from your original machine or run the rebuild script if you have the source CSV).*

### 6. Run the Application

You can start the backend FastAPI server via the API batch script:
```powershell
.\5.APIs\start_api.bat
```
*(This will close any running Python instances on the server port and restart the FastAPI app on `http://127.0.0.1:8000`)*

To run the Gradio Chatbot Interface manually:
```powershell
python agent_gui.py
```
Open `http://127.0.0.1:7861` in your browser.

---

## Project Structure

- `1.Agent/` - Core LangGraph state machine, nodes, and LLM orchestration.
- `2.Recommender/` - Property retrieval, XGBoost ranking, and FAISS indexing.
- `3.Analyzer/` - Market stats evaluation and Fair Price checks.
- `4.Data/` - SQLite database connection layer.
- `5.APIs/` - FastAPI backend endpoints.
- `6.GUI/` - Vanilla HTML/JS frontend (with Alpine.js).
- `agent_gui.py` - Gradio-based frontend UI.
