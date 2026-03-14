# 🔍 Two-Stage Bias & Sexism Scanner (Mistral Edition)

> **Hackathon Project** — A "Filter-and-Refine" AI pipeline for educational textbook publishers to detect and neutralize biased language.

| Stage | Engine | Purpose |
|-------|--------|---------|
| **1 — Filter** | HuggingFace `valurank/distilroberta-bias` | Fast, local sentence classification |
| **2 — Refiner** | Mistral AI `mistral-large-latest` | Explain bias + suggest inclusive rewrite |

---

## 📂 Project Structure

```
Byeias/
├── configs/
│   └── config.yaml       # Configuration file including Mistral properties
├── data/
│   ├── rules/            # Custom guidelines (e.g., eu_guidelines.txt)
│   └── ...               # PDF documents to scan
├── src/
│   └── byeias/
│       ├── backend/      # FastAPI application (Stages 1 & 2)
│       └── frontend/     # React frontend prototype UI
├── app.py                # Streamlit UI
├── requirements.txt      # Python dependencies for the Streamlit app
├── pyproject.toml        # Application dependencies (poetry)
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

You can install the dependencies via Poetry:

```bash
poetry install
```

*(Note: Mistral requires `mistralai>=2.0.2`)*

### 2. Configure API Key

Edit the `configs/config.yaml` file and insert your Mistral API Key at `model.mistral.api_key`.

### 3. Start the Backend API

```bash
poetry run uvicorn byeias.backend.api:app --reload
```

The API runs at **http://localhost:8000** · Swagger docs at **http://localhost:8000/docs**

### 4. Start the Streamlit Frontend (new terminal)

```bash
poetry run streamlit run app.py
```

Opens at **http://localhost:8501**

---

## ⚛️ React Frontend Prototype (Bias Scanner UI)

For the hackathon prototype UI (Grammarly-style Bias Scanner), a dedicated React frontend is available in `src/byeias/frontend/`.

### Setup

```bash
cd src/byeias/frontend
npm install
```

### Run in Development

```bash
npm run dev
```

Vite will print the local URL (typically **http://localhost:5173**).

### Production Build

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```
