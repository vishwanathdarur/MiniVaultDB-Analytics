# MiniVaultDB Analytics

An end-to-end data pipeline that bridges **low-level systems engineering** with **high-level data science** — using a custom LSM-tree storage engine (MiniVaultDB) as the backbone for ingestion, storage, and retrieval, then feeding the data into a full machine learning pipeline.

---

## Architecture

```
Raw CSV Dataset
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                     MiniVaultDB                         │
│                                                         │
│  CSV row ──► key-value pair ──► MemTable (in-memory)    │
│                                      │                  │
│                               WAL (durability)          │
│                                      │                  │
│                            SSTable flush (disk)         │
│                                      │                  │
│              Read path: MemTable ──► SSTables           │
└─────────────────────────────────────────────────────────┘
      │
      ▼  (retrieve → tabular)
┌─────────────────┐
│  Preprocessing  │  missing values · encoding · scaling
└─────────────────┘
      │
      ▼
┌─────────────────┐
│      EDA        │  distributions · correlations · patterns
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  Model Training │  RandomForest / LogisticRegression
└─────────────────┘
      │
      ▼
┌─────────────────┐
│   Predictions   │  new data ──► DB ──► model ──► output
└─────────────────┘
```

---

## Project Structure

```
MiniVaultDB-Analytics/
├── minivaultdb/
│   ├── adapter.py          # Thin wrapper around your MiniVaultDB engine
│   └── db.py               # ← YOUR existing MiniVaultDB code goes here
├── pipeline/
│   ├── ingest.py           # CSV → key-value → MiniVaultDB
│   ├── retrieve.py         # MiniVaultDB → pandas DataFrame
│   ├── preprocess.py       # Clean, encode, scale, train/test split
│   ├── train.py            # Train & evaluate ML model
│   └── predict.py          # End-to-end prediction on new data
├── notebooks/
│   └── eda_and_model.ipynb # Interactive EDA + model training notebook
├── data/                   # Datasets (gitignored)
├── models/                 # Saved models (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/MiniVaultDB-Analytics.git
cd MiniVaultDB-Analytics
pip install -r requirements.txt
```

### 2. Wire in your MiniVaultDB engine

Open `minivaultdb/adapter.py` and replace the import stub at the top:

```python
# Replace this:
from minivaultdb.db import MiniVaultDB   # adjust to your actual module path
```

Your engine needs to expose:

| Method | Signature | Description |
|--------|-----------|-------------|
| `put`  | `(key: str, value: str) → None` | Write a key-value pair |
| `get`  | `(key: str) → str \| None` | Read by key |
| `scan` | `() → list[tuple[str, str]]` | Return all pairs |

---

## Running the Pipeline

### Step 1 — Ingest a dataset

```bash
# Download the Titanic sample dataset and ingest it automatically
python -m pipeline.ingest --download-sample

# Or use your own CSV
python -m pipeline.ingest --csv data/your_dataset.csv --id-col PassengerId
```

### Step 2 — Retrieve and inspect

```bash
python -m pipeline.retrieve --db vault_data --out data/retrieved.csv
```

### Step 3 — Preprocess

```bash
python -m pipeline.preprocess --db vault_data --target Survived
```

### Step 4 — Train

```bash
# RandomForest (default)
python -m pipeline.train --db vault_data --target Survived --model rf

# Logistic Regression
python -m pipeline.train --db vault_data --target Survived --model lr
```

### Step 5 — Predict on new data

```bash
# Run with built-in demo records
python -m pipeline.predict --demo

# Or predict on a new CSV
python -m pipeline.predict --csv data/new_records.csv --target Survived --out data/predictions.csv
```

---

## Jupyter Notebook

For interactive EDA and model training:

```bash
cd notebooks
jupyter notebook eda_and_model.ipynb
```

The notebook covers:
- Dataset ingestion via MiniVaultDB
- Missing value analysis
- Feature distributions and correlations
- Survival rate breakdowns
- Model training, confusion matrix, ROC curve
- Feature importance chart

---

## Key Design Decisions

**Why key-value storage for tabular data?**
Each row is stored as `key → JSON string`, giving flexible schema handling and fast writes via the MemTable. The LSM-tree architecture ensures high write throughput with durability via WAL.

**Why separate ingest and predict DBs?**
The prediction pipeline writes new records to a temporary DB (`vault_predict`) to keep inference data isolated from the training data in `vault_data`.

**Why sklearn ColumnTransformer?**
It handles numeric and categorical columns in one pass, is serializable with `joblib`, and cleanly handles unseen categories at inference time via `handle_unknown='ignore'`.

---

## Example Results (Titanic)

| Metric | RandomForest |
|--------|-------------|
| Accuracy | ~0.82 |
| ROC-AUC  | ~0.87 |

Top predictive features: `Sex`, `Fare`, `Age`, `Pclass`

---

## Git Commit History (suggested)

```
feat: initial project structure and .gitignore
feat: MiniVaultDB adapter with JSON serialization
feat: data ingestion pipeline (CSV → key-value → DB)
feat: retrieval pipeline (DB → pandas DataFrame)
feat: preprocessing pipeline (impute, encode, scale)
feat: model training and evaluation (RF + LR)
feat: end-to-end prediction pipeline
feat: EDA and model training notebook
docs: README with architecture and usage
```

---

## License

MIT
