# 🔋 Li-ion Battery Degradation Analysis

**Improved scientific study & demo — Li-ion battery degradation modeling with 5 ML models + FastAPI web interface.**

This repository contains an enhanced version of a physics research project: I trained **five** models on a larger publicly available dataset (NASA battery dataset on Kaggle), produced diagnostic plots, and added a FastAPI-based web interface to explore results and run predictions.

---

## ✨ Highlights / What’s new

- ✅ Trained on the larger **NASA Battery Dataset** (Kaggle).  
- 🤖 Trains **5 different models** (run together by `train.py`) instead of a single model — useful for model comparison and ensembling.  
- 📈 `train.py` creates models **and** plots (saved into `plots/`).  
- 🌐 FastAPI server + simple frontend to browse results and call the API.  
- 📂 Move the generated graphs in `plots/` into the `static/` folder so they are served by the site.

---

## 🧾 Tech stack (short)

- Python 3.x  
- FastAPI (web / API)  
- scikit-learn / PyTorch / XGBoost (depending on included models) — check `requirements.txt`  
- SQL/CSV dataset (Kaggle)  
- Docker (optional)  

---

## 📥 Dataset

Download the dataset used for training from Kaggle:

https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset?resource=download

Place the dataset file (CSV or extracted folder) into the repo — e.g. `data/` — or provide its path when running training.

---

---

## 🚀 Quick start

> Commands assume you run them from the repository root.

### 1. Clone & prepare env
```bash
git clone https://github.com/tmchhhhhhhhhhhhh/Li-ion-Battery-Degradation-Analysis.git
cd Li-ion-Battery-Degradation-Analysis
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Download the Kaggle dataset
Download and extract the dataset from Kaggle and put it into data/ (or another folder — supply the path to train.py if needed).

### 3. Train models & generate plots

Run the training script — it will train **5 models** and save artifacts and figures to `plots/`:

```bash
# Default: reads data, trains 5 models, saves outputs to `plots/`
python train.py
```
or
```bash
python3 train.py
```

### 4. Move plots into the static folder

Make generated plots available to the FastAPI frontend by moving them into static:

```bash
mkdir -p static
mv plots static/plots
```

### 5. Run the FastAPI server

Start the API + frontend (adjust the module path if your app entrypoint differs):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- Frontend site: http://127.0.0.1:8000
- API docs (Swagger): http://127.0.0.1:8000/docs

### 6. Enjoy!!! 
🔥I will be extremely excited to see your feedback!🔥
