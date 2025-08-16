from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import os


MAX_LIFETIME = 6471.25

app = FastAPI()

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # папка с main.py
app.mount("/plots", StaticFiles(directory=BASE_DIR / "plots"), name="plots")

# Монтируем static (css, js, images)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Шаблоны
templates = Jinja2Templates(directory="templates")


models = {
    "RandomForest": joblib.load("model_RandomForest.joblib"),
    "GradientBoosting": joblib.load("model_GradientBoosting.joblib"),
    "XGBoost": joblib.load("model_XGBoost.joblib"),
    "LightGBM": joblib.load("model_LightGBM.joblib"),
    "CatBoost": joblib.load("model_CatBoost.joblib"),
}

class BatteryData(BaseModel):
    Voltage_measured: float
    Current_measured: float
    Temperature_measured: float
    Current_load: float
    Voltage_load: float

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/compare", response_class=HTMLResponse)
def compare_page(request: Request):
    return templates.TemplateResponse("compare.html", {"request": request})

@app.post("/predict")
def predict(data: BatteryData):
    features = [[
        data.Voltage_measured,
        data.Current_measured,
        data.Temperature_measured,
        data.Current_load,
        data.Voltage_load
    ]]

    results = {}

    for name, model in models.items():
        pred = float(model.predict(features)[0])
        remaining_capacity_percent = (1 - (pred / MAX_LIFETIME)) * 100
    # защита границ [0,100]
        remaining_capacity_percent = max(0.0, min(100.0, remaining_capacity_percent))
        results[name] = round(remaining_capacity_percent, 2)

    return JSONResponse({"predictions": results})

