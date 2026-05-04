import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os
import csv

# Charger le modèle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Schéma des données d'entrée
class EmployeeData(BaseModel):
    satisfaction: float
    last_evaluation: float
    projects: int
    monthly_hours: int
    tenure: int
    work_accident: int
    promoted_last_5years: int
    salary_level: int

@app.get("/")
def home():
    return {"message": "Smart Predictor API fonctionne !"}

@app.post("/predict")
def predict(data: EmployeeData):
    features = [[
        data.satisfaction, data.last_evaluation, data.projects,
        data.monthly_hours, data.tenure, data.work_accident,
        data.promoted_last_5years, data.salary_level
    ]]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    # Log dans current_data.csv pour le monitoring
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/current_data.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["satisfaction", "last_evaluation", "projects",
                             "monthly_hours", "tenure", "work_accident",
                             "promoted_last_5years", "salary_level"])
        writer.writerow(features[0])

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }