from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Student Mental Health Prediction")


# Load trained model
model = joblib.load("Mental_Health_Model.pkl")


# Input data
class StudentData(BaseModel):
    age: int
    gender: str
    country: str
    academic_level: str
    most_used_platform: str
    purpose_of_use: str
    avg_daily_usage_hours: float
    daily_unlocks: int
    study_hours: float
    physical_activity_hours: float
    sleep_hours_per_night: float
    stress_level: str


# Serve the website
@app.get("/")
def home():
    return FileResponse("index.html")


# Serve CSS
@app.get("/style.css")
def css():
    return FileResponse("style.css", media_type="text/css")


# Serve JavaScript
@app.get("/script.js")
def javascript():
    return FileResponse("script.js", media_type="application/javascript")


# Prediction endpoint
@app.post("/predict")
def predict(data: StudentData):

    country_group = "India" if data.country.lower() == "india" else "Other"

    input_data = pd.DataFrame([{
        "Age": data.age,
        "Gender": data.gender,
        "Country": data.country,
        "Academic_Level": data.academic_level,
        "Most_Used_Platform": data.most_used_platform,
        "Purpose_Of_Use": data.purpose_of_use,
        "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
        "Daily_Unlocks": data.daily_unlocks,
        "Study_Hours": data.study_hours,
        "Physical_Activity_Hours": data.physical_activity_hours,
        "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
        "Stress_Level": data.stress_level,
        "Grouped_country": country_group
    }])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_mental_health_score": float(prediction)
    }