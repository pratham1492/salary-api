from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Study Hours Pass Prediction API")

# mymodel.pkl file load kar rahe hain
try:
    model = joblib.load("mymodel.pkl")
except Exception as e:
    model = None

# Single feature input: StudyHours
class StudyData(BaseModel):
    study_hours: float

@app.get("/")
def home():
    return {"message": "Logistic Regression API is Running!"}

@app.post("/predict")
def predict(data: StudyData):
    if model is None:
        return {"error": "mymodel.pkl file not found!"}

    # Model input reshape
    features = np.array([[data.study_hours]])
    prediction = int(model.predict(features)[0])
    probs = model.predict_proba(features)[0]

    result = "Pass" if prediction == 1 else "Fail"

    return {
        "study_hours": data.study_hours,
        "result": result,
        "probability_pass": round(float(probs[1]), 4)
    }