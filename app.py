from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(
    title="Student Performance AI API",
    description="يتنبأ بمستوى الطالب (H/M/L) بناءً على نشاطه في الفصل",
    version="1.0"
)

# تحميل المودل
model = joblib.load("model.pkl")

class PredictRequest(BaseModel):
    raisedHands: int        # عدد مرات رفع اليد
    visitedResources: int   # عدد زيارات المواد
    discussion: int         # عدد المشاركات في المناقشة
    absence: int            # 1 = أقل من 7 أيام غياب، 0 = أكتر من 7

class PredictResponse(BaseModel):
    predicted_class: str    # H = High, M = Medium, L = Low
    confidence: float       # نسبة الثقة في التنبؤ

@app.get("/")
def home():
    return {"message": "API شغال! جرب /docs عشان تختبر"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # تحويل البيانات لـ numpy array
    X = np.array([
        req.raisedHands,
        req.visitedResources,
        req.discussion,
        req.absence
    ]).reshape(1, -1)

    # التنبؤ
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    confidence = float(max(prob)) * 100  # أعلى احتمالية

    # تحويل الرقم للحرف
    mapping = {0: "H", 1: "M", 2: "L"}
    predicted_class = mapping.get(int(pred), "Unknown")

    return PredictResponse(
        predicted_class=predicted_class,
        confidence=round(confidence, 2)
    )