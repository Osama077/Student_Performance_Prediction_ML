import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import joblib

# تحميل البيانات
data = pd.read_csv("AI-Data.csv")

# حذف الأعمدة الغير ضرورية
drop_cols = [
    "gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth",
    "SectionID", "Topic", "Semester", "Relation", "ParentschoolSatisfaction",
    "ParentAnsweringSurvey", "AnnouncementsView"
]
data = data.drop(drop_cols, axis=1)

# تحويل أي عمود نصي لأرقام
for column in data.columns:
    if data[column].dtype == object:
        le = preprocessing.LabelEncoder()
        data[column] = le.fit_transform(data[column])

# المميزات والـ target
X = data.iloc[:, 0:4].values          # أول 4 أعمدة
y = data.iloc[:, 4].values            # العمود الخامس (Class)

# تدريب المودل
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# حفظ المودل
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")