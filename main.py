# Main Script
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from bson import ObjectId
import bcrypt
import certifi
import uvicorn
import traceback
import shutil
import uuid
from datetime import datetime
from google.cloud import vision
from google.oauth2 import service_account
import pandas as pd
import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Initialize the FastAPI app
app = FastAPI()

# Configure CORS
origins = ["http://localhost:3000", "http://localhost:3001", "http://localhost:64349"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = "./uploads"
Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)

# DB Settings
MONGODB_CONNECTION_URL = "mongodb+srv://dbuser:111222333@cluster0.3ktcg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = AsyncIOMotorClient(MONGODB_CONNECTION_URL, tlsCAFile=certifi.where())
db = client["nvd_db"]
user_collection = db["users"]
records_collection = db["records"]
feedback_collection = db["feedback-reinforcement"]
direction_records_collection = db["preposition-game records"]
vocabulary_records_collection = db.vocabulary_records
difference_identifications_collection = db["difference_identifications_collection"]

# Global variable to store the best model
best_model = None

# Model training function
def train_model():
    global best_model
    # Load dataset
    data = pd.read_csv('dataset/personalized_data_updated.csv')    
    print(data.head())

    # Handle missing values
    data.fillna(0, inplace=True)

    # Encode categorical columns
    label_encoders = {}
    categorical_columns = ['grade', 'motivation_msg', 'time_category']
    for col in categorical_columns:
        data[col] = data[col].astype(str)
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Normalize 'time_taken'
    scaler = MinMaxScaler()
    data[['time_taken']] = scaler.fit_transform(data[['time_taken']])

    # Convert 'score_rate' into categories
    score_bins = [0, 0.4, 0.7, 1.0]
    score_labels = ['Low', 'Medium', 'High']
    data['score_rate'] = pd.cut(data['score_rate'], bins=score_bins, labels=score_labels, include_lowest=True)

    # Encode target variable
    y_encoder = LabelEncoder()
    data['score_rate'] = y_encoder.fit_transform(data['score_rate'])

    # Feature selection
    X = data[['grade', 'time_taken']]
    y = data['score_rate']

    # Add noise
    np.random.seed(42)
    X['noise'] = np.random.normal(0, 0.1, X.shape[0])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=2),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=2),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=2)
    }

    best_accuracy = 0
    print("\nModel Results:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name}: Accuracy = {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

    # Save the best model
    joblib.dump(best_model, 'best_model.pkl')
    print("\n🔹 Best Model:", best_model)
    print("🔹 Best Accuracy:", best_accuracy)
    print("\nClassification Report for Best Model:")
    print(classification_report(y_test, best_model.predict(X_test)))

# Run model training on startup
@app.on_event("startup")
async def startup_event():
    train_model()

# Prediction request model
class PredictionRequest(BaseModel):
    grade: str
    time_taken: float

# Prediction endpoint
@app.post("/predict")
async def predict_score_rate(request: PredictionRequest):
    global best_model
    if best_model is None:
        raise HTTPException(status_code=500, detail="Model not trained yet")

    # Prepare input data
    input_data = pd.DataFrame({
        'grade': [request.grade],
        'time_taken': [request.time_taken],
        'noise': [np.random.normal(0, 0.1)]  # Add noise as per training
    })

    # Encode grade (assuming label encoder was used during training)
    le_grade = LabelEncoder()
    le_grade.fit(['A', 'B', 'C', 'D', 'F'])  # Adjust based on your actual grades
    input_data['grade'] = le_grade.transform(input_data['grade'])

    # Normalize time_taken
    scaler = MinMaxScaler()
    scaler.fit([[0], [100]])  # Adjust range based on your data
    input_data[['time_taken']] = scaler.transform(input_data[['time_taken']])

    # Make prediction
    prediction = best_model.predict(input_data)[0]
    score_labels = ['Low', 'Medium', 'High']
    predicted_label = score_labels[prediction]

    return {"predicted_score_rate": predicted_label}

# Existing Endpoints
class UserSignUpRequest(BaseModel):
    guardian_name: str = Field(..., min_length=3, max_length=100)
    guardian_email: str = Field(..., min_length=10)
    guardian_contact: str = Field(..., min_length=10, max_length=15)
    child_name: str = Field(..., min_length=3, max_length=100)
    child_age: int = Field(..., ge=1, le=99)
    child_gender: str
    password: str = Field(..., min_length=8, max_length=100)
    vocabulary: int = Field(0, ge=0, description="Vocabulary score, default is 0")
    identify_difference: int = Field(0, ge=0, description="Identify difference score, default is 0")

@app.post("/signup")
async def sign_up(data: UserSignUpRequest):
    try:
        user_data = {
            "guardian_name": data.guardian_name,
            "guardian_email": data.guardian_email,
            "guardian_contact": data.guardian_contact,
            "child_name": data.child_name,
            "child_age": data.child_age,
            "child_gender": data.child_gender,
            "password": data.password,
            "vocabulary": data.vocabulary,
            "identify_difference": data.identify_difference
        }
        await user_collection.insert_one(user_data)
        return JSONResponse(status_code=201, content={"message": "Registration successful"})
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=400, detail=str(e))

class SignInData(BaseModel):
    guardian_email: str
    password: str

@app.post("/token")
async def sign_in(data: SignInData):
    user = await user_collection.find_one({"guardian_email": data.guardian_email})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if not data.password == user['password']:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {
        "access_token": '',
        "token_type": "bearer",
        "user": {
            "_id": str(user['_id']),
            "full_name": user['child_name'],
            "email": user['guardian_email'],
            "username": user['guardian_email'],
            "contact_number": user['guardian_contact'],
            "vocabulary": user.get('vocabulary', 0),
            "identify_difference": user.get('identify_difference', 0),
            "created_at": ""
        }
    }

@app.get("/users", response_model=List[UserSignUpRequest])
async def get_all_users():
    users = await user_collection.find().to_list(length=None)
    if not users:
        raise HTTPException(status_code=404, detail="No users found")
    return users

class UpdateScoreRequest(BaseModel):
    vocabulary: int = Field(None, ge=0, description="Vocabulary score to update")
    identify_difference: int = Field(None, ge=0, description="Identify difference score to update")

@app.put("/users/{user_id}/update_score")
async def update_score(user_id: str, data: UpdateScoreRequest):
    try:
        if data.vocabulary is None and data.identify_difference is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one score (vocabulary or identify_difference) must be provided"
            )
        
        if not ObjectId.is_valid(user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user ID format"
            )
        
        update_fields = {}
        if data.vocabulary is not None:
            update_fields["vocabulary"] = data.vocabulary
        if data.identify_difference is not None:
            update_fields["identify_difference"] = data.identify_difference

        result = await user_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_fields}
        )

        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {"message": "User score updated successfully"}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=400, detail=str(e))

# Model for Vocabulary Record
class VocabularyRecordModel(BaseModel):
    user: str
    activity: str
    type: str
    recorded_date: datetime = Field(default_factory=datetime.utcnow)
    score: float
    time_taken: int
    difficulty: int
    suggestions: Optional[List[str]]

def format_id(record):
    record["id"] = str(record.pop("_id"))
    return record

@app.post("/vocabulary-records", status_code=201)
async def create_vocabulary_record(record: VocabularyRecordModel):
    record_data = record.dict()
    result = await vocabulary_records_collection.insert_one(record_data)
    if result.inserted_id:
        return {"message": "Vocabulary record created successfully", "id": str(result.inserted_id)}
    raise HTTPException(status_code=500, detail="Failed to create vocabulary record")

@app.get("/vocabulary-records", response_model=List[dict])
async def get_all_vocabulary_records():
    records = await vocabulary_records_collection.find().to_list(length=100)
    return [format_id(record) for record in records]

@app.get("/vocabulary-records/user/{user}", response_model=dict)
async def get_vocabulary_records_by_user(user: str):
    records = await vocabulary_records_collection.find({"user": user}).sort("recorded_date", -1).to_list(length=2)
    if not records:
        raise HTTPException(status_code=404, detail="No records found for the given user")

    comparison = None
    if len(records) == 2:
        prev, latest = records[1], records[0]
        score_diff = latest["score"] - prev["score"]
        score_change = "improved" if score_diff > 0 else "lacked" if score_diff < 0 else "same"
        time_diff = prev["time_taken"] - latest["time_taken"]
        time_change = "less time taken" if time_diff > 0 else "more time taken" if time_diff < 0 else "same"
        difficulty_change = latest["difficulty"] - prev["difficulty"]
        comparison = {
            "score_change": score_change,
            "score_difference": abs(score_diff),
            "time_change": time_change,
            "time_difference": abs(time_diff),
            "difficulty_change": difficulty_change
        }

    return {
        "records": [format_id(record) for record in records],
        "comparison": comparison
    }

@app.delete("/vocabulary-records/{record_id}", status_code=200)
async def delete_vocabulary_record(record_id: str):
    result = await vocabulary_records_collection.delete_one({"_id": ObjectId(record_id)})
    if result.deleted_count:
        return {"message": "Vocabulary record deleted successfully"}
    raise HTTPException(status_code=404, detail="Vocabulary record not found")

# Placeholder QA model (assuming it was intended in your original code)
class QA(BaseModel):
    question: str

@app.post("/vocablury-endpoint")
async def recognize_im(qa: QA):
    try:
        return {"ai_response": ""}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}. See server logs for more details.")

# OCR Endpoint
class RecognizedResult(BaseModel):
    recognized_text: str

@app.post("/api/recognize-word-ocr")
async def recognize_and_parse_prescription(file: UploadFile = File(...)):
    try:
        # Step 1: Save the uploaded file
        file_path = f"{UPLOADS_DIR}/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Step 2: Initialize Google Cloud Vision client for OCR
        credentials = service_account.Credentials.from_service_account_file('ocr-key.json')
        client = vision.ImageAnnotatorClient(credentials=credentials)

        # Load the image into memory
        with open(file_path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Perform text detection
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(f'{response.error.message}')

        # Get the recognized text
        recognized_text = texts[0].description if texts else ""
        print(f"Recognized text: {recognized_text}")

        # Clean up the file
        try:
            os.remove(file_path)
        except:
            pass

        # Return the recognized text
        return {"recognized_text": recognized_text.strip()}

    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return {"error": f"Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)