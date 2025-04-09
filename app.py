from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from huggingface_hub import login
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Authenticate with Hugging Face{generate the token with all necessary permissions}
HF_ACCESS_TOKEN = "HF_ACESS_TOKEN"
try:
    login(HF_ACCESS_TOKEN)
except Exception as e:
    logger.error(f"Failed to authenticate with Hugging Face: {str(e)}")

# Load data
try:
    df = pd.read_sql("SELECT * FROM bookings", "sqlite:///C:\\Users\\Admin\\Desktop\\LLMbasedhotelbookinganalytics\\project\\project\\bookings.db")
except Exception as e:
    logger.error(f"Failed to load data: {str(e)}")

# Initialize ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="bookings")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {str(e)}")

# Load embedding model
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logger.error(f"Failed to load embedding model: {str(e)}")

class QueryRequest(BaseModel):
    question: str

class AnalyticsRequest(BaseModel):
    month: int = None
    year: int = None

# Function to query Hugging Face API
def query_model_via_api(prompt, max_new_tokens=50):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {HF_ACCESS_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "do_sample": True}
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        output = response.json()
        return output[0]["generated_text"] if isinstance(output, list) and "generated_text" in output[0] else output
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to query Hugging Face API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


app = FastAPI()

#  Authenticate with Hugging Face{generate the token with all necessary permissions}
HF_ACCESS_TOKEN = "HF_ACCESS_TOKEN"
login(HF_ACCESS_TOKEN)

#  Load data
df = pd.read_sql("SELECT * FROM bookings", "sqlite:///C:\\Users\\Admin\\Desktop\\LLMbasedhotelbookinganalytics\\project\\project\\bookings.db")

#  Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="bookings")

#  Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    question: str

class AnalyticsRequest(BaseModel):
    month: int = None
    year: int = None

#  Function to query Hugging Face API
def query_model_via_api(prompt, max_new_tokens=50):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {HF_ACCESS_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "do_sample": True}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    output = response.json()
    return output[0]["generated_text"] if isinstance(output, list) and "generated_text" in output[0] else output

@app.post("/analytics")
def get_analytics(request: AnalyticsRequest = None):
    try:
        df["arrival_date"] = pd.to_datetime(df["arrival_date"])
        
        # Filter data by year and month if specified
        data = df
        if request and request.year:
            data = data[data["arrival_date"].dt.year == request.year]
        if request and request.month:
            data = data[data["arrival_date"].dt.month == request.month]

        # 🔢 Calculate various analytics
        total_revenue = data["adr"].sum()
        avg_booking_price = data["adr"].mean()
        num_bookings = len(data)
        most_popular_hotels = data["hotel"].value_counts().head(5).to_dict()
        
        monthly_revenue = data.resample('M', on='arrival_date')['adr'].sum().to_dict()
        yearly_revenue = data.resample('Y', on='arrival_date')['adr'].sum().to_dict()
        
        cancellation_rate = data['is_canceled'].mean() * 100 if 'is_canceled' in data.columns else "N/A"
        
        analytics_report = {
            "total_revenue": total_revenue,
            "average_booking_price": avg_booking_price,
            "num_bookings": num_bookings,
            "most_popular_hotels": most_popular_hotels,
            "monthly_revenue": monthly_revenue,
            "yearly_revenue": yearly_revenue,
            "cancellation_rate": cancellation_rate,
        }

        return analytics_report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import re
import pandas as pd

from datetime import datetime

@app.post("/ask")
def ask_question(request: QueryRequest):
    question = request.question.lower()

    # Extract year and month for optional filtering
    year, month = None, None
    year_match = re.search(r'\b(20\d{2})\b', question)
    month_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', question)

    if year_match:
        year = int(year_match.group())
    if month_match:
        month = pd.to_datetime(month_match.group(), format='%B').month

    # Vectorize and query ChromaDB
    query_vector = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=50)

    # Build context from metadata
    context = ""
    for metadata in results["metadatas"][0]:
        # Optional filtering based on question's implied year/month
        if (year and metadata.get("arrival_date_year") != year) or \
           (month and metadata.get("arrival_date_month") != month):
            continue

        row = [
            f"hotel: {metadata.get('hotel')}",
            f"is_canceled: {metadata.get('is_canceled')}",
            f"lead_time: {metadata.get('lead_time')}",
            f"arrival_date_year: {metadata.get('arrival_date_year')}",
            f"arrival_date_month: {metadata.get('arrival_date_month')}",
            f"arrival_date_week_number: {metadata.get('arrival_date_week_number')}",
            f"arrival_date_day_of_month: {metadata.get('arrival_date_day_of_month')}",
            f"stays_in_weekend_nights: {metadata.get('stays_in_weekend_nights')}",
            f"stays_in_week_nights: {metadata.get('stays_in_week_nights')}",
            f"adults: {metadata.get('adults')}",
            f"children: {metadata.get('children')}",
            f"babies: {metadata.get('babies')}",
            f"meal: {metadata.get('meal')}",
            f"country: {metadata.get('country')}",
            f"market_segment: {metadata.get('market_segment')}",
            f"distribution_channel: {metadata.get('distribution_channel')}",
            f"is_repeated_guest: {metadata.get('is_repeated_guest')}",
            f"previous_cancellations: {metadata.get('previous_cancellations')}",
            f"previous_bookings_not_canceled: {metadata.get('previous_bookings_not_canceled')}",
            f"reserved_room_type: {metadata.get('reserved_room_type')}",
            f"assigned_room_type: {metadata.get('assigned_room_type')}",
            f"booking_changes: {metadata.get('booking_changes')}",
            f"deposit_type: {metadata.get('deposit_type')}",
            f"agent: {metadata.get('agent')}",
            f"company: {metadata.get('company')}",
            f"days_in_waiting_list: {metadata.get('days_in_waiting_list')}",
            f"customer_type: {metadata.get('customer_type')}",
            f"adr: {metadata.get('adr')}",
            f"required_car_parking_spaces: {metadata.get('required_car_parking_spaces')}",
            f"total_of_special_requests: {metadata.get('total_of_special_requests')}",
            f"reservation_status: {metadata.get('reservation_status')}",
            f"reservation_status_date: {metadata.get('reservation_status_date')}",
        ]
        context += " | ".join(row) + "\n"

    # Final LLM prompt
    prompt = f"Use the below hotel booking data to answer the user's question accurately.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer in a single sentence:"
    response = query_model_via_api(prompt)

    return {"answer": response}



#  Health Check Endpoint
@app.get("/health")
def health_check():
    health_report = {}

    # Check database connection
    try:
        pd.read_sql("SELECT * FROM bookings LIMIT 1", "sqlite:///C:\\Users\\Admin\\Desktop\\LLMbasedhotelbookinganalytics\\project\\project\\bookings.db")
        health_report["database"] = "✅ Database connection successful"
    except Exception as e:
        health_report["database"] = f"❌ Database connection failed: {str(e)}"

    # Check ChromaDB connection
    try:
        chroma_client.get_or_create_collection(name="bookings")
        health_report["chroma_db"] = "✅ ChromaDB connection successful"
    except Exception as e:
        health_report["chroma_db"] = f"❌ ChromaDB connection failed: {str(e)}"

    # Check embedding model loading
    try:
        SentenceTransformer("all-MiniLM-L6-v2")
        health_report["embedding_model"] = "✅ Embedding model loaded successfully"
    except Exception as e:
        health_report["embedding_model"] = f"❌ Embedding model load failed: {str(e)}"

    # Check Hugging Face Inference API connection
    try:
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        headers = {"Authorization": f"Bearer {HF_ACCESS_TOKEN}"}
        response = requests.get(API_URL, headers=headers)
        
        if response.status_code == 200:
            health_report["huggingface_api"] = "✅ Hugging Face Inference API connected successfully"
        else:
            health_report["huggingface_api"] = f"❌ Hugging Face API connection failed: {response.status_code}"
    except Exception as e:
        health_report["huggingface_api"] = f"❌ Hugging Face API connection failed: {str(e)}"

    return health_report
