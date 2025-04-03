from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from huggingface_hub import login

app = FastAPI()

#  Authenticate with Hugging Face
HF_ACCESS_TOKEN = "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"
login(HF_ACCESS_TOKEN)

#  Load data
df = pd.read_sql("SELECT * FROM bookings", "sqlite:///C:\\Users\\Admin\\Downloads\\project\\project\\bookings.db")

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

@app.post("/ask")
def ask_question(request: QueryRequest):
    question = request.question.lower()
    
    # Extract year and month from the question using regex (if available)
    year = None
    month = None

    year_match = re.search(r'\b(20\d{2})\b', question)
    month_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', question)
    
    if year_match:
        year = int(year_match.group())
    
    if month_match:
        month_name = month_match.group()
        month = pd.to_datetime(month_name, format='%B').month

    # If the question is about revenue, call get_analytics with extracted year/month
    if "revenue" in question:
        return get_analytics(AnalyticsRequest(month=month, year=year))
    
    # Otherwise, perform the embedding-based query
    query_vector = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=5)

    context = " ".join([
        f"Booking at {doc['hotel']} on {doc['arrival_date']} with price {doc['adr']}."
        for doc in results["metadatas"][0]
    ])

    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = query_model_via_api(prompt)
    return {"answer": response}

#  Health Check Endpoint
@app.get("/health")
def health_check():
    health_report = {}

    # Check database connection
    try:
        pd.read_sql("SELECT * FROM bookings LIMIT 1", "sqlite:///C:\\Users\\Admin\\Downloads\\project\\project\\bookings.db")
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
