import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from huggingface_hub import login

#  Authenticate with Hugging Face using your token (for API requests){generate the token with all necessary permissions}
HF_ACCESS_TOKEN = "HF_ACCESS_TOKEN"  # Replace with your actual token
login(HF_ACCESS_TOKEN)

#  Load data from your SQLite database
df = pd.read_sql("SELECT * FROM bookings", "sqlite:///C:\\Users\\Admin\\Desktop\\LLMbasedhotelbookinganalytics\\project\\project\\bookings.db")
#  Initialize the vector database using ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="bookings")

#  Load the embedding model for encoding text queries
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#  Index your data in ChromaDB (only once).
# Comment out this block after the initial indexing to avoid re-indexing on every run.
from tqdm import tqdm

for i, row in tqdm(df.iterrows(), total=len(df), desc="Indexing"):
    text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
    vector = embedding_model.encode(text).tolist()
    collection.add(
        ids=[str(i)],
        embeddings=[vector],
        metadatas={col: row[col] for col in df.columns}
    )


print("✅ Data indexed in ChromaDB!")

#  Define a function to call the Hugging Face Inference API for text generation.
def query_model_via_api(prompt, max_new_tokens=50):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {HF_ACCESS_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "do_sample": True}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # Handle potential errors
    if response.status_code != 200:
        raise Exception(f"Inference API error {response.status_code}: {response.text}")
    
    output = response.json()
    # The API may return a list of generated texts
    return output[0]["generated_text"] if isinstance(output, list) and "generated_text" in output[0] else output

#  Query function with improved filtering
def query_rag(question):
    if "revenue" in question.lower():
        # 🔹 Compute revenue from the DataFrame for July 2017
        df["arrival_date"] = pd.to_datetime(df["arrival_date"])
        revenue = df[(df["arrival_date"].dt.year == 2017) & (df["arrival_date"].dt.month == 7)]["adr"].sum()
        
        # 🔹 Use the Inference API to summarize the revenue information
        prompt = f"Total revenue in July 2017 was {revenue}. Summarize this information in a professional way."
        return query_model_via_api(prompt, max_new_tokens=50)
    
    # 🔹 If the question is not about revenue, use vector search for context
    query_vector = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=5)

    # Extract context from search results
    context = " ".join([
        f"Booking at {doc['hotel']} on {doc['arrival_date']} with price {doc['adr']}."
        for doc in results["metadatas"][0]
    ])

    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    return query_model_via_api(prompt, max_new_tokens=50)

#  Test query
if __name__ == "__main__":
    question = "What is the average price of a hotel booking?"
    print(query_rag(question))
