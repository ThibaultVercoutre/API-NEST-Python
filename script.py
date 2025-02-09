from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import time
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

# Global variables
vector_store = None
embedding_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing vector store...")
    initialize_vector_store()
    yield
    # Shutdown
    if vector_store:
        vector_store.persist()

app = FastAPI(title="Enhanced Email Classification API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    sender: str
    subject: str
    body: str

models = {
    "phi": "phi",
    "phi3": "phi3",
    "mistral": "mistral",
    "deepseek": "deepseek-r1:8b"
}

def clean_text(text: str) -> str:
    """Clean text from NaN values and special characters"""
    if pd.isna(text) or isinstance(text, float):
        return ""
    return str(text).strip()

def initialize_vector_store(csv_path: str = "enron_data_fraud_labeled.csv"):
    """Initialize and populate the vector store with training emails from CSV"""
    global vector_store, embedding_model
    
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create or load the vector store
    vector_store = Chroma(
        persist_directory="enron_db",
        embedding_function=embedding_model
    )
    
    # Only populate if the store is empty
    if vector_store._collection.count() == 0:
        print("Loading training data from CSV...")
        df = pd.read_csv(csv_path, low_memory=False, on_bad_lines='skip', quoting=3)
        
        # Clean the data
        df['From'] = df['From'].apply(clean_text)
        df['Subject'] = df['Subject'].apply(clean_text)
        df['Body'] = df['Body'].apply(clean_text)
        
        # Filter out empty rows
        df = df[
            (df['Body'] != '') & 
            (df['Subject'] != '') & 
            (df['From'] != '')
        ]
        
        # Create document chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        # Prepare documents for vector store
        print("Processing emails...")
        BATCH_SIZE = 40000  # Moins que la limite de 41666
        all_texts = []
        all_metadatas = []
        
        for _, row in df.iterrows():
            email_text = f"From: {row['From']}\nSubject: {row['Subject']}\nBody: {row['Body']}"
            chunks = text_splitter.split_text(email_text)
            
            # Add metadata about fraud status
            metadata = {"poi_present": bool(row['POI-Present'])}
            for chunk in chunks:
                all_texts.append(chunk)
                all_metadatas.append(metadata)
                
                # Process in batches when we reach the batch size
                if len(all_texts) >= BATCH_SIZE:
                    print(f"Vectorizing batch of {len(all_texts)} chunks...")
                    vector_store.add_texts(texts=all_texts, metadatas=all_metadatas)
                    all_texts = []
                    all_metadatas = []
        
        # Process any remaining documents
        if all_texts:
            print(f"Vectorizing final batch of {len(all_texts)} chunks...")
            vector_store.add_texts(texts=all_texts, metadatas=all_metadatas)
        
        vector_store.persist()
        print(f"Successfully vectorized {len(df)} emails")

def find_similar_emails(email_text: str, k: int = 3) -> tuple[list[str], list[bool]]:
    """Find similar emails in the vector store and return their content and fraud status"""
    if vector_store is None:
        return [], []
    
    # Version simplifiée utilisant juste similarity_search
    results = vector_store.similarity_search(
        query=email_text,
        k=k
    )
    
    similar_texts = [doc.page_content for doc in results]
    fraud_status = [doc.metadata.get('poi_present', False) for doc in results]
    
    return similar_texts, fraud_status

def extract_json_response(text: str) -> str:
    """Extract classification from model response"""
    start = text.find("{")
    end = text.find("}")
    if start != -1 and end != -1:
        json_str = text[start:end + 1]
        return json_str.replace("\n", "").replace(" ", "")
    
    if "CLASSIFICATION:" in text:
        classification = text.split("CLASSIFICATION:")[1].strip().split()[0]
        return f'{{CLASSIFICATION:"{classification}"}}'
    
    return "{}"

@app.post("/llm/{model}")
async def classify_email(model: str, email: EmailRequest):
    try:
        t = time.time()
        print(f"Sending request to Ollama for {model}...")
        
        # Get similar emails from the vector store
        email_text = f"From: {email.sender}\nSubject: {email.subject}\nBody: {email.body}"
        similar_emails, fraud_status = find_similar_emails(email_text)
        
        # Construct context from similar emails
        context = "\n\nSimilar emails from the database:\n"
        for i, (similar_email, is_fraud) in enumerate(zip(similar_emails, fraud_status), 1):
            fraud_label = "Previously marked as suspicious" if is_fraud else "Previously marked as normal"
            context += f"\nExample {i} ({fraud_label}):\n{similar_email}\n"

        prompt = f"""You are a mail classifier enhanced with a knowledge base of similar emails. Your task is to classify the following email as either SPAM, PHISHING, or OK.
            
            Rules:
            - Only answer with one word: SPAM, PHISHING, or OK
            - If it looks like a scam or malicious link, classify as PHISHING
            - If it's unwanted commercial email, classify as SPAM
            - If it seems legitimate, classify as OK
            
            Use the following similar emails from our database as context for your decision:
            {context}

            If you have any doubt, mark it as PHISHING or SPAM. You must respond with "OK" only if you are 100% certain that it contains nothing dangerous.

            Please respond EXACTLY in this format:
            {{
            Classification: "SPAM" or "PHISHING" or "OK"
            }}

            Email to classify:
            From: {email.sender}
            Subject: {email.subject}
            Body: {email.body}

            Your classification:"""

        response = requests.post('http://localhost:11434/api/generate', 
            json={
                "model": models[model],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_k": 3
                }
            })
        
        if response.status_code == 200:
            result = response.json()
            raw_response = result['response'].strip().upper()
            json_response = extract_json_response(raw_response)
            
            valid_classifications = ["SPAM", "PHISHING", "OK"]
            classification = next((c for c in valid_classifications if c in json_response), "UNKNOWN")

            if classification == "UNKNOWN":
                classification = next((c for c in valid_classifications if c in raw_response), "UNKNOWN")
            
            print(f"Final classification: {classification}")
            print(f"Execution time: {time.time() - t:.2f} seconds")
            
            return {
                "classification": classification,
                "raw_response": raw_response,
                "json_response": json_response,
                "similar_emails_count": len(similar_emails),
                "similar_emails_fraud_count": sum(fraud_status)
            }
        else:
            raise HTTPException(status_code=500, detail="Ollama API Error")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)