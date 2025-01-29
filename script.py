from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Email Classification API")

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

@app.post("/llm/phi")
async def classify_email(email: EmailRequest):
    try:
        t = time.time()
        print("Sending request to Ollama...")
        
        prompt = f"""You are a mail classifier. Your task is to classify the following email as either SPAM, PHISHING, or OK.
            Rules:
            - Only answer with one word: SPAM, PHISHING, or OK
            - If it looks like a scam or malicious link, classify as PHISHING
            - If it's unwanted commercial email, classify as SPAM
            - If it seems legitimate, classify as OK

            Email:
            From: {email.sender}
            Subject: {email.subject}
            Body: {email.body}

            Your classification:"""

        response = requests.post('http://localhost:11434/api/generate', 
            json={
                "model": "phi",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_k": 3,
                    "num_predict": 5
                }
            })
        
        if response.status_code == 200:
            result = response.json()
            raw_response = result['response'].strip().upper()
            print(f"Réponse brute: '{raw_response}'")
            
            valid_classifications = ["SPAM", "PHISHING", "OK"]
            classification = next((c for c in valid_classifications if c in raw_response), "UNKNOWN")
            
            print(f"Classification finale: {classification}")
            print(f"Temps d'exécution: {time.time() - t:.2f} secondes")
            
            return {"classification": classification, "raw_response": raw_response}
        else:
            raise HTTPException(status_code=500, detail="Erreur de l'API Ollama")

    except Exception as e:
        print(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/mistral")
async def classify_email_mistral(email: EmailRequest):
    try:
        t = time.time()
        print("Sending request to Ollama (Mistral)...")
        
        prompt = f"""You are a mail classifier. Your task is to classify the following email as either SPAM, PHISHING, or OK.
            Rules:
            - Only answer with one word: SPAM, PHISHING, or OK
            - If it looks like a scam or malicious link, classify as PHISHING
            - If it's unwanted commercial email, classify as SPAM
            - If it seems legitimate, classify as OK
            Email:
            From: {email.sender}
            Subject: {email.subject}
            Body: {email.body}
            Your classification:"""

        response = requests.post('http://localhost:11434/api/generate', 
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_k": 3,
                    "num_predict": 5
                }
            })
        
        if response.status_code == 200:
            result = response.json()
            raw_response = result['response'].strip().upper()
            print(f"Réponse brute: '{raw_response}'")
            
            valid_classifications = ["SPAM", "PHISHING", "OK"]
            classification = next((c for c in valid_classifications if c in raw_response), "UNKNOWN")
            
            print(f"Classification finale: {classification}")
            print(f"Temps d'exécution: {time.time() - t:.2f} secondes")
            
            return {"classification": classification, "raw_response": raw_response}
        else:
            raise HTTPException(status_code=500, detail="Erreur de l'API Ollama")
    except Exception as e:
        print(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=2)