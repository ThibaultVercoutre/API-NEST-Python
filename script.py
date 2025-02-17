from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import time
from fastapi.middleware.cors import CORSMiddleware
import json

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

models = {
   "phi": "phi",
   "phi3": "phi3", 
   "mistral": "mistral",
   "deepseek": "deepseek-r1:8b"
}

def extract_json_response(text: str) -> dict:
    try:
        # Cherche la structure JSON
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start:end + 1]
            result = json.loads(json_str)
            
            # Normalise la classification
            classification = result.get("Classification", "UNKNOWN").strip().strip('"').upper()
            if "OK" in classification:
                classification = "OK"
            elif "NON" in classification or "SPAM" in classification or "PHISH" in classification:
                classification = "NONOK"
            else:
                classification = "UNKNOWN"
                
            # Normalise le rate en entier
            rate = result.get("Rate", "UNKNOWN")
            if isinstance(rate, (int, float)):
                rate = str(int(rate))
            else:
                try:
                    # Essaie d'extraire le premier nombre
                    import re
                    numbers = re.findall(r'\d+', str(rate))
                    if numbers:
                        rate = numbers[0]
                    else:
                        rate = "UNKNOWN"
                except:
                    rate = "UNKNOWN"

            return {
                "CLASSIFICATION": classification,
                "RATE": rate
            }
            
    except:
        pass
    
    return {
        "CLASSIFICATION": "UNKNOWN",
        "RATE": "UNKNOWN"
    }

@app.post("/llm/{model}")
async def classify_email(model: str, email: EmailRequest):
   try:
       t = time.time()
       print(f"Sending request to Ollama for {model}...")

       prompt = f"""You are a mail classifier. Classify the following email as NONOK or OK.
           
           Rules:
           - Answer EXACTLY with format below
           - Only answer with one word: NONOK or OK
           - If it looks like a scam or malicious link, classify as NONOK
           - If it's unwanted commercial email, classify as NONOK
           - If it seems legitimate, classify as OK
           - NONOK (risky):
             - Score 7-10 only
             - Scams, malicious links, spam
           - OK (safe):
             - Score 0-3 only
             - Legitimate business emails
           
           Rate risk 0-10 (SINGLE INTEGER ONLY):
            0 = definitely legitimate
            10 = definitely scam

           Please respond EXACTLY in this format:
           {{
           Classification: "NONOK" or "OK"
           Rate: [single integer 0-10]
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
           
           valid_classifications = ["NONOK", "OK"]
           classification = json_response.get('CLASSIFICATION', "UNKNOWN")
           rate = json_response.get('RATE', "UNKNOWN")

           if classification == "UNKNOWN":
               classification = next((c for c in valid_classifications if c in raw_response), "UNKNOWN")
           
           print(f"Final classification: {classification}")
           print(f"Execution time: {time.time() - t:.2f} seconds")
           
           return {
               "classification": classification,
               "rate": rate,
               "raw_response": raw_response,
               "json_response": json_response
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