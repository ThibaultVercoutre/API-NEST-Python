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
       start = text.find("{")
       end = text.rfind("}")
       if start != -1 and end != -1:
           json_str = text[start:end + 1]
           return json.loads(json_str)
   except json.JSONDecodeError:
       pass
   
   classification = "UNKNOWN"
   rate = "UNKNOWN"
   
   if "CLASSIFICATION:" in text:
       classification = text.split("CLASSIFICATION:")[1].strip().split()[0]
   
   if "RATE:" in text:
       rate = text.split("RATE:")[1].strip().split()[0]

   return {
       "CLASSIFICATION": classification,
       "RATE": rate
   }

@app.post("/llm/{model}")
async def classify_email(model: str, email: EmailRequest):
   try:
       t = time.time()
       print(f"Sending request to Ollama for {model}...")

       prompt = f"""You are a mail classifier. Classify the following email as SPAM, PHISHING, or OK.
           
           Rules:
           - Only answer with one word: SPAM, PHISHING, or OK
           - If it looks like a scam or malicious link, classify as PHISHING
           - If it's unwanted commercial email, classify as SPAM
           - If it seems legitimate, classify as OK
           
           Rate the mail from 0 to 10. 10 is sure it's a scam, 0 is sure isn't a scam.

           Please respond EXACTLY in this format:
           {{
           Classification: "SPAM" or "PHISHING" or "OK"
           Rate: 0-10
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