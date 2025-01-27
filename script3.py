from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
import torch
from fastapi.middleware.cors import CORSMiddleware
import gc
import time

# Optimisations mémoire
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Load Phi-2 model and tokenizer 
model_name = "./phi2_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
    max_memory={0: "6GB"}
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

@app.post("/llm")
async def classify_email(email: EmailRequest):
    try:
        t = time.time()
        print("Prompt...")
        prompt = f"""Analyze the following email and classify it strictly as either SPAM, PHISHING, or OK. Only respond with one of these three words.

        Email details:
        From: {email.sender}
        Subject: {email.subject}
        Content: {email.body}

        Classification (respond with exactly one word - SPAM, PHISHING, or OK):"""

        print(prompt)
        
        print("Tokenization...")
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(model.device)

        print("Generation...")
        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                num_beams=1,
                temperature=0.1,  # Réduire la température pour des réponses plus déterministes
                top_p=0.95,      # Contrôler la diversité des tokens générés
                early_stopping=True  # Arrêter la génération dès qu'une réponse valide est trouvée
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Réponse brute: {response}")
        
        # Nettoyage de la réponse
        response = response.strip().upper()
        valid_classifications = ["SPAM", "PHISHING", "OK"]
        
        # Si la réponse contient un des mots valides, on le prend
        classification = next((c for c in valid_classifications if c in response), "UNKNOWN")
        
        print(f"Classification finale: {classification}")
        print(f"Temps d'exécution: {time.time() - t:.2f} secondes")

        return {"classification": classification}

    except Exception as e:
        print(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=2)