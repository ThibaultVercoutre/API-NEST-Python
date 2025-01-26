from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
import torch
from fastapi.middleware.cors import CORSMiddleware
import gc

# Optimisations mémoire
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Load Phi-2 model and tokenizer 
model_name = "./phi2_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu",  # Force l'utilisation du CPU
    low_cpu_mem_usage=True,
    max_memory={0: "6GB"}  # On alloue 6GB max pour garder de la marge
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

# Ajout d'un cache simple
from functools import lru_cache
@lru_cache(maxsize=1000)  # Cache les 1000 dernières requêtes

class EmailRequest(BaseModel):
    sender: str
    subject: str
    body: str

@app.post("/llm")
async def classify_email(email: EmailRequest):
    try:
        prompt = f"""
        Classify this email as SPAM, PHISHING, or OK:
        Sender: {email.sender}
        Subject: {email.subject}
        Body: {email.body}
        Classification:
        """

        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                num_beams=1  # Beam search désactivé pour plus de rapidité
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        classification = response.strip().split()[0].upper()

        valid_classifications = ["SPAM", "PHISHING", "OK"]
        if classification not in valid_classifications:
            classification = "OK"

        # Nettoyage mémoire explicite
        del outputs
        del inputs
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return {"classification": classification}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=2)