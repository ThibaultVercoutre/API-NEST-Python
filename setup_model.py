from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

import os
token = os.getenv('HUGGING_FACE_TOKEN')

# Remplacez "VOTRE_TOKEN_HUGGINGFACE" par votre token personnel
login(token=token)

model_name = "mistralai/Mistral-7B-v0.1"

print("Téléchargement du modèle Mistral 7B...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
print("Modèle prêt à l'emploi !")
