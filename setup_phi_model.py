from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
import os

import os
token = os.getenv('HUGGING_FACE_TOKEN')

# Login to Hugging Face
login(token=token)

# Model configuration
model_name = "microsoft/phi-2"
output_dir = "./phi2_model"
os.makedirs(output_dir, exist_ok=True)

# Load tokenizer
print("Téléchargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad token if not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("pad_token non défini. Utilisation de eos_token comme pad_token.")

# Load model with specific configurations to avoid meta device issues
print("Chargement du modèle...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    low_cpu_mem_usage=True,     # Helps with large models
    device_map="auto"           # Distribute across available devices
)

# Resize embeddings if necessary
if len(tokenizer) != model.config.vocab_size:
    print("Redimensionnement des embeddings...")
    model.resize_token_embeddings(len(tokenizer))

# Save locally
print("Sauvegarde du modèle et du tokenizer...")
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir, safe_serialization=True)

print("Le modèle et le tokenizer Phi-2 sont prêts à l'emploi.")