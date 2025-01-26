from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Charger le modèle et le tokenizer dynamiquement
model_name = "microsoft/phi-2"

print("Téléchargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Téléchargement du modèle...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Définir la fonction de classification des emails
def classify_email(sender, subject, body):
    prompt = f"Classify this email as SPAM, PHISHING, or OK:\nSender: {sender}\nSubject: {subject}\nBody: {body}\nClassification:"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)[0]
    classification = tokenizer.decode(outputs[len(inputs[0]):], skip_special_tokens=True).strip().split()[0].upper()
    return {"classification": classification}

# Exemple d'utilisation
email = classify_email(
    "test@example.com",
    "Win a free iPhone!",
    "Click here to claim your prize."
)
print(email)
