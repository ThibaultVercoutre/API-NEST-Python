from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import uvicorn
import requests
import time
from fastapi.middleware.cors import CORSMiddleware
import json
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
import sqlite3
from typing import Optional
from functools import lru_cache
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(
    "cybersectony/phishing-email-detection-distilbert_v2.4.1",
    model_max_length=512  # Définir explicitement la longueur maximale
)
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "cybersectony/phishing-email-detection-distilbert_v2.4.1"
)

# Configuration de l'authentification
SECRET_KEY = "votre_cle_secrete_a_changer_en_production"  # À changer en production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 jours

# Configuration de la base de données
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root', 
    'password': '',
    'database': 'nest_db'
}

# Initialisation de l'application
app = FastAPI(title="Email Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Configuration de la sécurité
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Modèles de données
class EmailRequest(BaseModel):
    sender: str
    subject: str
    body: str

class UserIn(BaseModel):
    email: str
    password: str

class UserOut(BaseModel):
    email: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserOut

class TokenData(BaseModel):
    email: Optional[str] = None

# Initialisation de la base de données
def init_db():
    conn = sqlite3.connect(DB_CONFIG['database'])
    cursor = conn.cursor()
    
    # Création de la table utilisateurs si elle n'existe pas
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        hashed_password TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

# Fonctions utilitaires pour l'authentification
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(email):
    conn = sqlite3.connect(DB_CONFIG['database'])
    cursor = conn.cursor()
    
    cursor.execute("SELECT email, hashed_password FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {"email": user[0], "hashed_password": user[1]}
    return None

def authenticate_user(email, password):
    user = get_user(email)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

# Configuration des modèles LLM
models = {
    "phi": "phi",
    "phi3": "phi3", 
    "mistral": "mistral",
    "deepseek": "deepseek-r1:8b",
    "phishing": "phishing"
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

@lru_cache(maxsize=1000)
def analyze_segment(segment_text: str):
    """Analyse un segment avec mise en cache des résultats."""
    inputs = tokenizer(
        segment_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return predictions[0].tolist()

def segment_text(text, max_length=512, overlap=100):
    """Divise le texte en segments avec chevauchement."""
    segments = []
    positions = []  # Pour stocker la position relative de chaque segment
    
    # Tokenize tout le texte
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Calculer le pas effectif (en tenant compte du chevauchement)
    stride = max_length - 2 - overlap
    
    # Diviser en segments avec chevauchement
    for i in range(0, len(tokens), stride):
        segment_tokens = tokens[i:i + max_length - 2]
        segment_text = tokenizer.decode(segment_tokens)
        segments.append(segment_text)
        
        # Calculer la position relative (0 = début, 1 = fin)
        position = i / len(tokens)
        positions.append(position)
    
    return segments, positions

def calculate_position_weights(positions):
    """Calcule les poids basés sur la position des segments."""
    # Les segments du début et de la fin ont plus de poids
    weights = []
    for pos in positions:
        # Fonction en U : donne plus de poids au début et à la fin
        weight = 1.0 + 0.5 * (np.exp(-4 * (pos - 0.5) ** 2))
        weights.append(weight)
    
    # Normaliser les poids
    weights = np.array(weights)
    weights = weights / weights.sum()
    return weights

def predict_email(email_text):
    try:
        # Segmenter le texte avec chevauchement
        segments, positions = segment_text(email_text, overlap=100)
        print(f"Email divisé en {len(segments)} segments avec chevauchement")
        
        # Calculer les poids basés sur la position
        weights = calculate_position_weights(positions)
        print("Poids des segments:", [f"{w:.3f}" for w in weights])
        
        # Stocker les prédictions pour chaque segment
        all_predictions = []
        
        for i, (segment, weight) in enumerate(zip(segments, weights)):
            print(f"Analyse du segment {i+1}/{len(segments)} (poids: {weight:.3f})")
            
            # Utiliser le cache pour l'analyse
            predictions = analyze_segment(segment)
            
            # Appliquer le poids au segment
            weighted_predictions = [p * weight for p in predictions]
            all_predictions.append(weighted_predictions)
        
        # Sommer les prédictions pondérées
        avg_probs = [sum(x) for x in zip(*all_predictions)]
        
        # Calculer les probabilités combinées
        phishing_prob = avg_probs[1] + avg_probs[3]  # Classes phishing
        legitimate_prob = avg_probs[0] + avg_probs[2]  # Classes légitimes
        
        # Seuil de confiance
        CONFIDENCE_THRESHOLD = 0.8
        
        # Classification finale
        if phishing_prob > legitimate_prob:
            classification = "NONOK"
            confidence = phishing_prob
        else:
            classification = "OK"
            confidence = legitimate_prob
        
        # Ajustement basé sur le seuil
        if confidence < CONFIDENCE_THRESHOLD and classification == "OK":
            classification = "NONOK"  # Par précaution
        
        return {
            "classification": classification,
            "confidence": confidence,
            "details": {
                "legitimate_email": avg_probs[0],
                "phishing_url": avg_probs[1],
                "legitimate_url": avg_probs[2],
                "phishing_url_alt": avg_probs[3],
                "combined_phishing": phishing_prob,
                "combined_legitimate": legitimate_prob,
                "segments_analyzed": len(segments),
                "overlap_size": 100,
                "position_weights": [float(w) for w in weights]
            }
        }
    except Exception as e:
        print(f"Error in predict_email: {str(e)}")
        raise e

# Routes d'authentification
@app.post("/auth/register", response_model=Token)
async def register(user: UserIn):
    # Vérifier si l'utilisateur existe déjà
    existing_user = get_user(user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hasher le mot de passe et créer l'utilisateur
    hashed_password = get_password_hash(user.password)
    
    conn = sqlite3.connect(DB_CONFIG['database'])
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO users (email, hashed_password) VALUES (?, ?)",
            (user.email, hashed_password)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    finally:
        conn.close()
    
    # Générer le token JWT
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"email": user.email}
    }

@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Générer le token JWT
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"email": user["email"]}
    }

# Route pour vérifier le token (utile pour l'extension)
@app.get("/auth/me", response_model=UserOut)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {"email": current_user["email"]}


# Route pour l'analyse d'email avec BERT (protégée par authentification) , current_user: dict = Depends(get_current_user)
@app.post("/bert")
async def bert_email(email: EmailRequest):
    try:
        t = time.time()
        print("Analyzing email with BERT...")
        
        # Combine email parts into single text
        email_text = f"From: {email.sender}\nSubject: {email.subject}\nBody: {email.body}"
        
        # Tronquer le texte si nécessaire (pour éviter l'erreur de dimension)
        max_chars = 1024 * 4  # Estimation approximative pour 1024 tokens
        if len(email_text) > max_chars:
            print(f"Email trop long ({len(email_text)} chars), troncature à {max_chars} chars")
            email_text = email_text[:max_chars]
        
        # Get prediction from BERT model
        prediction = predict_email(email_text)
        print(prediction)
        
        # Utiliser directement la classification et la confiance du modèle
        classification = prediction["classification"]
        confidence = prediction["confidence"]
        
        # Convertir la confiance en rate (0-10)
        rate = round(confidence * 10, 2)
        if classification == "NONOK":
            rate = max(7, rate)  # Minimum 7 pour NONOK
        else:
            rate = min(3, rate)  # Maximum 3 pour OK
        
        response_time = time.time() - t
        
        return {
            "classification": classification,
            "rate": rate,
            "response_time": response_time,
            "response_length": len(email_text),
            "details": prediction["details"]
        }
        
    except Exception as e:
        print(f"Error in bert_email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Route pour l'analyse d'email (protégée par authentification)
@app.post("/llm/{model}")
async def classify_email(model: str, email: EmailRequest, current_user: dict = Depends(get_current_user)):
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
                    "top_k": 3,
                    "timeout": 60  # Timeout de 30 secondes pour Ollama
                }
            },
            timeout=60  # Timeout global de 60 secondes pour la requête HTTP
        )
        
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
            
            # Optionnel: Enregistrer l'analyse dans la base de données
            save_analysis(current_user["email"], email.sender, email.subject, classification, rate)
            
            return {
                "classification": classification,
                "rate": rate,
                "raw_response": raw_response,
                "json_response": json_response
            }
        else:
            print(f"Ollama API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail="Ollama API Error")

    except requests.exceptions.Timeout:
        print(f"Timeout error: Request took too long to complete")
        raise HTTPException(
            status_code=504,
            detail="Request timeout - The analysis took too long to complete"
        )
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Fonction pour enregistrer les analyses (historique)
def save_analysis(user_email, sender, subject, classification, rate):
    try:
        conn = sqlite3.connect(DB_CONFIG['database'])
        cursor = conn.cursor()
        
        # Créer la table si elle n'existe pas
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            sender TEXT,
            subject TEXT,
            classification TEXT,
            rate TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_email) REFERENCES users(email)
        )
        ''')
        
        cursor.execute(
            "INSERT INTO analyses (user_email, sender, subject, classification, rate) VALUES (?, ?, ?, ?, ?)",
            (user_email, sender, subject, classification, rate)
        )
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving analysis: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Initialisation de la base de données au démarrage
@app.on_event("startup")
async def startup_event():
    init_db()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)