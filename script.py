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
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
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
            
            # Optionnel: Enregistrer l'analyse dans la base de données
            save_analysis(current_user["email"], email.sender, email.subject, classification, rate)
            
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