import pandas as pd
import requests
import time
import hashlib
from tqdm import tqdm
import sqlite3

model = "bert"

def init_db():
   conn = sqlite3.connect('email_classifications.db')
   c = conn.cursor()
   c.execute('''CREATE TABLE IF NOT EXISTS results
                (hash TEXT PRIMARY KEY, 
                 sender TEXT,
                 subject TEXT, 
                 body TEXT,
                 poi_present INTEGER,
                 classification TEXT,
                 rate TEXT,
                 response_length INTEGER,
                 response_time REAL,
                 model TEXT)''')
   conn.commit()
   return conn

def clean_text(text):
   if pd.isna(text) or isinstance(text, float):
       return ""
   return str(text).strip()

def create_hash(row):
   content = f"{clean_text(row['From'])}{clean_text(row['Subject'])}{clean_text(row['Body'])}"
   return hashlib.md5(content.encode()).hexdigest()

def test_llm(email_data, headers):
    try:
        start_time = time.time() 
        clean_data = {
            "sender": clean_text(email_data['From']),
            "subject": clean_text(email_data['Subject']),
            "body": clean_text(email_data['Body'])
        }
        
        # Log de la taille des données
        print(f"\nTaille du corps de l'email: {len(clean_data['body'])} caractères")
            
        response = requests.post(
            'http://127.0.0.1:8000/' + model,
            json=clean_data,
            headers=headers,
            timeout=300  # Timeout après 5 minutes
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"Temps d'exécution: {response_time:.2f} secondes")
            return {
                'classification': result['classification'],
                'rate': result.get('rate', 'UNKNOWN'),  # Valeur par défaut
                'response_length': len(result.get('raw_response', '')),
                'response_time': response_time
            }
        else:
            print(f"Error {response.status_code}: {response.text}")
            time.sleep(5)  # Pause en cas d'erreur
            return None
            
    except requests.exceptions.Timeout:
        print(f"Timeout après 60 secondes - email ignoré (taille: {len(clean_data['body'])} caractères)")
        time.sleep(5)
        return None
    except Exception as e:
        print(f"Request error: {str(e)}")
        time.sleep(5)
        return None

def delete_all_results(model: str):
    conn = sqlite3.connect('email_classifications.db')
    c = conn.cursor()
    c.execute('DELETE FROM results WHERE model = ?', (model,))
    conn.commit()
    conn.close()

def main():
    # Création du compte et connexion à l'API
    register_data = {
        "email": "vercoutre.thibault@gmail.com",
        "password": "Tv210802"
    }

    try:
        # Tentative d'inscription
        register_response = requests.post(
            'http://127.0.0.1:8000/auth/register',
            json=register_data
        )
        
        # Si l'inscription échoue car l'utilisateur existe déjà, on passe à la connexion
        if register_response.status_code != 200:
            print("L'utilisateur existe déjà, tentative de connexion...")
        
        # Connexion pour obtenir le token
        login_response = requests.post(
            'http://127.0.0.1:8000/auth/login',
            data={
                "username": register_data["email"],
                "password": register_data["password"]
            }
        )

        if login_response.status_code == 200:
            token_data = login_response.json()
            headers = {
                "Authorization": f"Bearer {token_data['access_token']}"
            }
            print("Connexion réussie!")
        else:
            print("Échec de la connexion")
            return

    except Exception as e:
        print(f"Erreur lors de l'authentification: {str(e)}")
        return

    conn = init_db()
    cursor = conn.cursor()
    saved_count = 0
    
    data = pd.read_csv('enron_data_fraud_labeled.csv', low_memory=False)
    
    for feature in ['Body', 'Subject', 'From']:
        data[feature] = data[feature].apply(clean_text)
    
    data = data[data[['Body', 'Subject', 'From']].ne('').all(axis=1)]
    
    fraud_data = data[data['POI-Present'] == 1].copy()
    normal_data = data.head(9000)
    data = pd.concat([fraud_data, normal_data]).sample(frac=1, random_state=42)
    
    data['hash'] = data.apply(create_hash, axis=1)
    
    cursor.execute("SELECT hash FROM results WHERE model = ?", (model,))
    tested_hashes = {row[0] for row in cursor.fetchall()}
    
    untested_data = data[~data['hash'].isin(tested_hashes)]
    
    for _, row in tqdm(untested_data.iterrows(), total=len(untested_data)):
        result = test_llm(row, headers)
        if result:
            cursor.execute('''INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?)''', 
                (row['hash'], row['From'], row['Subject'], row['Body'], 
                row['POI-Present'], result['classification'], result['rate'],
                result['response_length'], result['response_time'], model))
            saved_count += 1
        
        if saved_count % 10 == 0:
            conn.commit()
            print(f"\nSauvegardé: {saved_count} emails")
            
            time.sleep(1) # Pause de 1 seconde
        
    conn.commit()
    print(f"\nTotal sauvegardé: {saved_count} emails")
    conn.close()

if __name__ == "__main__":
#    delete_all_results(model)
   main()