import pandas as pd
import requests
import time
import hashlib
from tqdm import tqdm
import numpy as np

model = "phi"
RAG = '_RAG'

def clean_text(text):
    """Nettoie le texte des valeurs NaN et caractères spéciaux"""
    if pd.isna(text) or isinstance(text, float):
        return ""
    return str(text).strip()

def create_hash(row):
    """Crée un hash unique pour chaque email basé sur son contenu"""
    content = f"{clean_text(row['From'])}{clean_text(row['Subject'])}{clean_text(row['Body'])}"
    return hashlib.md5(content.encode()).hexdigest()

def test_llm(email_data):
    """Teste un email avec le LLM via l'API"""
    try:
        start_time = time.time() 
        clean_data = {
            "sender": clean_text(email_data['From']),
            "subject": clean_text(email_data['Subject']),
            "body": clean_text(email_data['Body'])
        }
            
        response = requests.post(
            # 'http://46.202.131.99:8000/llm/' + model,
            'http://127.0.0.1:8000/llm/' + model,
            json=clean_data
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                'classification': result['classification'],
                'response_length': len(result.get('raw_response', '')),
                'response_time': response_time
            }
        else:
            print(f"Erreur API: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Erreur lors de la requête: {str(e)}")
        return None

def main():
    # Chargement des données sources
    print('Chargement des données...')
    data = pd.read_csv('enron_data_fraud_labeled.csv', low_memory=False)
    initial_count = len(data)
    
    print('Prétraitement des données...')
    features = ['Body', 'Subject', 'From']
    
    # Nettoyer les données
    for feature in features:
        data[feature] = data[feature].apply(clean_text)
    
    # Filtrer les lignes qui ont des données vides
    data = data[
        (data['Body'] != '') & 
        (data['Subject'] != '') & 
        (data['From'] != '')
    ]
    
    # Ne garder que les cas frauduleux
    data = data[data['POI-Present'] == 1].copy()

    # Ne garder que les 9000 premières lignes
    # data = data.head(9000)
    
    print(f"Nombre total d'emails au départ: {initial_count}")
    print(f"Nombre d'emails après filtrage des données incomplètes: {len(data)}")
    
    # Création des hashes pour chaque email
    data['hash'] = data.apply(create_hash, axis=1)
    
    # Chargement des résultats précédents s'ils existent
    try:
        results_df = pd.read_csv('llm_' + model + '_test_results' + RAG + '.csv')
        tested_hashes = set(results_df['hash'])
        print(f"Chargement de {len(tested_hashes)} résultats précédents")
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=['hash', 'From', 'Subject', 'Body', 'POI-Present', 'llm_classification', 'response_length', 'response_time'])
        tested_hashes = set()
        print("Aucun résultat précédent trouvé")
    
    # Filtrer les emails non testés
    untested_data = data[~data['hash'].isin(tested_hashes)]
    print(f"Nombre d'emails à tester: {len(untested_data)}")
    
    # Test des nouveaux emails
    new_results = []
    for _, row in tqdm(untested_data.iterrows(), total=len(untested_data)):
        result = test_llm(row)
        if result:
            new_results.append({
                'hash': row['hash'],
                'From': row['From'],
                'Subject': row['Subject'],
                'Body': row['Body'],
                'POI-Present': row['POI-Present'],
                'llm_classification': result['classification'],
                'response_length': result['response_length'],
                'response_time': result['response_time']
            })
            
        # Sauvegarde intermédiaire tous les 10 emails
        if len(new_results) % 10 == 0 and new_results:
            temp_df = pd.DataFrame(new_results)
            updated_results = pd.concat([results_df, temp_df], ignore_index=True)
            updated_results.to_csv('llm_' + model + '_test_results' + RAG + '.csv', index=False)
            print(f"\nSauvegarde intermédiaire effectuée - {len(new_results)} nouveaux résultats")
        
        # Pause entre chaque requête pour éviter de surcharger l'API
        time.sleep(1)
    
    # Sauvegarde finale
    if new_results:
        new_results_df = pd.DataFrame(new_results)
        final_results = pd.concat([results_df, new_results_df], ignore_index=True)
        final_results.to_csv('llm_' + model + '_test_results' + RAG + '.csv', index=False)
        
        # Analyse des résultats
        total = len(final_results)
        spam_count = len(final_results[final_results['llm_classification'] == 'SPAM'])
        phishing_count = len(final_results[final_results['llm_classification'] == 'PHISHING'])
        ok_count = len(final_results[final_results['llm_classification'] == 'OK'])
        
        print("\nRésultats finaux:")
        print(f"Total d'emails testés: {total}")
        print(f"SPAM: {spam_count} ({spam_count/total*100:.1f}%)")
        print(f"PHISHING: {phishing_count} ({phishing_count/total*100:.1f}%)")
        print(f"OK: {ok_count} ({ok_count/total*100:.1f}%)")

if __name__ == "__main__":
    main()