# Système de Détection de Fraude par Email

## 📋 Description

Ce projet combine des approches d'apprentissage automatique et de modèles de langage pour détecter les emails frauduleux à partir du jeu de données Enron. Il utilise à la fois des techniques de classification ML traditionnelles et une analyse basée sur des modèles de langage pour identifier les communications potentiellement frauduleuses.

## 🔧 Prérequis

- Python 3.8+
- Ollama installé localement
- Modèles de langage : phi, phi-3, mistral:7b, deepseak-r1:8b
- Jeu de données : `enron_data_fraud_labeled.csv`

## 🚀 Installation

1. Cloner le dépôt
2. Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```
3. Démarrer Ollama :
    ```bash
    ollama serve
    ```
4. Télécharger le modèle de langage :
    ```bash
    ollama pull phi-3
    ```

## 💻 Utilisation


### API Ollama Documentation

#### 1. Lancement de l'API
```bash
python script.py
```
L'API sera accessible sur `http://localhost:8000` ou `http://127.0.0.1:8000`

#### 2. Points d'accès (Endpoints)

**Classification des emails:**
- **POST** `/llm/{model}`
- Modèles disponibles: phi, phi3, mistral, deepseek
- Exemple: `http://localhost:8000/llm/phi3`

**Vérification de santé:**
- **GET** `/health`
- Exemple: `http://localhost:8000/health`

#### 3. Format de la Requête

**Headers:**
```
Content-Type: application/json
```

**Corps de la requête (JSON):**
```json
{
    "sender": "john.doe@enron.com",
    "subject": "Meeting Tomorrow",
    "body": "Hello, Can we meet tomorrow at 2pm to discuss the project?"
}
```

#### 4. Format de la Réponse

**Réponse réussie (200 OK):**
```json
{
    "classification": "OK",
    "rate": "2",
    "raw_response": "...",
    "json_response": {
        "CLASSIFICATION": "OK",
        "RATE": "2"
    }
}
```

**Classifications possibles:**
- `OK`: Email non frauduleux
- `NONOK`: Email potentiellement frauduleux
- `UNKNOWN`: Classification incertaine

**Rate:**
- Échelle de 0 à 10
- 0: Confiance maximale en `OK`
- 10: Confiance maximale en `NONOK`

#### 5. Exemples d'Utilisation

**Avec cURL:**
```bash
curl -X POST "http://localhost:8000/llm/phi3" \
     -H "Content-Type: application/json" \
     -d '{
           "sender": "john.doe@enron.com",
           "subject": "Meeting Tomorrow",
           "body": "Hello, Can we meet tomorrow at 2pm to discuss the project?"
         }'
```

**Avec Postman:**
1. Nouvelle requête POST
2. URL: `http://localhost:8000/llm/phi3`
3. Headers: 
   - Key: `Content-Type`
   - Value: `application/json`
4. Body:
   - Sélectionner "raw"
   - Type "JSON"
   - Copier le JSON d'exemple
5. Envoyer la requête

### Entraînement & Test

1. Exécuter le classificateur (Pas très fonctionnel) :
    ```bash
    python classification.py
    ```

2. Tester le modèle de langage :
    ```bash
    python test_llm.py
    ```
    > **Note** : Nécessite `enron_data_fraud_labeled.csv` dans le répertoire racine

    Le script effectue les étapes suivantes :

    a. Récupère 9000 mails spam et 9000 mails non-spam, triés en fonction des mails déjà traités.

    b. Teste ces mails avec le modèle de langage spécifié au début du script :
    ```python
    model = "phi3"
    ```
    c. Enregistre les résultats tous les 10 mails dans un fichier `email_classification.db` contenant les informations suivantes :

    - `hash TEXT PRIMARY KEY` : Identifiant unique de l'email (pour le retrouver)
    - `sender TEXT` : Expéditeur de l'email
    - `subject TEXT` : Sujet de l'email
    - `body TEXT` : Corps de l'email
    - `poi_present INTEGER` : Indicateur de présence de point d'intérêt
    - `classification TEXT` : Classification de l'email (spam/non-spam)
    - `rate TEXT` : Taux de précision de classification donné par le LLM (s'il est sur que l'email est spam, il est de 10, s'il est sur que l'email n'est pas spam, il est de 0)
    - `response_length INTEGER` : Longueur de la réponse
    - `response_time REAL` : Temps de réponse
    - `model TEXT` : Modèle de langage utilisé


3. Voir les statistiques du modèle de langage :
    ```bash
    python stats_llm.py
    ```

    ### Résultats de l'analyse

    Voici un exemple des résultats que vous pouvez obtenir en exécutant `stats_llm.py` ici avec `phi3` comme modèle de langage :

    Nombre total d'emails analysés: 7530

    #### Résultats de l'analyse :
    - Nombre total d'échantillons: 7530
    - Nombre de prédictions correctes: 3380
    - Précision globale: 44.89%

    #### Distribution des classifications :
    - OK: 4185 (55.58%)
    - NONOK: 3021 (40.12%)
    - UNKNOWN: 324 (4.30%)

    #### Analyse détaillée :

    Pour la classification NONOK :
    - Nombre total: 3021
    - Prédictions correctes: 1439
    - Précision: 47.63%

    Pour la classification OK :
    - Nombre total: 4185
    - Prédictions correctes: 1941
    - Précision: 46.38%

    #### Analyse détaillée des POI :

    Emails contenant des POI (poi_present = 1) :
    - Nombre total: 3714
    - Correctement identifiés comme NONOK: 1439
    - Précision: 38.75%

    Emails sans POI (poi_present = 0) :
    - Nombre total: 3816
    - Correctement identifiés comme OK: 1941
    - Précision: 50.86%

## 📁 Structure du Projet

- `classification.py` : Entraînement et évaluation du classificateur ML
- `test_llm.py` : Module de test du modèle de langage
- `stats_llm.py` : Analyse des performances du modèle de langage
- `requirements.txt` : Dépendances du projet

## 📊 Résultats

Le système fournit :
- Métriques de classification ML (précision, rappel, F1)
- Résultats et statistiques de l'analyse du modèle de langage
- Comparaison des performances entre les approches

## ⚠️ Notes Importantes

1. Assurez-vous qu'Ollama est en cours d'exécution avant de tester le modèle de langage
2. Le fichier du jeu de données doit être présent dans le répertoire racine
3. Pour utiliser un jeu de données différent, mettez à jour le chemin du fichier dans le code
4. Les résultats sont stockés pour une analyse ultérieure