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

3. Voir les statistiques du modèle de langage :
    ```bash
    python stats_llm.py
    ```

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