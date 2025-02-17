import pandas as pd
import sqlite3

model = "phi"

def analyze_accuracy():
    """
    Analyse la précision des classifications du LLM en comparant avec les labels POI
    """
    conn = sqlite3.connect('email_classifications.db')
    # Chargement des résultats
    try:
        results = pd.read_sql_query(f"SELECT * FROM results WHERE model = '{model}'", conn)
        print(f"Nombre total d'emails analysés: {len(results)}")
        
        # Création d'une colonne pour les bonnes prédictions
        results['correct_prediction'] = (
            ((results['poi_present'] == 1) & 
            (results['classification'] == 'NONOK')) |
            ((results['poi_present'] == 0) & 
            (results['classification'] == 'OK'))
        )
        
        # Calcul des métriques
        total_samples = len(results)
        correct_predictions = results['correct_prediction'].sum()
        accuracy = (correct_predictions / total_samples) * 100
        
        # Distribution des classifications
        classification_dist = results['classification'].value_counts()
        
        # Affichage des résultats
        print("\nRésultats de l'analyse:")
        print(f"Nombre total d'échantillons: {total_samples}")
        print(f"Nombre de prédictions correctes: {correct_predictions}")
        print(f"Précision globale: {accuracy:.2f}%")
        
        print("\nDistribution des classifications:")
        for class_name, count in classification_dist.items():
            percentage = (count / total_samples) * 100
            print(f"{class_name}: {count} ({percentage:.2f}%)")
            
        # Analyse détaillée par type de classification
        print("\nAnalyse détaillée:")
        for classification in ['NONOK', 'OK']:
            subset = results[results['classification'] == classification]
            if len(subset) > 0:
                correct = subset['correct_prediction'].sum()
                accuracy = (correct / len(subset)) * 100
                print(f"\nPour la classification {classification}:")
                print(f"Nombre total: {len(subset)}")
                print(f"Prédictions correctes: {correct}")
                print(f"Précision: {accuracy:.2f}%")

        # Analyse des vrais positifs et vrais négatifs
        print("\nAnalyse détaillée des POI:")
        
        # Pour les emails contenant réellement des POI (poi_present = 1)
        poi_emails = results[results['poi_present'] == 1]
        if len(poi_emails) > 0:
            correct_poi = poi_emails[poi_emails['classification'] == 'NONOK']
            print(f"\nEmails contenant des POI (poi_present = 1):")
            print(f"Nombre total: {len(poi_emails)}")
            print(f"Correctement identifiés comme NONOK: {len(correct_poi)}")
            print(f"Précision: {(len(correct_poi) / len(poi_emails) * 100):.2f}%")
        
        # Pour les emails ne contenant pas de POI (poi_present = 0)
        non_poi_emails = results[results['poi_present'] == 0]
        if len(non_poi_emails) > 0:
            correct_non_poi = non_poi_emails[non_poi_emails['classification'] == 'OK']
            print(f"\nEmails sans POI (poi_present = 0):")
            print(f"Nombre total: {len(non_poi_emails)}")
            print(f"Correctement identifiés comme OK: {len(correct_non_poi)}")
            print(f"Précision: {(len(correct_non_poi) / len(non_poi_emails) * 100):.2f}%")
        
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")

def remove_ok_classifications():
    """
    Supprime toutes les lignes où la classification est 'OK' du fichier CSV.
    """
    try:
        results = pd.read_csv('llm_' + model + '_test_results.csv')
        initial_count = len(results)
        
        # Suppression des lignes où la classification est 'OK'
        results = results[results['llm_classification'] != 'OK']
        final_count = len(results)
        
        # Sauvegarde du fichier CSV mis à jour
        results.to_csv('llm_' + model + '_test_results.csv', index=False)
        
        print(f"Nombre de lignes supprimées: {initial_count - final_count}")
        print(f"Nombre de lignes restantes: {final_count}")
        
    except FileNotFoundError:
        print("Erreur: Le fichier 'llm_mistral_test_results.csv' n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")

if __name__ == "__main__":
    # remove_ok_classifications()
    analyze_accuracy()