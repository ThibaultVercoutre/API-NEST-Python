import pandas as pd
import numpy as np

model = "phi"

def analyze_accuracy():
    """
    Analyse la précision des classifications du LLM en comparant avec les labels POI
    """
    # Chargement des résultats
    try:
        results = pd.read_csv('llm_' + model + '_test_results.csv')
        print(f"Nombre total d'emails analysés: {len(results)}")
        
        # Création d'une colonne pour les bonnes prédictions
        # On considère que c'est une bonne prédiction si:
        # - POI-Present = 1 et llm_classification = PHISHING
        # - POI-Present = 1 et llm_classification = SPAM
        results['correct_prediction'] = (
            ((results['POI-Present'] == 1) & 
            ((results['llm_classification'] == 'PHISHING') | 
            (results['llm_classification'] == 'SPAM'))) |
            ((results['POI-Present'] == 0) & 
            (results['llm_classification'] == 'OK'))
        )
        
        # Calcul des métriques
        total_samples = len(results)
        correct_predictions = results['correct_prediction'].sum()
        accuracy = (correct_predictions / total_samples) * 100
        
        # Distribution des classifications
        classification_dist = results['llm_classification'].value_counts()
        
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
        for classification in ['PHISHING', 'SPAM', 'OK']:
            subset = results[results['llm_classification'] == classification]
            if len(subset) > 0:
                correct = subset['correct_prediction'].sum()
                accuracy = (correct / len(subset)) * 100
                print(f"\nPour la classification {classification}:")
                print(f"Nombre total: {len(subset)}")
                print(f"Prédictions correctes: {correct}")
                print(f"Précision: {accuracy:.2f}%")
        
    except FileNotFoundError:
        print("Erreur: Le fichier 'llm_mistral_test_results.csv' n'a pas été trouvé.")
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